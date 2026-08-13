import numpy as np
from tqdm import tqdm
import os

# Paths
metadata_path = 'data/caez5g/mobility_dataset/metadata.npz'
shard_dir     = 'data/caez5g/mobility_dataset/'
output_dir    = 'outputs/caez5g/ORU0/anchor_only_l11_l12/'
os.makedirs(output_dir, exist_ok=True)

# Parameters
ORU_IDX   = 0
TAP_START = 1602
TAP_END   = 1670      # → 68 taps
J         = 100       # neighbourhood size for TW / CT / NPR

# Load metadata
print("Loading metadata...")
meta                 = np.load(metadata_path, allow_pickle=True)
ts_anchors           = meta['ts_anchors']
pos_anchors          = meta['pos_anchors']
anchor_pos_in_needed = meta['anchor_pos_in_needed'].astype(np.int64)
shard_map            = meta['shard_map'].astype(np.int64)
pos_in_shard         = meta['pos_in_shard'].astype(np.int64)
n_shards             = int(meta['n_shards'])
shard_size           = int(meta['shard_size'])

N_anchors = len(ts_anchors)
print(f"Anchors         : {N_anchors}")
print(f"X: [{pos_anchors[:,0].min():.3f}, {pos_anchors[:,0].max():.3f}]")
print(f"Y: [{pos_anchors[:,1].min():.3f}, {pos_anchors[:,1].max():.3f}]")
print(f"Shards          : {n_shards}  (shard_size={shard_size})")

# Load anchor channel data from shards
print("\nLoading anchor channel data from shards...")

N_SC_CFR = 273        # every 12th subcarrier: 3276 // 12 = 273
N_TAPS   = TAP_END - TAP_START   # 68
N_ANT    = 4
N_DMRS   = 3

CFR = np.empty((N_anchors, N_ANT, N_SC_CFR, N_DMRS), dtype=np.complex64)
CIR = np.empty((N_anchors, N_ANT, N_TAPS,   N_DMRS), dtype=np.complex64)

needed_indices        = anchor_pos_in_needed
shard_ids             = shard_map[needed_indices]
rows_in_shard         = pos_in_shard[needed_indices]
anchor_order_by_shard = np.argsort(shard_ids, kind='stable')

current_shard_id   = -1
current_shard_data = None

for anchor_i in tqdm(anchor_order_by_shard, desc="Loading anchors from shards"):
    sh  = int(shard_ids[anchor_i])
    row = int(rows_in_shard[anchor_i])

    if sh != current_shard_id:
        shard_path         = os.path.join(shard_dir, f'ch_est_shard_{sh:03d}.npz')
        current_shard_data = np.load(shard_path)['ch_est']   # (shard_n, 4_cells, 4, 1, 3276, 3)
        current_shard_id   = sh

    # ch shape: (4, 1, 3276, 3) → remove size-1 dim → (4, 3276, 3)
    ch = current_shard_data[row, ORU_IDX][:, 0, :, :]   # (4, 3276, 3)

    CFR[anchor_i] = ch[:, ::12, :]   # (4, 273, 3)

    ch_cir = np.fft.fftshift(
        np.fft.ifft(ch.astype(np.complex128), axis=1, norm="ortho"),
        axes=1
    )
    CIR[anchor_i] = ch_cir[:, TAP_START:TAP_END, :].astype(np.complex64)   # (4, 68, 3)

del current_shard_data
print(f"CFR shape: {CFR.shape}  dtype: {CFR.dtype}")
print(f"CIR shape: {CIR.shape}  dtype: {CIR.dtype}")

# Four feature representations
print("\nBuilding feature representations...")

H_anchor          = CFR                                                # (N, 4, 273, 3)
C_cir_anchor      = CIR                                               # (N, 4,  68, 3)
C_beam_cir_anchor = np.fft.ifft(C_cir_anchor, axis=1, norm="ortho")

print(f"H_anchor:          {H_anchor.shape}")
print(f"C_cir_anchor:      {C_cir_anchor.shape}")
print(f"C_beam_cir_anchor: {C_beam_cir_anchor.shape}")

# Physical distance matrix
print("\nComputing physical distance matrix...")
diff   = pos_anchors[:, np.newaxis, :] - pos_anchors[np.newaxis, :, :]
D_phys = np.sqrt(np.sum(diff**2, axis=-1))
np.fill_diagonal(D_phys, 0.0)
print(f"D_phys: {D_phys.shape} | "
      f"min={D_phys[D_phys>0].min():.4f} m, "
      f"max={D_phys.max():.4f} m")

# L1,1 uses the arithmetic mean of Manhattan distances.
# L1,2 uses the root mean square of Manhattan distances.

def manhattan_distance_matrix(hi, hj):
    """Compute pairwise Manhattan distances between complex vectors."""
    # diff[i,j,k] = hi[i,k] - hj[j,k]
    diff = hi[:, np.newaxis, :] - hj[np.newaxis, :, :]   # (N, N, vec_dim)
    return np.sum(np.abs(diff), axis=-1)                  # (N, N) real


def compute_l11_distance_matrix(H_win, avg_axis):
    """Compute the L1,1 distance matrix."""
    N_locs = H_win.shape[0]
    n_dmrs = H_win.shape[3]   # 3 — arithmetic mean, independent of p,q

    if avg_axis == 'row':
        n_outer      = H_win.shape[1]
        outer_slices = [H_win[:, m, :, :] for m in range(n_outer)]
    else:
        n_outer      = H_win.shape[2]
        outer_slices = [H_win[:, :, k, :] for k in range(n_outer)]

    D = np.zeros((N_locs, N_locs), dtype=np.float64)

    for slc in tqdm(outer_slices, desc=f"  L_1,1 ({avg_axis})", leave=False):
        # slc: (N, F, 3) [row] or (N, 4, 3) [col]
        D_slice = np.zeros((N_locs, N_locs), dtype=np.float64)

        for d in range(n_dmrs):
            hi = slc[:, :, d].astype(np.complex128)   # (N, vec_dim)
            # Manhattan distance — L1 norm of complex difference vector
            D_slice += manhattan_distance_matrix(hi, hi)

        # arithmetic mean over 3 DMRS symbols (noise reduction step)
        D_slice /= n_dmrs
        D += D_slice

    # arithmetic mean over outer slices (q=1)
    D /= n_outer
    np.fill_diagonal(D, 0.0)
    return D


def compute_l12_distance_matrix(H_win, avg_axis):
    """Compute the L1,2 distance matrix."""
    N_locs = H_win.shape[0]
    n_dmrs = H_win.shape[3]   # 3

    if avg_axis == 'row':
        n_outer      = H_win.shape[1]
        outer_slices = [H_win[:, m, :, :] for m in range(n_outer)]
    else:
        n_outer      = H_win.shape[2]
        outer_slices = [H_win[:, :, k, :] for k in range(n_outer)]

    total_terms = n_outer * n_dmrs   # denominator inside the sqrt
    D_sq        = np.zeros((N_locs, N_locs), dtype=np.float64)

    for slc in tqdm(outer_slices, desc=f"  L_1,2 ({avg_axis})", leave=False):
        # slc: (N, F, 3) [row] or (N, 4, 3) [col]
        for d in range(n_dmrs):
            hi = slc[:, :, d].astype(np.complex128)   # (N, vec_dim)
            Dd = manhattan_distance_matrix(hi, hi)     # (N, N) scalar distances
            D_sq += Dd ** 2                            # accumulate D^2 per term

    # single sqrt over mean of all squared distances
    D = np.sqrt(D_sq / total_terms)
    np.fill_diagonal(D, 0.0)
    return D

# Metric functions
def rank_matrix(D):
    D_no_self = D.copy()
    np.fill_diagonal(D_no_self, np.inf)
    return np.argsort(np.argsort(D_no_self, axis=1), axis=1) + 1

def trustworthiness(R_phys, R_chan, J):
    N       = R_phys.shape[0]
    U_mask  = (R_chan <= J) & (R_phys > J)
    penalty = np.where(U_mask, R_phys - J, 0)
    a       = 2.0 / (N * J * (2*N - 3*J - 1))
    return 1.0 - a * penalty.sum()

def continuity(R_phys, R_chan, J):
    N       = R_phys.shape[0]
    V_mask  = (R_phys <= J) & (R_chan > J)
    penalty = np.where(V_mask, R_chan - J, 0)
    a       = 2.0 / (N * J * (2*N - 3*J - 1))
    return 1.0 - a * penalty.sum()

def kruskal_stress(D_phys, D_chan):
    idx   = np.triu_indices(D_phys.shape[0], k=1)
    d_bar = D_phys[idx]
    d     = D_chan[idx]
    lam   = np.sum(d_bar * d) / np.sum(d**2)
    ks    = np.sqrt(np.sum((d_bar - lam*d)**2) / np.sum(d_bar**2))
    return float(ks), float(lam)

def npr(R_phys, R_chan, J):
    in_phys      = R_phys <= J
    in_chan       = R_chan  <= J
    intersection = (in_phys & in_chan).sum(axis=1)
    return float(intersection.mean() / J)




def compute_all_metrics(D_phys, D_chan, J):
    R_phys = rank_matrix(D_phys)
    R_chan = rank_matrix(D_chan)
    tw = trustworthiness(R_phys, R_chan, J)
    ct = continuity(R_phys, R_chan, J)
    ks, _ = kruskal_stress(D_phys, D_chan)
    np_r = npr(R_phys, R_chan, J)
    return dict(TW=tw, CT=ct, KS=ks, NPR=np_r)

# Feature configurations
# Manhattan distance is used for the L1 aggregations.
configs = [
    (
        'Antenna-Frequency H (CFR, ORU0, anchor-only)',
        H_anchor,
        [('col', 'antenna_per_freq')],
        'metrics_H_antenna_freq_CFR_273sc_4ant_ORU0_anchoronly_l11_l12.npz',
    ),
    (
        'Antenna-Delay C (CIR 68taps, ORU0, anchor-only)',
        C_cir_anchor,
        [('row', 'delay_per_antenna'), ('col', 'antenna_per_delay')],
        'metrics_C_antenna_delay_CIR_68taps_4ant_ORU0_anchoronly_l11_l12.npz',
    ),
    (
        'Beam-Delay C_beam (CIR 68taps, ORU0, anchor-only)',
        C_beam_cir_anchor,
        [('row', 'delay_per_beam')],
        'metrics_Cbeam_beam_delay_CIR_68taps_4beam_ORU0_anchoronly_l11_l12.npz',
    ),
]

norm_labels = {
    'l11': 'Manhattan (L_{1,1})',
    'l12': 'Manhattan (L_{1,2})',
}

# Main computation loop
for cfg_label, H_win, feature_types, out_fname in configs:
    print(f"\n{'='*70}")
    print(f"  {cfg_label}")
    print(f"{'='*70}")

    save_dict = {
        'pos_anchors':  pos_anchors,
        'D_phys':       D_phys,
        'metric_names': np.array(['TW', 'CT', 'KS', 'NPR']),
        'norm_labels':  np.array(list(norm_labels.values())),
    }

    all_results = {}

    for feat_type, feat_label in feature_types:
        print(f"\n  Feature type : {feat_label}")
        feat_results = {}

        # L_{1,1}
        print(f"    Computing L_{{1,1}} ...")
        D_l11   = compute_l11_distance_matrix(H_win, avg_axis=feat_type)
        m_l11   = compute_all_metrics(D_phys, D_l11, J)
        feat_results['l11'] = m_l11
        save_dict[f'D_{feat_label}_l11']       = D_l11
        save_dict[f'metrics_{feat_label}_l11'] = np.array(list(m_l11.values()))

        # L_{1,2}
        print(f"    Computing L_{{1,2}} ...")
        D_l12   = compute_l12_distance_matrix(H_win, avg_axis=feat_type)
        m_l12   = compute_all_metrics(D_phys, D_l12, J)
        feat_results['l12'] = m_l12
        save_dict[f'D_{feat_label}_l12']       = D_l12
        save_dict[f'metrics_{feat_label}_l12'] = np.array(list(m_l12.values()))

        all_results[feat_label] = feat_results

    # Summary table
    print(f"\n{'='*110}")
    print(f"  SUMMARY — {cfg_label}")
    print(f"{'='*110}")
    for feat_type, feat_label in feature_types:
        print(f"\n  Feature: {feat_label}")
        print(f"  {'Norm':<35} {'TW':>8} {'CT':>8} {'NPR':>8} {'KS':>8}")
        print(f"  {'-'*85}")
        for nkey, nlabel in norm_labels.items():
            m = all_results[feat_label][nkey]
            print(f"  {nlabel:<35} {m['TW']:>8.4f} {m['CT']:>8.4f} "
                  f"{m['NPR']:>8.4f} {m['KS']:>8.4f} "
                  f"")

    out_path = os.path.join(output_dir, out_fname)
    np.savez(out_path, **save_dict)
    print(f"\nSaved: {out_path}")

print("\nAll done!")
