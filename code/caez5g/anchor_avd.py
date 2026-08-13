import numpy as np
from tqdm import tqdm
import os

# Paths
metadata_path = 'data/caez5g/mobility_dataset/metadata.npz'
shard_dir     = 'data/caez5g/mobility_dataset/'
output_dir    = 'outputs/caez5g/ORU0/anchor_only/'
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
# anchor_pos_in_needed[i] → index into the needed array
# shard_map[needed_idx]   → which shard file
# pos_in_shard[needed_idx]→ row within that shard
# ch_est[row, oru_idx] shape: (4, 1, 3276, 3) → take [:, 0, :, :] → (4, 3276, 3)

print("\nLoading anchor channel data from shards...")

N_SC_FULL = 3276
N_SC_CFR  = 273        # every 12th subcarrier: 3276 // 12 = 273
N_TAPS    = TAP_END - TAP_START   # 68
N_ANT     = 4
N_DMRS    = 3

CFR = np.empty((N_anchors, N_ANT, N_SC_CFR, N_DMRS), dtype=np.complex64)
CIR = np.empty((N_anchors, N_ANT, N_TAPS,   N_DMRS), dtype=np.complex64)

# Group anchors by shard to minimise repeated file opens
needed_indices  = anchor_pos_in_needed          # (N_anchors,) into needed-space
shard_ids       = shard_map[needed_indices]     # which shard each anchor lives in
rows_in_shard   = pos_in_shard[needed_indices]  # row within that shard

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

    # ch shape: (4, 1, 3276, 3) → remove the size-1 dim → (4, 3276, 3)
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

H_anchor          = CFR                                           # (N, 4, 273, 3)
C_cir_anchor      = CIR                                          # (N, 4,  68, 3)
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

# Distance computation
def compute_vector_distance_matrix(H_win, dist_type, avg_axis):
    """Compute pairwise CSI distances for the selected aggregation."""
    N_locs = H_win.shape[0]
    n_dmrs = H_win.shape[3]   # 3

    D = np.zeros((N_locs, N_locs), dtype=np.float64)

    if avg_axis == 'row':
        n_outer      = H_win.shape[1]
        outer_slices = [H_win[:, m, :, :] for m in range(n_outer)]
    else:
        n_outer      = H_win.shape[2]
        outer_slices = [H_win[:, :, k, :] for k in range(n_outer)]

    for slc in tqdm(outer_slices, desc=f"  {dist_type} ({avg_axis})", leave=False):
        # slc: (N, F, 3) [row] or (N, 4, 3) [col]
        D_slice = np.zeros((N_locs, N_locs), dtype=np.float64)

        for d in range(n_dmrs):
            hi = slc[:, :, d].astype(np.complex128)   # (N, vec_dim)

            # shared quantities
            ni2   = np.sum(np.abs(hi)**2, axis=1)     # (N,)
            inner = hi @ hi.conj().T                   # (N, N) complex

            norms      = np.linalg.norm(hi, axis=-1, keepdims=True)
            safe_norms = np.where(norms == 0.0, 1.0, norms)
            hi_n       = hi / safe_norms               # unit-norm rows

            inner_norm     = hi_n @ hi_n.conj().T
            abs_inner_norm = np.clip(np.abs(inner_norm), 0.0, 1.0)

            # distance formula
            if dist_type == 'euclidean':
                G       = np.real(inner)
                D2      = ni2[:, None] + ni2[None, :] - 2.0 * G
                D_slice += np.sqrt(np.maximum(D2, 0.0))

            elif dist_type == 'norm_euclidean':
                Gn      = np.real(inner_norm)
                D2      = 2.0 - 2.0 * Gn
                D_slice += np.sqrt(np.maximum(D2, 0.0))

            elif dist_type == 'norm_geodesic_sphere':
                val     = np.clip(np.real(inner_norm), -1.0, 1.0)
                D_slice += np.arccos(val)

            elif dist_type == 'global_phase_chordal':
                abs_inner = np.abs(inner)
                term      = 0.5 * (ni2[:, None]**2 + ni2[None, :]**2) - abs_inner**2
                D_slice  += np.sqrt(np.maximum(term, 0.0))

            elif dist_type == 'global_phase_bw':
                abs_inner = np.abs(inner)
                term      = 0.5 * (ni2[:, None] + ni2[None, :]) - abs_inner
                D_slice  += np.sqrt(np.maximum(term, 0.0))


            elif dist_type == 'norm_chordal':
                D_slice += np.sqrt(np.maximum(1.0 - abs_inner_norm**2, 0.0))

            elif dist_type == 'norm_geodesic_grass':
                D_slice += np.arccos(np.clip(abs_inner_norm, 0.0, 1.0))

            elif dist_type == 'norm_bw':
                D_slice += np.sqrt(np.maximum(1.0 - abs_inner_norm, 0.0))

            else:
                raise ValueError(f"Unknown dist_type: {dist_type}")

        D_slice /= n_dmrs
        np.fill_diagonal(D_slice, 0.0)
        D += D_slice

    D /= n_outer
    np.fill_diagonal(D, 0.0)
    return D

# Metric functions
def rank_matrix(D):
    D_no_self = D.copy()
    np.fill_diagonal(D_no_self, np.inf)
    return np.argsort(np.argsort(D_no_self, axis=1), axis=1) + 1

def trustworthiness(R_phys, R_chan, J):
    N      = R_phys.shape[0]
    U_mask = (R_chan <= J) & (R_phys > J)
    penalty = np.where(U_mask, R_phys - J, 0)
    a = 2.0 / (N * J * (2*N - 3*J - 1))
    return 1.0 - a * penalty.sum()

def continuity(R_phys, R_chan, J):
    N      = R_phys.shape[0]
    V_mask = (R_phys <= J) & (R_chan > J)
    penalty = np.where(V_mask, R_chan - J, 0)
    a = 2.0 / (N * J * (2*N - 3*J - 1))
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

# Distance types
dist_types = [
    'euclidean',
    'norm_euclidean',
    'norm_geodesic_sphere',
    'global_phase_chordal',
    'global_phase_bw',
    'norm_chordal',
    'norm_geodesic_grass',
    'norm_bw',
]

dist_labels = [
    'Euclidean',
    'Normalized, Normalized Euclidean',
    'Normalized, Geodesic on sphere S^{2D-1}',
    'Global phase, Chordal',
    'Global phase, Bur.-Was.',
    'Normalized and global phase, Chordal',
    'Normalized and global phase, Geodesic on Grass.',
    'Normalized and global phase, Bur.-Was.',
]

# Feature configurations
configs = [
    (
        'Antenna-Frequency H (CFR, ORU0, anchor-only)',
        H_anchor,
        'metrics_H_antenna_freq_CFR_273sc_4ant_ORU0_anchoronly.npz',
        [('col', 'antenna_per_freq')]
    ),
    (
        'Antenna-Delay C (CIR 68taps, ORU0, anchor-only)',
        C_cir_anchor,
        'metrics_C_antenna_delay_CIR_68taps_4ant_ORU0_anchoronly.npz',
        [('row', 'delay_per_antenna'), ('col', 'antenna_per_delay')]
    ),
    (
        'Beam-Delay C_beam (CIR 68taps, ORU0, anchor-only)',
        C_beam_cir_anchor,
        'metrics_Cbeam_beam_delay_CIR_68taps_4beam_ORU0_anchoronly.npz',
        [('row', 'delay_per_beam')]
    ),
]

# Main computation loop
for cfg_label, H_win, out_fname, feature_types in configs:
    print(f"\n{'='*70}")
    print(f"  {cfg_label}")
    print(f"{'='*70}")

    save_dict = {
        'pos_anchors':  pos_anchors,
        'D_phys':       D_phys,
        'metric_names': np.array(['TW', 'CT', 'KS', 'NPR']),
        'dist_types':   np.array(dist_types),
        'dist_labels':  np.array(dist_labels),
    }

    all_results = {}

    for feat_type, feat_label in feature_types:
        print(f"\n  Feature type: {feat_label}")
        feat_results = {}

        for dt, dl in zip(dist_types, dist_labels):
            D_chan  = compute_vector_distance_matrix(H_win, dt, avg_axis=feat_type)
            metrics = compute_all_metrics(D_phys, D_chan, J)
            feat_results[dt] = metrics
            save_dict[f'D_{feat_label}_{dt}']       = D_chan
            save_dict[f'metrics_{feat_label}_{dt}'] = np.array(list(metrics.values()))

        all_results[feat_label] = feat_results

    # Summary table
    print(f"\n{'='*110}")
    print(f"  SUMMARY — {cfg_label}")
    print(f"{'='*110}")
    for feat_type, feat_label in feature_types:
        print(f"\n  Feature: {feat_label}")
        print(f"  {'Distance':<50} {'TW':>8} {'CT':>8} {'NPR':>8} {'KS':>8}")
        print(f"  {'-'*100}")
        for dt, dl in zip(dist_types, dist_labels):
            m = all_results[feat_label][dt]
            print(f"  {dl:<50} {m['TW']:>8.4f} {m['CT']:>8.4f} "
                  f"{m['NPR']:>8.4f} {m['KS']:>8.4f} "
                  f"")

    out_path = os.path.join(output_dir, out_fname)
    np.savez(out_path, **save_dict)
    print(f"\nSaved: {out_path}")

print("\nAll done!")
