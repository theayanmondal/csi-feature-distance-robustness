import numpy as np
from tqdm import tqdm
import os

# Paths
metadata_path = 'data/caez5g/mobility_dataset/metadata.npz'
shard_dir     = 'data/caez5g/mobility_dataset/'
output_dir    = 'outputs/caez5g/ORU0/avg6_avd/'
os.makedirs(output_dir, exist_ok=True)

# Parameters
ORU_IDX    = 0
TAP_START  = 1602
TAP_END    = 1670      # → 68 taps
J          = 100       # neighbourhood size for metrics
N_WINDOW   = 6    # anchor + 5 immediate temporal neighbours
N_DMRS     = 3
K          = N_DMRS * N_WINDOW   # 18 slots per outer slice (flat pool)

# Load metadata
print("Loading metadata...")
meta                     = np.load(metadata_path, allow_pickle=True)
ts_anchors               = meta['ts_anchors']
pos_anchors              = meta['pos_anchors']
anchor_pos_in_needed     = meta['anchor_pos_in_needed'].astype(np.int64)
window_indices_in_needed = meta['window_indices_in_needed'].astype(np.int64)[:, :5]  # first 5 of 99 neighbours
shard_map                = meta['shard_map'].astype(np.int64)
pos_in_shard             = meta['pos_in_shard'].astype(np.int64)
n_shards                 = int(meta['n_shards'])
shard_size               = int(meta['shard_size'])

N_anchors = len(ts_anchors)
print(f"Anchors         : {N_anchors}")
print(f"Window size     : {N_WINDOW}  (anchor + 5 immediate temporal neighbours)")
print(f"K (slots)       : {K}  (N_DMRS={N_DMRS} × N_WINDOW={N_WINDOW})")
print(f"Shards          : {n_shards}  (shard_size={shard_size})")

# Build full needed-index array for all window samples
# Shape: (N_anchors, N_WINDOW) — column 0 = anchor, columns 1..5 = 5 immediate neighbours
print("\nBuilding full window index array...")
all_window_needed = np.empty((N_anchors, N_WINDOW), dtype=np.int64)
all_window_needed[:, 0]  = anchor_pos_in_needed          # anchor itself
all_window_needed[:, 1:] = window_indices_in_needed       # 5 immediate temporal neighbours
print(f"Window index array shape: {all_window_needed.shape}")

# Load all window channel data from shards
# CFR: (N_anchors, N_WINDOW, 4, 273, 3)
# CIR: (N_anchors, N_WINDOW, 4,  68, 3)
print("\nLoading all window channel data from shards...")

N_SC_CFR = 273
N_TAPS   = TAP_END - TAP_START   # 68
N_ANT    = 4

CFR_win = np.empty((N_anchors, N_WINDOW, N_ANT, N_SC_CFR, N_DMRS), dtype=np.complex64)
CIR_win = np.empty((N_anchors, N_WINDOW, N_ANT, N_TAPS,   N_DMRS), dtype=np.complex64)

# Flatten all (anchor, window_pos) pairs, sort by shard to minimise file opens
flat_needed   = all_window_needed.ravel()                   # (N_anchors * N_WINDOW,)
flat_shards   = shard_map[flat_needed]                      # which shard
flat_rows     = pos_in_shard[flat_needed]                   # row within shard
sort_order    = np.argsort(flat_shards, kind='stable')

current_shard_id   = -1
current_shard_data = None

for flat_i in tqdm(sort_order, desc="Loading window samples from shards"):
    sh  = int(flat_shards[flat_i])
    row = int(flat_rows[flat_i])

    if sh != current_shard_id:
        shard_path         = os.path.join(shard_dir, f'ch_est_shard_{sh:03d}.npz')
        current_shard_data = np.load(shard_path)['ch_est']   # (shard_n, 4_cells, 4, 1, 3276, 3)
        current_shard_id   = sh

    # ch: (4, 1, 3276, 3) → (4, 3276, 3)
    ch = current_shard_data[row, ORU_IDX][:, 0, :, :]   # (4, 3276, 3)

    # unravel flat index back to (anchor_i, win_pos)
    anchor_i = flat_i // N_WINDOW
    win_pos  = flat_i  % N_WINDOW

    CFR_win[anchor_i, win_pos] = ch[:, ::12, :]   # (4, 273, 3)

    ch_cir = np.fft.fftshift(
        np.fft.ifft(ch.astype(np.complex128), axis=1, norm="ortho"),
        axes=1
    )
    CIR_win[anchor_i, win_pos] = ch_cir[:, TAP_START:TAP_END, :].astype(np.complex64)

del current_shard_data
print(f"CFR_win shape: {CFR_win.shape}  dtype: {CFR_win.dtype}")
print(f"CIR_win shape: {CIR_win.shape}  dtype: {CIR_win.dtype}")

# Four feature representations
# Input shape for all: (N_anchors, N_WINDOW, 4, F, 3)
# After rearranging for distance function: (N_anchors, 4, F, N_DMRS, N_WINDOW)
# = (N, antenna, freq_or_delay, n_dmrs, max_nb) — matches demo shape convention
print("\nBuilding feature representations...")

def make_feature(win_data):
    """Reorder axes for feature-distance evaluation."""
    # (N, N_WINDOW, 4, F, 3) → (N, 4, F, 3, N_WINDOW)
    out = win_data.transpose(0, 2, 3, 4, 1)
    return out   # (N, 4, F, N_DMRS=3, N_WINDOW=100)

H_win          = make_feature(CFR_win)                                          # (N, 4, 273, 3, 100)
C_cir_win      = make_feature(CIR_win)                                          # (N, 4,  68, 3, 100)
C_beam_cir_win = make_feature(np.fft.ifft(CIR_win, axis=2, norm="ortho"))

print(f"H_win:          {H_win.shape}")
print(f"C_cir_win:      {C_cir_win.shape}")
print(f"C_beam_cir_win: {C_beam_cir_win.shape}")

# Physical distance matrix (anchor positions only)
print("\nComputing physical distance matrix...")
diff   = pos_anchors[:, np.newaxis, :] - pos_anchors[np.newaxis, :, :]
D_phys = np.sqrt(np.sum(diff**2, axis=-1))
np.fill_diagonal(D_phys, 0.0)
print(f"D_phys: {D_phys.shape} | "
      f"min={D_phys[D_phys>0].min():.4f} m, "
      f"max={D_phys.max():.4f} m")

# AVD distance computation over window
def compute_avd_window_distance_matrix(H_feat, dist_type, avg_axis):
    """Compute AVD over corresponding temporal samples."""
    N      = H_feat.shape[0]
    n_dmrs = H_feat.shape[3]   # 3
    n_win  = H_feat.shape[4]   # 100
    Kslots = n_dmrs * n_win    # 18

    if avg_axis == 'row':
        n_outer      = H_feat.shape[1]   # 4
        outer_slices = [H_feat[:, m, :, :, :] for m in range(n_outer)]
        # each slice: (N, F, N_DMRS, N_WINDOW)
    else:
        n_outer      = H_feat.shape[2]   # F
        outer_slices = [H_feat[:, :, k, :, :] for k in range(n_outer)]
        # each slice: (N, 4, N_DMRS, N_WINDOW)

    D = np.zeros((N, N), dtype=np.float64)

    for slc in tqdm(outer_slices, desc=f"  {dist_type} ({avg_axis})", leave=False):
        # slc shape: (N, vec_dim, N_DMRS, N_WINDOW)
        # flatten last two dims → (N, vec_dim, K)
        # then transpose → (N, K, vec_dim)
        vec_dim = slc.shape[1]
        slc_K   = slc.reshape(N, vec_dim, Kslots).transpose(0, 2, 1)
        # slc_K: (N, K=18, vec_dim)

        D_slice = np.zeros((N, N), dtype=np.float64)

        # slot-by-slot loop over all K=18 slots
        # AVD requires sum of per-slot L2 norms:
        #   D_slice = (1/K) * sum_k sqrt( sum_f |hi(k,f) - hj(k,f)|^2 )
        # Each slot contributes one independent sqrt — cannot be collapsed
        # into a single matmul across slots.
        for k in range(Kslots):
            hi = slc_K[:, k, :].astype(np.complex128)   # (N, vec_dim)

            ni2   = np.sum(np.abs(hi)**2, axis=1)        # (N,)
            inner = hi @ hi.conj().T                      # (N, N) complex

            norms      = np.linalg.norm(hi, axis=-1, keepdims=True)
            safe_norms = np.where(norms == 0.0, 1.0, norms)
            hi_n       = hi / safe_norms

            inner_norm     = hi_n @ hi_n.conj().T
            abs_inner_norm = np.clip(np.abs(inner_norm), 0.0, 1.0)

            if dist_type == 'euclidean':
                G       = np.real(inner)
                D2      = ni2[:, None] + ni2[None, :] - 2.0 * G
                D_slice += np.sqrt(np.maximum(D2, 0.0))

            elif dist_type == 'norm_euclidean':
                Gn      = np.real(inner_norm)
                D2      = 2.0 - 2.0 * Gn
                D_slice += np.sqrt(np.maximum(D2, 0.0))

            elif dist_type == 'norm_geodesic_sphere':
                val      = np.clip(np.real(inner_norm), -1.0, 1.0)
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

        # AVD: arithmetic mean over all K=18 slots
        D_slice /= Kslots
        np.fill_diagonal(D_slice, 0.0)
        D += D_slice

    # arithmetic mean over outer slices (q=1)
    D /= n_outer
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
        'Antenna-Frequency H (CFR, ORU0, avg6-AVD)',
        H_win,
        'metrics_H_antenna_freq_CFR_273sc_4ant_ORU0_avg6_avd.npz',
        [('col', 'antenna_per_freq')]
    ),
    (
        'Antenna-Delay C (CIR 68taps, ORU0, avg6-AVD)',
        C_cir_win,
        'metrics_C_antenna_delay_CIR_68taps_4ant_ORU0_avg6_avd.npz',
        [('row', 'delay_per_antenna'), ('col', 'antenna_per_delay')]
    ),
    (
        'Beam-Delay C_beam (CIR 68taps, ORU0, avg6-AVD)',
        C_beam_cir_win,
        'metrics_Cbeam_beam_delay_CIR_68taps_4beam_ORU0_avg6_avd.npz',
        [('row', 'delay_per_beam')]
    ),
]

# Main computation loop
for cfg_label, H_feat, out_fname, feature_types in configs:
    print(f"\n{'='*70}")
    print(f"  {cfg_label}")
    print(f"{'='*70}")

    save_dict = {
        'pos_anchors':  pos_anchors,
        'D_phys':       D_phys,
        'metric_names': np.array(['TW', 'CT', 'KS', 'NPR']),
        'dist_types':   np.array(dist_types),
        'dist_labels':  np.array(dist_labels),
        'n_window':     np.int32(N_WINDOW),
        'K_slots':      np.int32(K),
    }

    all_results = {}

    for feat_type, feat_label in feature_types:
        print(f"\n  Feature type: {feat_label}")
        feat_results = {}

        for dt, dl in zip(dist_types, dist_labels):
            D_chan  = compute_avd_window_distance_matrix(H_feat, dt, avg_axis=feat_type)
            metrics = compute_all_metrics(D_phys, D_chan, J)
            feat_results[dt] = metrics
            save_dict[f'D_{feat_label}_{dt}']       = D_chan
            save_dict[f'metrics_{feat_label}_{dt}'] = np.array(list(metrics.values()))

        all_results[feat_label] = feat_results

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
