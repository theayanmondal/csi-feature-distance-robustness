import numpy as np
from tqdm import tqdm
import os

# Paths
metadata_path = 'data/caez5g/mobility_dataset/metadata.npz'
shard_dir     = 'data/caez5g/mobility_dataset/'
output_dir    = 'outputs/caez5g/ORU0/anchor_avd_magnitude/'
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

N_SC_FULL = 3276
N_SC_CFR  = 273        # every 12th subcarrier: 3276 // 12 = 273
N_TAPS    = TAP_END - TAP_START   # 68
N_ANT     = 4
N_DMRS    = 3

CFR = np.empty((N_anchors, N_ANT, N_SC_CFR, N_DMRS), dtype=np.complex64)
CIR = np.empty((N_anchors, N_ANT, N_TAPS,   N_DMRS), dtype=np.complex64)

needed_indices  = anchor_pos_in_needed
shard_ids       = shard_map[needed_indices]
rows_in_shard   = pos_in_shard[needed_indices]

anchor_order_by_shard = np.argsort(shard_ids, kind='stable')

current_shard_id   = -1
current_shard_data = None

for anchor_i in tqdm(anchor_order_by_shard, desc="Loading anchors from shards"):
    sh  = int(shard_ids[anchor_i])
    row = int(rows_in_shard[anchor_i])

    if sh != current_shard_id:
        shard_path         = os.path.join(shard_dir, f'ch_est_shard_{sh:03d}.npz')
        current_shard_data = np.load(shard_path)['ch_est']
        current_shard_id   = sh

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

# Magnitude feature representations
# Take absolute value here; all downstream computation is purely real.
print("\nBuilding magnitude feature representations...")

H_anchor          = np.abs(CFR)                                              # (N, 4, 273, 3)
C_cir_anchor      = np.abs(CIR)                                              # (N, 4,  68, 3)
C_beam_cir_anchor = np.abs(np.fft.ifft(CIR, axis=1, norm="ortho"))          # (N, 4,  68, 3)

print(f"H_anchor:          {H_anchor.shape}  dtype: {H_anchor.dtype}")
print(f"C_cir_anchor:      {C_cir_anchor.shape}  dtype: {C_cir_anchor.dtype}")
print(f"C_beam_cir_anchor: {C_beam_cir_anchor.shape}  dtype: {C_beam_cir_anchor.dtype}")

# Physical distance matrix
print("\nComputing physical distance matrix...")
diff   = pos_anchors[:, np.newaxis, :] - pos_anchors[np.newaxis, :, :]
D_phys = np.sqrt(np.sum(diff**2, axis=-1))
np.fill_diagonal(D_phys, 0.0)
print(f"D_phys: {D_phys.shape} | "
      f"min={D_phys[D_phys>0].min():.4f} m, "
      f"max={D_phys.max():.4f} m")

# Distance computation (real-valued inputs)
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
            hi = slc[:, :, d].astype(np.float64)   # (N, vec_dim), real

            # shared quantities (all real)
            ni2   = np.sum(hi**2, axis=1)           # (N,)  squared L2 norms
            inner = hi @ hi.T                        # (N, N) real dot products

            norms      = np.sqrt(ni2)                              # (N,)
            safe_norms = np.where(norms == 0.0, 1.0, norms)
            hi_n       = hi / safe_norms[:, np.newaxis]            # unit-norm rows

            inner_norm = hi_n @ hi_n.T                             # (N, N) real, in [-1, 1]
            # magnitudes are non-negative so inner_norm >= 0 always,
            # but clip for numerical safety
            inner_norm = np.clip(inner_norm, -1.0, 1.0)

            # distance formula
            if dist_type == 'euclidean':
                D2      = ni2[:, None] + ni2[None, :] - 2.0 * inner
                D_slice += np.sqrt(np.maximum(D2, 0.0))

            elif dist_type == 'norm_euclidean':
                D2      = 2.0 - 2.0 * inner_norm
                D_slice += np.sqrt(np.maximum(D2, 0.0))

            elif dist_type == 'norm_geodesic_sphere':
                D_slice += np.arccos(np.clip(inner_norm, -1.0, 1.0))

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

# Distance types (magnitude-compatible only)
dist_types = [
    'euclidean',
    'norm_euclidean',
    'norm_geodesic_sphere',
]

dist_labels = [
    'Euclidean',
    'Normalized Euclidean',
    'Geodesic on sphere S^{2D-1}',
]

# Feature configurations
configs = [
    (
        'Antenna-Frequency |H| (CFR, ORU0, anchor-only AVD, magnitude)',
        H_anchor,
        'metrics_H_antenna_freq_CFR_273sc_4ant_ORU0_anchor_avd_magnitude.npz',
        [('col', 'antenna_per_freq')]
    ),
    (
        'Antenna-Delay |C| (CIR 68taps, ORU0, anchor-only AVD, magnitude)',
        C_cir_anchor,
        'metrics_C_antenna_delay_CIR_68taps_4ant_ORU0_anchor_avd_magnitude.npz',
        [('row', 'delay_per_antenna'), ('col', 'antenna_per_delay')]
    ),
    (
        'Beam-Delay |C_beam| (CIR 68taps, ORU0, anchor-only AVD, magnitude)',
        C_beam_cir_anchor,
        'metrics_Cbeam_beam_delay_CIR_68taps_4beam_ORU0_anchor_avd_magnitude.npz',
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
        print(f"  {'Distance':<45} {'TW':>8} {'CT':>8} {'NPR':>8} {'KS':>8}")
        print(f"  {'-'*95}")
        for dt, dl in zip(dist_types, dist_labels):
            m = all_results[feat_label][dt]
            print(f"  {dl:<45} {m['TW']:>8.4f} {m['CT']:>8.4f} "
                  f"{m['NPR']:>8.4f} {m['KS']:>8.4f} "
                  f"")

    out_path = os.path.join(output_dir, out_fname)
    np.savez(out_path, **save_dict)
    print(f"\nSaved: {out_path}")

print("\nAll done!")





import numpy as np
from tqdm import tqdm
import os

# Paths
metadata_path = 'data/caez5g/mobility_dataset/metadata.npz'
shard_dir     = 'data/caez5g/mobility_dataset/'
output_dir    = 'outputs/caez5g/ORU0/anchor_rasd_magnitude/'
os.makedirs(output_dir, exist_ok=True)

# Parameters
ORU_IDX    = 0
TAP_START  = 1602
TAP_END    = 1670
J          = 100
N_WINDOW   = 1          # anchor point only — no temporal neighbours
N_DMRS     = 3
K          = N_DMRS * N_WINDOW   # 3

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
print(f"Window size     : {N_WINDOW}  (anchor only, no temporal neighbours)")
print(f"K (slots)       : {K}  (N_DMRS={N_DMRS} × N_WINDOW={N_WINDOW})")
print(f"Shards          : {n_shards}  (shard_size={shard_size})")

# Build anchor-only index array
print("\nBuilding anchor-only index array...")
all_window_needed = anchor_pos_in_needed[:, np.newaxis]   # (N_anchors, 1)
print(f"Window index array shape: {all_window_needed.shape}")

# Load anchor channel data from shards (ABSOLUTE VALUE)
print("\nLoading anchor channel data from shards (taking |.| magnitude)...")

N_SC_CFR = 273
N_TAPS   = TAP_END - TAP_START   # 68
N_ANT    = 4

CFR_win = np.empty((N_anchors, N_WINDOW, N_ANT, N_SC_CFR, N_DMRS), dtype=np.float32)
CIR_win = np.empty((N_anchors, N_WINDOW, N_ANT, N_TAPS,   N_DMRS), dtype=np.complex64)

flat_needed   = all_window_needed.ravel()
flat_shards   = shard_map[flat_needed]
flat_rows     = pos_in_shard[flat_needed]
sort_order    = np.argsort(flat_shards, kind='stable')

current_shard_id   = -1
current_shard_data = None

for flat_i in tqdm(sort_order, desc="Loading anchor samples from shards"):
    sh  = int(flat_shards[flat_i])
    row = int(flat_rows[flat_i])

    if sh != current_shard_id:
        shard_path         = os.path.join(shard_dir, f'ch_est_shard_{sh:03d}.npz')
        current_shard_data = np.load(shard_path)['ch_est']
        current_shard_id   = sh

    ch = current_shard_data[row, ORU_IDX][:, 0, :, :]   # (4, 3276, 3)

    anchor_i = flat_i // N_WINDOW   # == flat_i since N_WINDOW=1
    win_pos  = flat_i  % N_WINDOW   # always 0

    CFR_win[anchor_i, win_pos] = np.abs(ch[:, ::12, :])

    ch_cir = np.fft.fftshift(
        np.fft.ifft(ch.astype(np.complex128), axis=1, norm="ortho"),
        axes=1
    )
    CIR_win[anchor_i, win_pos] = ch_cir[:, TAP_START:TAP_END, :].astype(np.complex64)

del current_shard_data
print(f"CFR_win shape: {CFR_win.shape}  dtype: {CFR_win.dtype}")
print(f"CIR_win shape: {CIR_win.shape}  dtype: {CIR_win.dtype}")

# Feature representations
print("\nBuilding feature representations...")

def make_feature(win_data):
    """Reorder axes for feature-distance evaluation."""
    return win_data.transpose(0, 2, 3, 4, 1)

H_win     = make_feature(CFR_win)   # (N, 4, 273, 3, 1)
C_cir_win = make_feature(np.abs(CIR_win).astype(np.float32))   # (N, 4,  68, 3, 1)

C_beam_cir_win = make_feature(
    np.abs(np.fft.ifft(CIR_win, axis=2, norm="ortho")).astype(np.float32)
)   # (N, 4, 68, 3, 1)

print(f"H_win:          {H_win.shape}")
print(f"C_cir_win:      {C_cir_win.shape}")
print(f"C_beam_cir_win: {C_beam_cir_win.shape}")

# Physical distance matrix
print("\nComputing physical distance matrix...")
diff   = pos_anchors[:, np.newaxis, :] - pos_anchors[np.newaxis, :, :]
D_phys = np.sqrt(np.sum(diff**2, axis=-1))
np.fill_diagonal(D_phys, 0.0)
print(f"D_phys: {D_phys.shape} | "
      f"min={D_phys[D_phys>0].min():.4f} m, "
      f"max={D_phys.max():.4f} m")

# SqASD distance computation
def compute_sqasd_window_distance_matrix(H_feat, dist_type, avg_axis):
    """Compute RASD over corresponding temporal samples."""
    N      = H_feat.shape[0]
    n_dmrs = H_feat.shape[3]
    n_win  = H_feat.shape[4]
    Kslots = n_dmrs * n_win    # 3

    if avg_axis == 'row':
        n_outer      = H_feat.shape[1]
        outer_slices = [H_feat[:, m, :, :, :] for m in range(n_outer)]
    else:
        n_outer      = H_feat.shape[2]
        outer_slices = [H_feat[:, :, k, :, :] for k in range(n_outer)]

    D2_accum = np.zeros((N, N), dtype=np.float64)

    for slc in tqdm(outer_slices, desc=f"  {dist_type} ({avg_axis})", leave=False):
        vec_dim = slc.shape[1]
        slc_K   = slc.reshape(N, vec_dim, Kslots).transpose(0, 2, 1)  # (N, Kslots, vec_dim)

        for k in range(Kslots):
            hi = slc_K[:, k, :].astype(np.float64)   # (N, vec_dim), real non-negative

            if dist_type == 'euclidean':
                ni2       = np.sum(hi**2, axis=1)
                inner     = hi @ hi.T
                D2_accum += np.maximum(ni2[:, None] + ni2[None, :] - 2.0 * inner, 0.0)

            elif dist_type == 'norm_euclidean':
                norms     = np.linalg.norm(hi, axis=-1, keepdims=True)
                safe_norms = np.where(norms == 0.0, 1.0, norms)
                hi_n      = hi / safe_norms
                inner_norm = hi_n @ hi_n.T
                D2_accum += np.maximum(2.0 - 2.0 * inner_norm, 0.0)

            elif dist_type == 'norm_geodesic_sphere':
                norms     = np.linalg.norm(hi, axis=-1, keepdims=True)
                safe_norms = np.where(norms == 0.0, 1.0, norms)
                hi_n      = hi / safe_norms
                inner_norm = hi_n @ hi_n.T
                val       = np.clip(inner_norm, -1.0, 1.0)
                D2_accum += np.arccos(val) ** 2

            else:
                raise ValueError(f"Unknown dist_type: {dist_type}")

    D = np.sqrt(D2_accum / (Kslots * n_outer))
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
    a      = 2.0 / (N * J * (2*N - 3*J - 1))
    return 1.0 - a * penalty.sum()

def continuity(R_phys, R_chan, J):
    N      = R_phys.shape[0]
    V_mask = (R_phys <= J) & (R_chan > J)
    penalty = np.where(V_mask, R_chan - J, 0)
    a      = 2.0 / (N * J * (2*N - 3*J - 1))
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
]

dist_labels = [
    'Euclidean',
    'Normalized, Normalized Euclidean',
    'Normalized, Geodesic on sphere S^{2D-1}',
]

# Feature configurations
configs = [
    (
        'Antenna-Frequency H (CFR, ORU0, anchor-only RASD, magnitude)',
        H_win,
        'metrics_H_antenna_freq_CFR_273sc_4ant_ORU0_anchor_rasd_magnitude.npz',
        [('col', 'antenna_per_freq')]
    ),
    (
        'Antenna-Delay C (CIR 68taps, ORU0, anchor-only RASD, magnitude)',
        C_cir_win,
        'metrics_C_antenna_delay_CIR_68taps_4ant_ORU0_anchor_rasd_magnitude.npz',
        [('row', 'delay_per_antenna'), ('col', 'antenna_per_delay')]
    ),
    (
        'Beam-Delay C_beam (CIR 68taps, ORU0, anchor-only RASD, magnitude)',
        C_beam_cir_win,
        'metrics_Cbeam_beam_delay_CIR_68taps_4beam_ORU0_anchor_rasd_magnitude.npz',
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
            D_chan  = compute_sqasd_window_distance_matrix(H_feat, dt, avg_axis=feat_type)
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





import numpy as np
from tqdm import tqdm
import os

# Paths
metadata_path = 'data/caez5g/mobility_dataset/metadata.npz'
shard_dir     = 'data/caez5g/mobility_dataset/'
output_dir    = 'outputs/caez5g/ORU0/avg6_avd_magnitude/'
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
shard_size                = int(meta['shard_size'])

N_anchors = len(ts_anchors)
print(f"Anchors         : {N_anchors}")
print(f"Window size     : {N_WINDOW}  (anchor + 5 immediate temporal neighbours)")
print(f"K (slots)       : {K}  (N_DMRS={N_DMRS} × N_WINDOW={N_WINDOW})")
print(f"Shards          : {n_shards}  (shard_size={shard_size})")

# Build full needed-index array for all window samples
print("\nBuilding full window index array...")
all_window_needed = np.empty((N_anchors, N_WINDOW), dtype=np.int64)
all_window_needed[:, 0]  = anchor_pos_in_needed          # anchor itself
all_window_needed[:, 1:] = window_indices_in_needed       # 5 immediate temporal neighbours
print(f"Window index array shape: {all_window_needed.shape}")

# Load all window channel data from shards (ABSOLUTE VALUE)
# CFR: (N_anchors, N_WINDOW, 4, 273, 3) — magnitude only (float32)
# CIR: (N_anchors, N_WINDOW, 4,  68, 3) — complex, magnitude taken after feature transform
print("\nLoading all window channel data from shards (taking |.| magnitude)...")

N_SC_CFR = 273
N_TAPS   = TAP_END - TAP_START   # 68
N_ANT    = 4

CFR_win = np.empty((N_anchors, N_WINDOW, N_ANT, N_SC_CFR, N_DMRS), dtype=np.float32)
CIR_win = np.empty((N_anchors, N_WINDOW, N_ANT, N_TAPS,   N_DMRS), dtype=np.complex64)

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

    # magnitude of CFR
    CFR_win[anchor_i, win_pos] = np.abs(ch[:, ::12, :])   # (4, 273, 3)

    ch_cir = np.fft.fftshift(
        np.fft.ifft(ch.astype(np.complex128), axis=1, norm="ortho"),
        axes=1
    )
    # Keep CIR complex until after the spatial transform.
    CIR_win[anchor_i, win_pos] = ch_cir[:, TAP_START:TAP_END, :].astype(np.complex64)

del current_shard_data
print(f"CFR_win shape: {CFR_win.shape}  dtype: {CFR_win.dtype}")
print(f"CIR_win shape: {CIR_win.shape}  dtype: {CIR_win.dtype}")

# Feature representations (real-valued magnitude CSI)
# Only need H_win (CFR antenna/freq) and C_cir_win (CIR antenna/delay) and
# Beam-domain CIR is formed before taking magnitudes.
print("\nBuilding feature representations...")

def make_feature(win_data):
    """Reorder axes for feature-distance evaluation."""
    out = win_data.transpose(0, 2, 3, 4, 1)
    return out

H_win     = make_feature(CFR_win)   # (N, 4, 273, 3, 6)
C_cir_win = make_feature(np.abs(CIR_win).astype(np.float32))   # (N, 4,  68, 3, 6)

# Beam transform is applied to the complex CIR first; magnitude is taken afterwards.
C_beam_cir_win = make_feature(
    np.abs(np.fft.ifft(CIR_win, axis=2, norm="ortho")).astype(np.float32)
)   # (N, 4, 68, 3, 6)

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

# AVD distance computation over window (ABSOLUTE / REAL-VALUED)
def compute_avd_window_distance_matrix(H_feat, dist_type, avg_axis):
    """Compute AVD over corresponding temporal samples."""
    N      = H_feat.shape[0]
    n_dmrs = H_feat.shape[3]   # 3
    n_win  = H_feat.shape[4]   # 6
    Kslots = n_dmrs * n_win    # 18

    if avg_axis == 'row':
        n_outer      = H_feat.shape[1]   # 4
        outer_slices = [H_feat[:, m, :, :, :] for m in range(n_outer)]
    else:
        n_outer      = H_feat.shape[2]   # F
        outer_slices = [H_feat[:, :, k, :, :] for k in range(n_outer)]

    D = np.zeros((N, N), dtype=np.float64)

    for slc in tqdm(outer_slices, desc=f"  {dist_type} ({avg_axis})", leave=False):
        vec_dim = slc.shape[1]
        slc_K   = slc.reshape(N, vec_dim, Kslots).transpose(0, 2, 1)

        D_slice = np.zeros((N, N), dtype=np.float64)

        for k in range(Kslots):
            hi = slc_K[:, k, :].astype(np.float64)   # (N, vec_dim), real, non-negative

            ni2   = np.sum(hi**2, axis=1)        # (N,)
            inner = hi @ hi.T                      # (N, N) real

            norms      = np.linalg.norm(hi, axis=-1, keepdims=True)
            safe_norms = np.where(norms == 0.0, 1.0, norms)
            hi_n       = hi / safe_norms

            inner_norm = hi_n @ hi_n.T   # (N, N) real, in [-1, 1] (actually [0,1] since hi >= 0)

            if dist_type == 'euclidean':
                D2      = ni2[:, None] + ni2[None, :] - 2.0 * inner
                D_slice += np.sqrt(np.maximum(D2, 0.0))

            elif dist_type == 'norm_euclidean':
                D2      = 2.0 - 2.0 * inner_norm
                D_slice += np.sqrt(np.maximum(D2, 0.0))

            elif dist_type == 'norm_geodesic_sphere':
                val      = np.clip(inner_norm, -1.0, 1.0)
                D_slice += np.arccos(val)

            else:
                raise ValueError(f"Unknown dist_type: {dist_type}")

        D_slice /= Kslots
        np.fill_diagonal(D_slice, 0.0)
        D += D_slice

    D /= n_outer
    np.fill_diagonal(D, 0.0)
    return D

# Metric functions (unchanged)
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

# Distance types (3 only)
dist_types = [
    'euclidean',
    'norm_euclidean',
    'norm_geodesic_sphere',
]

dist_labels = [
    'Euclidean',
    'Normalized, Normalized Euclidean',
    'Normalized, Geodesic on sphere S^{2D-1}',
]

# Feature configurations
# Antenna per freq, Antenna per delay, Delay per antenna, Delay per beam
configs = [
    (
        'Antenna-Frequency H (CFR, ORU0, average-over-6 AVD, magnitude)',
        H_win,
        'metrics_H_antenna_freq_CFR_273sc_4ant_ORU0_avg6_avd_magnitude.npz',
        [('col', 'antenna_per_freq')]
    ),
    (
        'Antenna-Delay C (CIR 68taps, ORU0, average-over-6 AVD, magnitude)',
        C_cir_win,
        'metrics_C_antenna_delay_CIR_68taps_4ant_ORU0_avg6_avd_magnitude.npz',
        [('row', 'delay_per_antenna'), ('col', 'antenna_per_delay')]   # both kept
    ),
    (
        'Beam-Delay C_beam (CIR 68taps, ORU0, average-over-6 AVD, magnitude)',
        C_beam_cir_win,
        'metrics_Cbeam_beam_delay_CIR_68taps_4beam_ORU0_avg6_avd_magnitude.npz',
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



import numpy as np
from tqdm import tqdm
import os

# Paths
metadata_path = 'data/caez5g/mobility_dataset/metadata.npz'
shard_dir     = 'data/caez5g/mobility_dataset/'
output_dir    = 'outputs/caez5g/ORU0/avg6_rasd_magnitude/'
os.makedirs(output_dir, exist_ok=True)

# Parameters
ORU_IDX    = 0
TAP_START  = 1602
TAP_END    = 1670
J          = 100
N_WINDOW   = 6
N_DMRS     = 3
K          = N_DMRS * N_WINDOW   # 18

# Load metadata
print("Loading metadata...")
meta                     = np.load(metadata_path, allow_pickle=True)
ts_anchors               = meta['ts_anchors']
pos_anchors              = meta['pos_anchors']
anchor_pos_in_needed     = meta['anchor_pos_in_needed'].astype(np.int64)
window_indices_in_needed = meta['window_indices_in_needed'].astype(np.int64)[:, :5]
shard_map                = meta['shard_map'].astype(np.int64)
pos_in_shard             = meta['pos_in_shard'].astype(np.int64)
n_shards                 = int(meta['n_shards'])
shard_size               = int(meta['shard_size'])

N_anchors = len(ts_anchors)
print(f"Anchors         : {N_anchors}")
print(f"Window size     : {N_WINDOW}  (anchor + 5 immediate temporal neighbours)")
print(f"K (slots)       : {K}  (N_DMRS={N_DMRS} × N_WINDOW={N_WINDOW})")
print(f"Shards          : {n_shards}  (shard_size={shard_size})")

# Build full needed-index array
print("\nBuilding full window index array...")
all_window_needed = np.empty((N_anchors, N_WINDOW), dtype=np.int64)
all_window_needed[:, 0]  = anchor_pos_in_needed
all_window_needed[:, 1:] = window_indices_in_needed
print(f"Window index array shape: {all_window_needed.shape}")

# Load all window channel data from shards (ABSOLUTE VALUE)
print("\nLoading all window channel data from shards (taking |.| magnitude)...")

N_SC_CFR = 273
N_TAPS   = TAP_END - TAP_START   # 68
N_ANT    = 4

CFR_win = np.empty((N_anchors, N_WINDOW, N_ANT, N_SC_CFR, N_DMRS), dtype=np.float32)
CIR_win = np.empty((N_anchors, N_WINDOW, N_ANT, N_TAPS,   N_DMRS), dtype=np.complex64)

flat_needed   = all_window_needed.ravel()
flat_shards   = shard_map[flat_needed]
flat_rows     = pos_in_shard[flat_needed]
sort_order    = np.argsort(flat_shards, kind='stable')

current_shard_id   = -1
current_shard_data = None

for flat_i in tqdm(sort_order, desc="Loading window samples from shards"):
    sh  = int(flat_shards[flat_i])
    row = int(flat_rows[flat_i])

    if sh != current_shard_id:
        shard_path         = os.path.join(shard_dir, f'ch_est_shard_{sh:03d}.npz')
        current_shard_data = np.load(shard_path)['ch_est']
        current_shard_id   = sh

    ch = current_shard_data[row, ORU_IDX][:, 0, :, :]   # (4, 3276, 3)

    anchor_i = flat_i // N_WINDOW
    win_pos  = flat_i  % N_WINDOW

    CFR_win[anchor_i, win_pos] = np.abs(ch[:, ::12, :])

    ch_cir = np.fft.fftshift(
        np.fft.ifft(ch.astype(np.complex128), axis=1, norm="ortho"),
        axes=1
    )
    CIR_win[anchor_i, win_pos] = ch_cir[:, TAP_START:TAP_END, :].astype(np.complex64)

del current_shard_data
print(f"CFR_win shape: {CFR_win.shape}  dtype: {CFR_win.dtype}")
print(f"CIR_win shape: {CIR_win.shape}  dtype: {CIR_win.dtype}")

# Feature representations
print("\nBuilding feature representations...")

def make_feature(win_data):
    """Reorder axes for feature-distance evaluation."""
    return win_data.transpose(0, 2, 3, 4, 1)

H_win     = make_feature(CFR_win)   # (N, 4, 273, 3, 6)
C_cir_win = make_feature(np.abs(CIR_win).astype(np.float32))   # (N, 4,  68, 3, 6)

C_beam_cir_win = make_feature(
    np.abs(np.fft.ifft(CIR_win, axis=2, norm="ortho")).astype(np.float32)
)   # (N, 4, 68, 3, 6)

print(f"H_win:          {H_win.shape}")
print(f"C_cir_win:      {C_cir_win.shape}")
print(f"C_beam_cir_win: {C_beam_cir_win.shape}")

# Physical distance matrix
print("\nComputing physical distance matrix...")
diff   = pos_anchors[:, np.newaxis, :] - pos_anchors[np.newaxis, :, :]
D_phys = np.sqrt(np.sum(diff**2, axis=-1))
np.fill_diagonal(D_phys, 0.0)
print(f"D_phys: {D_phys.shape} | "
      f"min={D_phys[D_phys>0].min():.4f} m, "
      f"max={D_phys.max():.4f} m")

# SqASD distance computation
def compute_sqasd_window_distance_matrix(H_feat, dist_type, avg_axis):
    """Compute RASD over corresponding temporal samples."""
    N      = H_feat.shape[0]
    n_dmrs = H_feat.shape[3]
    n_win  = H_feat.shape[4]
    Kslots = n_dmrs * n_win    # 18

    if avg_axis == 'row':
        n_outer      = H_feat.shape[1]
        outer_slices = [H_feat[:, m, :, :, :] for m in range(n_outer)]
    else:
        n_outer      = H_feat.shape[2]
        outer_slices = [H_feat[:, :, k, :, :] for k in range(n_outer)]

    D2_accum = np.zeros((N, N), dtype=np.float64)

    for slc in tqdm(outer_slices, desc=f"  {dist_type} ({avg_axis})", leave=False):
        vec_dim = slc.shape[1]
        slc_K   = slc.reshape(N, vec_dim, Kslots).transpose(0, 2, 1)  # (N, Kslots, vec_dim)

        for k in range(Kslots):
            hi = slc_K[:, k, :].astype(np.float64)   # (N, vec_dim), real non-negative

            if dist_type == 'euclidean':
                ni2          = np.sum(hi**2, axis=1)
                inner        = hi @ hi.T
                D2_accum    += np.maximum(ni2[:, None] + ni2[None, :] - 2.0 * inner, 0.0)

            elif dist_type == 'norm_euclidean':
                norms        = np.linalg.norm(hi, axis=-1, keepdims=True)
                safe_norms   = np.where(norms == 0.0, 1.0, norms)
                hi_n         = hi / safe_norms
                inner_norm   = hi_n @ hi_n.T
                D2_accum    += np.maximum(2.0 - 2.0 * inner_norm, 0.0)

            elif dist_type == 'norm_geodesic_sphere':
                norms        = np.linalg.norm(hi, axis=-1, keepdims=True)
                safe_norms   = np.where(norms == 0.0, 1.0, norms)
                hi_n         = hi / safe_norms
                inner_norm   = hi_n @ hi_n.T
                val          = np.clip(inner_norm, -1.0, 1.0)
                D2_accum    += np.arccos(val) ** 2

            else:
                raise ValueError(f"Unknown dist_type: {dist_type}")

    D = np.sqrt(D2_accum / (Kslots * n_outer))
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
    a      = 2.0 / (N * J * (2*N - 3*J - 1))
    return 1.0 - a * penalty.sum()

def continuity(R_phys, R_chan, J):
    N      = R_phys.shape[0]
    V_mask = (R_phys <= J) & (R_chan > J)
    penalty = np.where(V_mask, R_chan - J, 0)
    a      = 2.0 / (N * J * (2*N - 3*J - 1))
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
]

dist_labels = [
    'Euclidean',
    'Normalized, Normalized Euclidean',
    'Normalized, Geodesic on sphere S^{2D-1}',
]

# Feature configurations
configs = [
    (
        'Antenna-Frequency H (CFR, ORU0, average-over-6 RASD, magnitude)',
        H_win,
        'metrics_H_antenna_freq_CFR_273sc_4ant_ORU0_avg6_rasd_magnitude.npz',
        [('col', 'antenna_per_freq')]
    ),
    (
        'Antenna-Delay C (CIR 68taps, ORU0, average-over-6 RASD, magnitude)',
        C_cir_win,
        'metrics_C_antenna_delay_CIR_68taps_4ant_ORU0_avg6_rasd_magnitude.npz',
        [('row', 'delay_per_antenna'), ('col', 'antenna_per_delay')]
    ),
    (
        'Beam-Delay C_beam (CIR 68taps, ORU0, average-over-6 RASD, magnitude)',
        C_beam_cir_win,
        'metrics_Cbeam_beam_delay_CIR_68taps_4beam_ORU0_avg6_rasd_magnitude.npz',
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
            D_chan  = compute_sqasd_window_distance_matrix(H_feat, dt, avg_axis=feat_type)
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







