import os, json
import numpy as np
from tqdm import tqdm
import tensorflow as tf

# Paths
BASE_PATH  = 'data/dichasus'
output_dir = 'outputs/dichasus/ArrayA/anchor_only_sqasd_18taps/'
os.makedirs(output_dir, exist_ok=True)

tfrecord_files = [os.path.join(BASE_PATH, f"dichasus-cf0{i}.tfrecords") for i in range(2, 8)]
offset_files   = [os.path.join(BASE_PATH, f"reftx-offsets-dichasus-cf0{i}.json") for i in range(2, 8)]

ARRAY_A_GRID = np.array(
    [
        [6, 2, 16, 18],
        [28, 5, 10, 14],
    ],
    dtype=np.int32,
)
ARRAY_A_CHANNELS = ARRAY_A_GRID.ravel()
N_NEIGHBOURS   = 49
TARGET_ANCHORS = 4000
GRID_SIZE      = 20
SEED           = 42
J              = 100
TAP_START     = 509
TAP_STOP      = 527
N_TAPS        = TAP_STOP - TAP_START

# Feature description
feature_description = {
    "csi":       tf.io.FixedLenFeature([], tf.string,  default_value=''),
    "pos-tachy": tf.io.FixedLenFeature([], tf.string,  default_value=''),
    "time":      tf.io.FixedLenFeature([], tf.float32, default_value=0),
}

def parse_record(proto):
    record = tf.io.parse_single_example(proto, feature_description)
    csi = tf.ensure_shape(tf.io.parse_tensor(record["csi"], out_type=tf.float32), (32, 1024, 2))
    pos = tf.ensure_shape(tf.io.parse_tensor(record["pos-tachy"], out_type=tf.float64), (3,))
    return {"csi": csi, "pos": pos}

def to_complex(csi_np):
    return (csi_np[..., 0] + 1j * csi_np[..., 1]).astype(np.complex64)

def normalize_rev2_sto(csi_shifted):
    adjacent_products = (
        csi_shifted[:, 1:]
        * np.conj(csi_shifted[:, :-1])
    )
    phase_increment = np.float32(
        np.angle(np.sum(adjacent_products))
    )
    k = np.arange(
        csi_shifted.shape[-1],
        dtype=np.float32,
    )
    correction = np.exp(
        -1j * phase_increment * k
    ).astype(np.complex64)
    return (
        csi_shifted * correction[None, :]
    ).astype(np.complex64)

def apply_calibration(csi_complex, offsets):
    N          = csi_complex.shape[1]
    k          = np.arange(N, dtype=np.float32)
    sto        = np.array(offsets["sto"], dtype=np.float32)
    cpo        = np.array(offsets["cpo"], dtype=np.float32)
    sto_phase  = np.outer(sto, 2 * np.pi * k / N)
    cpo_phase  = np.outer(cpo, np.ones(N, dtype=np.float32))
    correction = np.exp(+1j * (sto_phase + cpo_phase)).astype(np.complex64)
    return (csi_complex * correction).astype(np.complex64)

def stratified_sample_exact(pos, target, grid_size, rng):
    x_edges = np.linspace(pos[:, 0].min(), pos[:, 0].max(), grid_size + 1)
    y_edges = np.linspace(pos[:, 1].min(), pos[:, 1].max(), grid_size + 1)
    cell_members = {}
    for row in range(grid_size):
        for col in range(grid_size):
            in_x = (pos[:, 0] >= x_edges[col]) & (pos[:, 0] < x_edges[col + 1])
            in_y = (pos[:, 1] >= y_edges[row]) & (pos[:, 1] < y_edges[row + 1])
            if col == grid_size - 1: in_x |= (pos[:, 0] == x_edges[col + 1])
            if row == grid_size - 1: in_y |= (pos[:, 1] == y_edges[row + 1])
            idxs = np.where(in_x & in_y)[0]
            if len(idxs) > 0:
                cell_members[(row, col)] = idxs
    n_nonempty = len(cell_members)
    fair_share  = target // n_nonempty
    selected, remaining = [], {}
    for (row, col), idxs in cell_members.items():
        k = min(fair_share, len(idxs))
        chosen = rng.choice(idxs, size=k, replace=False)
        selected.extend(chosen.tolist())
        leftover = np.setdiff1d(idxs, chosen)
        if len(leftover) > 0:
            remaining[(row, col)] = leftover
    if len(selected) < target:
        shortfall = target - len(selected)
        for (row, col), leftover in sorted(remaining.items(), key=lambda x: len(x[1]), reverse=True):
            if shortfall <= 0: break
            k = min(shortfall, len(leftover))
            chosen = rng.choice(leftover, size=k, replace=False)
            selected.extend(chosen.tolist())
            shortfall -= k
    return np.sort(np.array(selected))

def shifted_cfr_to_centered_cir(cfr_shifted):
    return np.fft.fftshift(
        np.fft.ifft(
            np.fft.ifftshift(cfr_shifted, axes=-1),
            axis=-1,
        ),
        axes=-1,
    ).astype(np.complex64)

# Load raw TFRecords, calibrate, extract Array 0
print("=" * 60)
print("Step 1: Loading and calibrating all TFRecord files...")
print("=" * 60)

file_data = []
for file_idx, (tf_file, off_file) in enumerate(zip(tfrecord_files, offset_files)):
    fname = os.path.basename(tf_file)
    with open(off_file) as f:
        offsets = json.load(f)
    ds = tf.data.TFRecordDataset(tf_file)
    ds = ds.map(parse_record, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
    file_csi, file_pos = [], []
    for sample in tqdm(ds, desc=fname):
        csi_np  = sample["csi"].numpy()
        csi_c   = to_complex(csi_np)
        csi_c = np.fft.fftshift(csi_c, axes=-1)
        csi_rev2 = normalize_rev2_sto(csi_c)
        csi_cal = apply_calibration(csi_rev2, offsets)
        file_csi.append(csi_cal[ARRAY_A_CHANNELS, :])
        file_pos.append(sample["pos"].numpy()[:2])
    file_data.append({
        "file_idx": file_idx,
        "n":        len(file_csi),
        "csi":      np.array(file_csi, dtype=np.complex64),
        "pos":      np.array(file_pos, dtype=np.float64),
    })
    print(f"  {fname}: {file_data[-1]['n']} samples")

# Stratified sampling → 4000 anchors
print("\n" + "=" * 60)
print("Step 2: Building eligible pool and sampling anchors...")
print("=" * 60)

elig_pos, elig_file_idx, elig_local = [], [], []
for d in file_data:
    max_anch = d["n"] - 1 - N_NEIGHBOURS
    if max_anch < 0: continue
    n_elig = max_anch + 1
    elig_pos.extend(d["pos"][:n_elig].tolist())
    elig_file_idx.extend([d["file_idx"]] * n_elig)
    elig_local.extend(list(range(n_elig)))
elig_pos      = np.array(elig_pos,      dtype=np.float64)
elig_file_idx = np.array(elig_file_idx, dtype=np.int32)
elig_local    = np.array(elig_local,    dtype=np.int32)

rng             = np.random.default_rng(seed=SEED)
sampled         = stratified_sample_exact(elig_pos, TARGET_ANCHORS, GRID_SIZE, rng)
anchor_file_idx = elig_file_idx[sampled]
anchor_local    = elig_local[sampled]
pos_anchors     = elig_pos[sampled]   # (4000, 2)

print(f"Anchors : {TARGET_ANCHORS}")
print(f"X: [{pos_anchors[:,0].min():.3f}, {pos_anchors[:,0].max():.3f}]")
print(f"Y: [{pos_anchors[:,1].min():.3f}, {pos_anchors[:,1].max():.3f}]")

# Extract anchor CFRs
print("\n" + "=" * 60)
print("Step 3: Extracting anchor CFRs...")
print("=" * 60)

CFR = np.array([
    file_data[anchor_file_idx[i]]["csi"][anchor_local[i]]
    for i in range(TARGET_ANCHORS)
], dtype=np.complex64)   # (4000, 8, 1024) — already fftshifted + calibrated
print(f"CFR shape: {CFR.shape}  dtype: {CFR.dtype}")

del file_data   # free memory — no longer needed

# Compute centred CIR and extract fixed 18 taps
print("\n" + "=" * 60)
print(f"Step 4: Computing centred CIR and extracting [{TAP_START}:{TAP_STOP}]...")
print("=" * 60)

CIR_full = shifted_cfr_to_centered_cir(CFR)
C_cir_anchor = CIR_full[:, :, TAP_START:TAP_STOP].copy()

del CIR_full
print(f"C_cir_anchor shape: {C_cir_anchor.shape}  dtype: {C_cir_anchor.dtype}")

# Beam-delay representation
print("\n" + "=" * 60)
print("Step 5: Computing beam-delay representation...")
print("=" * 60)

C_cir_grid = C_cir_anchor.reshape(
    TARGET_ANCHORS,
    2,
    4,
    N_TAPS,
)

C_beam_grid = np.fft.fftshift(
    np.fft.fft2(
        C_cir_grid,
        axes=(1, 2),
        norm="ortho",
    ),
    axes=(1, 2),
).astype(np.complex64)

C_beam_cir_anchor = C_beam_grid.reshape(
    TARGET_ANCHORS,
    8,
    N_TAPS,
)

del C_cir_grid, C_beam_grid

print(f"C_beam_cir_anchor shape: {C_beam_cir_anchor.shape}  dtype: {C_beam_cir_anchor.dtype}")

# Physical distance matrix
print("\n" + "=" * 60)
print("Step 6: Computing physical distance matrix...")
print("=" * 60)

diff   = pos_anchors[:, np.newaxis, :] - pos_anchors[np.newaxis, :, :]
D_phys = np.sqrt(np.sum(diff**2, axis=-1))
np.fill_diagonal(D_phys, 0.0)
print(f"D_phys: {D_phys.shape} | "
      f"min={D_phys[D_phys>0].min():.4f} m, max={D_phys.max():.4f} m")

# SqASD distance computation
def compute_sqasd_distance_matrix(H, dist_type, avg_axis):
    """Compute RASD for the selected CSI representation."""
    N = H.shape[0]

    if avg_axis == 'row':
        n_outer      = H.shape[1]                          # 8 antennas/beams
        outer_slices = [H[:, m, :] for m in range(n_outer)]
    else:
        n_outer      = H.shape[2]                          # F subcarriers/taps
        outer_slices = [H[:, :, k] for k in range(n_outer)]

    D_sq2 = np.zeros((N, N), dtype=np.float64)   # accumulates sum of Dd^2

    for slc in tqdm(outer_slices, desc=f"  {dist_type} ({avg_axis})", leave=False):
        hi = slc.astype(np.complex128)   # (N, vec_dim)

        ni2   = np.sum(np.abs(hi)**2, axis=1)
        inner = hi @ hi.conj().T

        norms      = np.linalg.norm(hi, axis=-1, keepdims=True)
        safe_norms = np.where(norms == 0.0, 1.0, norms)
        hi_n       = hi / safe_norms
        inner_norm     = hi_n @ hi_n.conj().T
        abs_inner_norm = np.clip(np.abs(inner_norm), 0.0, 1.0)

        if dist_type == 'euclidean':
            D2 = ni2[:, None] + ni2[None, :] - 2.0 * np.real(inner)
            Dd = np.sqrt(np.maximum(D2, 0.0))
        elif dist_type == 'norm_euclidean':
            D2 = 2.0 - 2.0 * np.real(inner_norm)
            Dd = np.sqrt(np.maximum(D2, 0.0))
        elif dist_type == 'norm_geodesic_sphere':
            Dd = np.arccos(np.clip(np.real(inner_norm), -1.0, 1.0))
        elif dist_type == 'global_phase_chordal':
            abs_inner = np.abs(inner)
            term      = 0.5 * (ni2[:, None]**2 + ni2[None, :]**2) - abs_inner**2
            Dd        = np.sqrt(np.maximum(term, 0.0))
        elif dist_type == 'global_phase_bw':
            abs_inner = np.abs(inner)
            term      = 0.5 * (ni2[:, None] + ni2[None, :]) - abs_inner
            Dd        = np.sqrt(np.maximum(term, 0.0))
        elif dist_type == 'norm_chordal':
            Dd = np.sqrt(np.maximum(1.0 - abs_inner_norm**2, 0.0))
        elif dist_type == 'norm_geodesic_grass':
            Dd = np.arccos(np.clip(abs_inner_norm, 0.0, 1.0))
        elif dist_type == 'norm_bw':
            Dd = np.sqrt(np.maximum(1.0 - abs_inner_norm, 0.0))
        else:
            raise ValueError(f"Unknown dist_type: {dist_type}")

        np.fill_diagonal(Dd, 0.0)
        D_sq2 += Dd ** 2   # accumulate squared distances

    # single sqrt over mean of squared distances (SqASD / L_{2,2})
    D = np.sqrt(D_sq2 / n_outer)
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

def neighbourhood_preservation_ratio(R_phys, R_chan, J):
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
    np_r = neighbourhood_preservation_ratio(R_phys, R_chan, J)
    return dict(TW=tw, CT=ct, KS=ks, NPR=np_r)

# Distance types
dist_types = [
    'euclidean', 'norm_euclidean', 'norm_geodesic_sphere',
    'global_phase_chordal', 'global_phase_bw',
    'norm_chordal', 'norm_geodesic_grass', 'norm_bw',
]
dist_labels = [
    'Euclidean',
    'Normalized Euclidean',
    'Geodesic on sphere S^{2D-1}',
    'Global phase, Chordal',
    'Global phase, Bur.-Was.',
    'Norm+Global phase, Chordal',
    'Norm+Global phase, Geodesic Grass.',
    'Norm+Global phase, Bur.-Was.',
]

# Configurations
# CFR              : (4000, 8, 1024) — antenna × subcarrier
# C_cir_anchor     : (4000, 8, 18)  — antenna × tap
# C_beam_cir_anchor: (4000, 8, 18)  — beam × tap
#
# antenna_per_freq  : CFR,   avg_axis='col' → outer=freq  (1024), inner=antenna (8)
# antenna_per_delay : CIR,   avg_axis='col' → outer=tap   (18),  inner=antenna (8)
# delay_per_antenna : CIR,   avg_axis='row' → outer=ant   (8),    inner=tap     (161)
# delay_per_beam    : beam,  avg_axis='row' → outer=beam  (8),    inner=tap     (161)

configs = [
    (
        'Antenna-per-Frequency (CFR, Array A, anchor-only, SqASD)',
        CFR,
        'col',
        'antenna_per_freq',
    ),
    (
        'Antenna-per-Delay (CIR, Array A, fixed 18 taps, anchor-only, SqASD)',
        C_cir_anchor,
        'col',
        'antenna_per_delay',
    ),
    (
        'Delay-per-Antenna (CIR, Array A, fixed 18 taps, anchor-only, SqASD)',
        C_cir_anchor,
        'row',
        'delay_per_antenna',
    ),
    (
        'Delay-per-Beam (2D beam-CIR, Array A, fixed 18 taps, anchor-only, SqASD)',
        C_beam_cir_anchor,
        'row',
        'delay_per_beam',
    ),
]

# Main computation loop
all_results = {}

for cfg_label, H, avg_axis, feat_label in configs:
    print(f"\n{'='*70}")
    print(f"  {cfg_label}  |  avg_axis={avg_axis}")
    print(f"{'='*70}")

    feat_results = {}
    for dt, dl in zip(dist_types, dist_labels):
        D_chan  = compute_sqasd_distance_matrix(H, dt, avg_axis=avg_axis)
        metrics = compute_all_metrics(D_phys, D_chan, J)
        feat_results[dt] = metrics

    all_results[feat_label] = feat_results

# Final summary
print(f"\n{'='*120}")
print("  FINAL SUMMARY — DICHASUS Array A, anchor-only, SqASD, 18 CIR taps")
print(f"{'='*120}")
for cfg_label, H, avg_axis, feat_label in configs:
    print(f"\n  Feature: {feat_label}")
    print(f"  {'Distance':<40} {'TW':>8} {'CT':>8} {'NPR':>8} {'KS':>8}")
    print(f"  {'-'*96}")
    for dt, dl in zip(dist_types, dist_labels):
        m = all_results[feat_label][dt]
        print(f"  {dl:<40} {m['TW']:>8.4f} {m['CT']:>8.4f} "
              f"{m['NPR']:>8.4f} {m['KS']:>8.4f} "
              f"")

print("\nAll done!")