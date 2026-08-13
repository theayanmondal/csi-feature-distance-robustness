import os
import json
import gc

import numpy as np
import tensorflow as tf
from tqdm import tqdm


# Paths and experiment settings
BASE_PATH = "data/dichasus"

OUTPUT_DIR = (
    "outputs/dichasus/"
    "ArrayA/anchor_only_avd_18taps/"
)
os.makedirs(OUTPUT_DIR, exist_ok=True)

TFRECORD_FILES = [
    os.path.join(BASE_PATH, f"dichasus-cf0{i}.tfrecords")
    for i in range(2, 8)
]

OFFSET_FILES = [
    os.path.join(BASE_PATH, f"reftx-offsets-dichasus-cf0{i}.json")
    for i in range(2, 8)
]

# Physical DICHASUS Antenna Array A, preserving its 2 x 4 arrangement.
ARRAY_A_GRID = np.array(
    [
        [6, 2, 16, 18],
        [28, 5, 10, 14],
    ],
    dtype=np.int32,
)
ARRAY_A_CHANNELS = ARRAY_A_GRID.ravel()
N_ANTENNAS = len(ARRAY_A_CHANNELS)

# Require 49 successors when selecting anchors.
N_NEIGHBOURS = 49
TARGET_ANCHORS = 4000
GRID_SIZE = 20
SEED = 42

# Neighbourhood size used by TW, CT, NPR.
J = 100

# Fixed CIR interval selected from the rev2 aggregate PDP analysis.
TAP_START = 509
TAP_STOP = 527  # exclusive
N_TAPS = TAP_STOP - TAP_START

assert N_TAPS == 18


# TFRecord parsing
FEATURE_DESCRIPTION = {
    "csi": tf.io.FixedLenFeature([], tf.string, default_value=""),
    "pos-tachy": tf.io.FixedLenFeature([], tf.string, default_value=""),
    "time": tf.io.FixedLenFeature([], tf.float32, default_value=0),
}


def parse_record(serialized_record):
    record = tf.io.parse_single_example(
        serialized_record,
        FEATURE_DESCRIPTION,
    )

    csi = tf.ensure_shape(
        tf.io.parse_tensor(
            record["csi"],
            out_type=tf.float32,
        ),
        (32, 1024, 2),
    )

    position = tf.ensure_shape(
        tf.io.parse_tensor(
            record["pos-tachy"],
            out_type=tf.float64,
        ),
        (3,),
    )

    return {
        "csi": csi,
        "pos": position,
    }


def to_complex(csi_iq):
    """Convert (32, 1024, 2) I/Q CSI to (32, 1024) complex CSI."""
    return (
        csi_iq[..., 0] + 1j * csi_iq[..., 1]
    ).astype(np.complex64)


# DICHASUS rev2 preprocessing
def normalize_rev2_sto(csi_shifted):
    """Remove the sample-dependent common phase slope in DICHASUS rev2."""
    adjacent_products = (
        csi_shifted[:, 1:]
        * np.conj(csi_shifted[:, :-1])
    )

    phase_increment = np.float32(
        np.angle(np.sum(adjacent_products))
    )

    subcarrier_indices = np.arange(
        csi_shifted.shape[-1],
        dtype=np.float32,
    )

    correction = np.exp(
        -1j * phase_increment * subcarrier_indices
    ).astype(np.complex64)

    return (
        csi_shifted * correction[None, :]
    ).astype(np.complex64)


def apply_calibration(csi_rev2_normalized, offsets):
    """Apply file-specific per-channel STO/CPO calibration."""
    n_channels, n_subcarriers = csi_rev2_normalized.shape

    sto = np.asarray(offsets["sto"], dtype=np.float32)
    cpo = np.asarray(offsets["cpo"], dtype=np.float32)

    if sto.shape != (n_channels,):
        raise ValueError(
            f"Expected STO shape ({n_channels},), got {sto.shape}."
        )

    if cpo.shape != (n_channels,):
        raise ValueError(
            f"Expected CPO shape ({n_channels},), got {cpo.shape}."
        )

    subcarrier_indices = np.arange(
        n_subcarriers,
        dtype=np.float32,
    )

    phase = (
        sto[:, None]
        * (
            2.0
            * np.pi
            * subcarrier_indices[None, :]
            / n_subcarriers
        )
        + cpo[:, None]
    )

    correction = np.exp(
        1j * phase
    ).astype(np.complex64)

    return (
        csi_rev2_normalized * correction
    ).astype(np.complex64)


def shifted_cfr_to_centered_cir(cfr_shifted):
    """Convert calibrated shifted CFR to centred CIR."""
    return np.fft.fftshift(
        np.fft.ifft(
            np.fft.ifftshift(
                cfr_shifted,
                axes=-1,
            ),
            axis=-1,
        ),
        axes=-1,
    ).astype(np.complex64)


# Spatially stratified anchor sampling
def stratified_sample_exact(positions, target, grid_size, rng):
    x_edges = np.linspace(
        positions[:, 0].min(),
        positions[:, 0].max(),
        grid_size + 1,
    )

    y_edges = np.linspace(
        positions[:, 1].min(),
        positions[:, 1].max(),
        grid_size + 1,
    )

    cell_members = {}

    for row in range(grid_size):
        for col in range(grid_size):
            in_x = (
                (positions[:, 0] >= x_edges[col])
                & (positions[:, 0] < x_edges[col + 1])
            )

            in_y = (
                (positions[:, 1] >= y_edges[row])
                & (positions[:, 1] < y_edges[row + 1])
            )

            if col == grid_size - 1:
                in_x |= positions[:, 0] == x_edges[col + 1]

            if row == grid_size - 1:
                in_y |= positions[:, 1] == y_edges[row + 1]

            indices = np.where(in_x & in_y)[0]

            if len(indices) > 0:
                cell_members[(row, col)] = indices

    if not cell_members:
        raise RuntimeError("No non-empty spatial grid cells were found.")

    fair_share = target // len(cell_members)
    selected = []
    remaining = {}

    for cell, indices in cell_members.items():
        count = min(fair_share, len(indices))

        chosen = rng.choice(
            indices,
            size=count,
            replace=False,
        )

        selected.extend(chosen.tolist())

        leftover = np.setdiff1d(indices, chosen)

        if len(leftover) > 0:
            remaining[cell] = leftover

    shortfall = target - len(selected)

    if shortfall > 0:
        ordered_cells = sorted(
            remaining.items(),
            key=lambda item: len(item[1]),
            reverse=True,
        )

        for _, leftover in ordered_cells:
            if shortfall <= 0:
                break

            count = min(shortfall, len(leftover))

            chosen = rng.choice(
                leftover,
                size=count,
                replace=False,
            )

            selected.extend(chosen.tolist())
            shortfall -= count

    selected = np.sort(
        np.asarray(selected, dtype=np.int64)
    )

    if len(selected) != target:
        raise RuntimeError(
            f"Could not select exactly {target} anchors; "
            f"selected {len(selected)}."
        )

    return selected


# Load raw rev2 data, calibrate, and select Array A
print("=" * 78)
print("STEP 1: LOADING AND PREPROCESSING DICHASUS REV2")
print("=" * 78)

file_data = []

for file_index, (tfrecord_path, offset_path) in enumerate(
    zip(TFRECORD_FILES, OFFSET_FILES)
):
    filename = os.path.basename(tfrecord_path)

    with open(offset_path, "r") as file:
        offsets = json.load(file)

    dataset = tf.data.TFRecordDataset(tfrecord_path)

    dataset = dataset.map(
        parse_record,
        num_parallel_calls=tf.data.AUTOTUNE,
    ).prefetch(tf.data.AUTOTUNE)

    file_csi = []
    file_positions = []

    for sample in tqdm(
        dataset,
        desc=filename,
    ):
        csi_iq = sample["csi"].numpy()

        # 1. Reconstruct complex CSI.
        csi_complex = to_complex(csi_iq)

        # 2. Put frequency bins in shifted order.
        csi_shifted = np.fft.fftshift(
            csi_complex,
            axes=-1,
        )

        # 3. Rev2-specific common STO normalization.
        csi_rev2_normalized = normalize_rev2_sto(
            csi_shifted
        )

        # 4. File-specific per-channel STO/CPO calibration.
        csi_calibrated = apply_calibration(
            csi_rev2_normalized,
            offsets,
        )

        # 5. Select physical Antenna Array A.
        csi_array_a = csi_calibrated[
            ARRAY_A_CHANNELS,
            :,
        ]

        file_csi.append(csi_array_a)
        file_positions.append(
            sample["pos"].numpy()[:2]
        )

    file_data.append(
        {
            "file_index": file_index,
            "filename": filename,
            "n": len(file_csi),
            "csi": np.asarray(
                file_csi,
                dtype=np.complex64,
            ),
            "pos": np.asarray(
                file_positions,
                dtype=np.float64,
            ),
        }
    )

    print(
        f"  {filename}: "
        f"{file_data[-1]['n']} samples"
    )


# Select exactly 4000 anchors
print("\n" + "=" * 78)
print("STEP 2: SELECTING 4000 SPATIALLY STRATIFIED ANCHORS")
print("=" * 78)

eligible_positions = []
eligible_file_indices = []
eligible_local_indices = []

for data in file_data:
    # Require 49 successors for anchor eligibility.
    last_eligible = data["n"] - 1 - N_NEIGHBOURS

    if last_eligible < 0:
        continue

    n_eligible = last_eligible + 1

    eligible_positions.extend(
        data["pos"][:n_eligible].tolist()
    )

    eligible_file_indices.extend(
        [data["file_index"]] * n_eligible
    )

    eligible_local_indices.extend(
        range(n_eligible)
    )

eligible_positions = np.asarray(
    eligible_positions,
    dtype=np.float64,
)

eligible_file_indices = np.asarray(
    eligible_file_indices,
    dtype=np.int32,
)

eligible_local_indices = np.asarray(
    eligible_local_indices,
    dtype=np.int32,
)

if len(eligible_positions) < TARGET_ANCHORS:
    raise RuntimeError(
        f"Only {len(eligible_positions)} eligible samples are available, "
        f"but {TARGET_ANCHORS} anchors are required."
    )

rng = np.random.default_rng(SEED)

sampled_indices = stratified_sample_exact(
    eligible_positions,
    TARGET_ANCHORS,
    GRID_SIZE,
    rng,
)

anchor_file_indices = eligible_file_indices[
    sampled_indices
]

anchor_local_indices = eligible_local_indices[
    sampled_indices
]

pos_anchors = eligible_positions[
    sampled_indices
]

print(f"Anchors selected       : {len(pos_anchors)}")
print(f"Neighbour eligibility  : {N_NEIGHBOURS}")
print(
    "Per-file anchor counts:",
    np.bincount(
        anchor_file_indices,
        minlength=len(file_data),
    ).tolist(),
)
print(
    f"X range: [{pos_anchors[:, 0].min():.3f}, "
    f"{pos_anchors[:, 0].max():.3f}]"
)
print(
    f"Y range: [{pos_anchors[:, 1].min():.3f}, "
    f"{pos_anchors[:, 1].max():.3f}]"
)


# Extract anchor-only CFR
print("\n" + "=" * 78)
print("STEP 3: EXTRACTING ANCHOR-ONLY CFR")
print("=" * 78)

CFR = np.asarray(
    [
        file_data[int(anchor_file_indices[index])]["csi"][
            int(anchor_local_indices[index])
        ]
        for index in range(TARGET_ANCHORS)
    ],
    dtype=np.complex64,
)

assert CFR.shape == (
    TARGET_ANCHORS,
    N_ANTENNAS,
    1024,
)

print(f"CFR shape: {CFR.shape}")
print("CFR status: fftshifted, rev2-normalized, and calibrated")


# Compute centred CIR and retain fixed taps [509:527]
print("\n" + "=" * 78)
print("STEP 4: COMPUTING CIR AND EXTRACTING FIXED 18-TAP INTERVAL")
print("=" * 78)

CIR_full = shifted_cfr_to_centered_cir(
    CFR
)

C_cir_anchor = CIR_full[
    :,
    :,
    TAP_START:TAP_STOP,
].copy()

assert C_cir_anchor.shape == (
    TARGET_ANCHORS,
    N_ANTENNAS,
    N_TAPS,
)

del CIR_full
gc.collect()

print(f"Fixed tap interval : [{TAP_START}:{TAP_STOP}]")
print(f"Number of taps     : {N_TAPS}")
print(f"CIR feature shape  : {C_cir_anchor.shape}")


# Compute 2D beam-delay representation
print("\n" + "=" * 78)
print("STEP 5: COMPUTING 2D BEAM-DELAY REPRESENTATION")
print("=" * 78)

# Restore Array A to its physical 2 x 4 layout.
C_cir_grid = C_cir_anchor.reshape(
    TARGET_ANCHORS,
    2,
    4,
    N_TAPS,
)

# Apply a 2D spatial DFT across the two physical array dimensions.
C_beam_grid = np.fft.fftshift(
    np.fft.fft2(
        C_cir_grid,
        axes=(1, 2),
        norm="ortho",
    ),
    axes=(1, 2),
).astype(np.complex64)

# Flatten the 2 x 4 beam grid back to eight beam bins.
C_beam_cir_anchor = C_beam_grid.reshape(
    TARGET_ANCHORS,
    N_ANTENNAS,
    N_TAPS,
)

del C_cir_grid, C_beam_grid
gc.collect()

print(
    "Beam-delay feature shape: "
    f"{C_beam_cir_anchor.shape}"
)


# Physical distance matrix
print("\n" + "=" * 78)
print("STEP 6: COMPUTING PHYSICAL DISTANCE MATRIX")
print("=" * 78)

position_difference = (
    pos_anchors[:, None, :]
    - pos_anchors[None, :, :]
)

D_phys = np.sqrt(
    np.sum(
        position_difference**2,
        axis=-1,
    )
).astype(np.float64)

np.fill_diagonal(D_phys, 0.0)

del position_difference
gc.collect()

print(
    f"D_phys shape: {D_phys.shape} | "
    f"min nonzero={D_phys[D_phys > 0].min():.4f} m | "
    f"max={D_phys.max():.4f} m"
)


# AVD distance matrices
def compute_avd_distance_matrix(features, distance_type, average_axis):
    n_samples = features.shape[0]

    if average_axis == "row":
        outer_slices = (
            features[:, outer_index, :]
            for outer_index in range(features.shape[1])
        )
        n_outer = features.shape[1]

    elif average_axis == "col":
        outer_slices = (
            features[:, :, outer_index]
            for outer_index in range(features.shape[2])
        )
        n_outer = features.shape[2]

    else:
        raise ValueError(
            "average_axis must be 'row' or 'col'."
        )

    distance_sum = np.zeros(
        (n_samples, n_samples),
        dtype=np.float64,
    )

    for feature_slice in tqdm(
        outer_slices,
        total=n_outer,
        desc=f"  {distance_type} ({average_axis})",
        leave=False,
    ):
        vectors = feature_slice.astype(
            np.complex128,
            copy=False,
        )

        norm_squared = np.sum(
            np.abs(vectors) ** 2,
            axis=1,
        )

        inner_product = (
            vectors @ vectors.conj().T
        )

        norms = np.linalg.norm(
            vectors,
            axis=1,
            keepdims=True,
        )

        safe_norms = np.where(
            norms > 0.0,
            norms,
            1.0,
        )

        normalized_vectors = (
            vectors / safe_norms
        )

        normalized_inner = (
            normalized_vectors
            @ normalized_vectors.conj().T
        )

        absolute_normalized_inner = np.clip(
            np.abs(normalized_inner),
            0.0,
            1.0,
        )

        if distance_type == "euclidean":
            distance_squared = (
                norm_squared[:, None]
                + norm_squared[None, :]
                - 2.0 * np.real(inner_product)
            )

            distance_slice = np.sqrt(
                np.maximum(
                    distance_squared,
                    0.0,
                )
            )

        elif distance_type == "norm_euclidean":
            distance_squared = (
                2.0
                - 2.0 * np.real(normalized_inner)
            )

            distance_slice = np.sqrt(
                np.maximum(
                    distance_squared,
                    0.0,
                )
            )

        elif distance_type == "norm_geodesic_sphere":
            distance_slice = np.arccos(
                np.clip(
                    np.real(normalized_inner),
                    -1.0,
                    1.0,
                )
            )

        elif distance_type == "global_phase_chordal":
            absolute_inner = np.abs(
                inner_product
            )

            distance_slice = np.sqrt(
                np.maximum(
                    0.5
                    * (
                        norm_squared[:, None] ** 2
                        + norm_squared[None, :] ** 2
                    )
                    - absolute_inner**2,
                    0.0,
                )
            )

        elif distance_type == "global_phase_bw":
            absolute_inner = np.abs(
                inner_product
            )

            distance_slice = np.sqrt(
                np.maximum(
                    0.5
                    * (
                        norm_squared[:, None]
                        + norm_squared[None, :]
                    )
                    - absolute_inner,
                    0.0,
                )
            )


        elif distance_type == "norm_chordal":
            distance_slice = np.sqrt(
                np.maximum(
                    1.0
                    - absolute_normalized_inner**2,
                    0.0,
                )
            )

        elif distance_type == "norm_geodesic_grass":
            distance_slice = np.arccos(
                absolute_normalized_inner
            )

        elif distance_type == "norm_bw":
            distance_slice = np.sqrt(
                np.maximum(
                    1.0
                    - absolute_normalized_inner,
                    0.0,
                )
            )

        else:
            raise ValueError(
                f"Unknown distance type: {distance_type}"
            )

        np.fill_diagonal(
            distance_slice,
            0.0,
        )

        distance_sum += distance_slice

    distance_matrix = (
        distance_sum / float(n_outer)
    )

    np.fill_diagonal(
        distance_matrix,
        0.0,
    )

    return distance_matrix


# Quality metrics
def rank_matrix(distance_matrix):
    without_self = distance_matrix.copy()

    np.fill_diagonal(
        without_self,
        np.inf,
    )

    return (
        np.argsort(
            np.argsort(
                without_self,
                axis=1,
            ),
            axis=1,
        )
        + 1
    )


def trustworthiness(physical_ranks, channel_ranks, neighbourhood_size):
    n_samples = physical_ranks.shape[0]

    false_neighbours = (
        (channel_ranks <= neighbourhood_size)
        & (physical_ranks > neighbourhood_size)
    )

    penalty = np.where(
        false_neighbours,
        physical_ranks - neighbourhood_size,
        0,
    )

    normalization = (
        2.0
        / (
            n_samples
            * neighbourhood_size
            * (
                2 * n_samples
                - 3 * neighbourhood_size
                - 1
            )
        )
    )

    return (
        1.0
        - normalization * penalty.sum()
    )


def continuity(physical_ranks, channel_ranks, neighbourhood_size):
    n_samples = physical_ranks.shape[0]

    missing_neighbours = (
        (physical_ranks <= neighbourhood_size)
        & (channel_ranks > neighbourhood_size)
    )

    penalty = np.where(
        missing_neighbours,
        channel_ranks - neighbourhood_size,
        0,
    )

    normalization = (
        2.0
        / (
            n_samples
            * neighbourhood_size
            * (
                2 * n_samples
                - 3 * neighbourhood_size
                - 1
            )
        )
    )

    return (
        1.0
        - normalization * penalty.sum()
    )


def kruskal_stress(physical_distances, channel_distances):
    upper_triangle = np.triu_indices(
        physical_distances.shape[0],
        k=1,
    )

    physical_vector = physical_distances[
        upper_triangle
    ].astype(np.float64)

    channel_vector = channel_distances[
        upper_triangle
    ].astype(np.float64)

    denominator = np.sum(
        channel_vector**2
    )

    if denominator <= 0.0:
        return float("nan"), float("nan")

    scale = (
        np.sum(
            physical_vector * channel_vector
        )
        / denominator
    )

    physical_energy = np.sum(
        physical_vector**2
    )

    if physical_energy <= 0.0:
        return float("nan"), float(scale)

    stress = np.sqrt(
        np.sum(
            (
                physical_vector
                - scale * channel_vector
            ) ** 2
        )
        / physical_energy
    )

    return float(stress), float(scale)


def neighbourhood_preservation_ratio(
    physical_ranks,
    channel_ranks,
    neighbourhood_size,
):
    physical_neighbours = (
        physical_ranks <= neighbourhood_size
    )

    channel_neighbours = (
        channel_ranks <= neighbourhood_size
    )

    intersection_size = (
        physical_neighbours
        & channel_neighbours
    ).sum(axis=1)

    return float(
        intersection_size.mean()
        / neighbourhood_size
    )








def compute_all_metrics(D_phys, D_chan, J):
    R_phys = rank_matrix(D_phys)
    R_chan = rank_matrix(D_chan)
    tw = trustworthiness(R_phys, R_chan, J)
    ct = continuity(R_phys, R_chan, J)
    ks, _ = kruskal_stress(D_phys, D_chan)
    np_r = neighbourhood_preservation_ratio(R_phys, R_chan, J)
    return dict(TW=tw, CT=ct, KS=ks, NPR=np_r)


# Distances and feature configurations
DISTANCE_TYPES = [
    "euclidean",
    "norm_euclidean",
    "norm_geodesic_sphere",
    "global_phase_chordal",
    "global_phase_bw",
    "norm_chordal",
    "norm_geodesic_grass",
    "norm_bw",
]

DISTANCE_LABELS = [
    "Euclidean",
    "Normalized Euclidean",
    "Geodesic on sphere S^{2D-1}",
    "Global phase, Chordal",
    "Global phase, Bur.-Was.",
    "Norm+Global phase, Chordal",
    "Norm+Global phase, Geodesic Grass.",
    "Norm+Global phase, Bur.-Was.",
]

CONFIGURATIONS = [
    (
        "Antenna-per-Frequency "
        "(CFR, Array A, anchor-only)",
        CFR,
        "col",
        "antenna_per_frequency",
    ),
    (
        "Antenna-per-Delay "
        "(CIR, Array A, fixed 18 taps)",
        C_cir_anchor,
        "col",
        "antenna_per_delay",
    ),
    (
        "Delay-per-Antenna "
        "(CIR, Array A, fixed 18 taps)",
        C_cir_anchor,
        "row",
        "delay_per_antenna",
    ),
    (
        "Delay-per-Beam "
        "(2D beam-CIR, Array A, fixed 18 taps)",
        C_beam_cir_anchor,
        "row",
        "delay_per_beam",
    ),
]


# Main experiment
all_results = {}

for (
    configuration_label,
    features,
    average_axis,
    feature_label,
) in CONFIGURATIONS:
    print("\n" + "=" * 78)
    print(configuration_label)
    print(
        f"Feature shape: {features.shape} | "
        f"average_axis={average_axis}"
    )
    print("=" * 78)

    feature_results = {}

    for distance_type, distance_label in zip(
        DISTANCE_TYPES,
        DISTANCE_LABELS,
    ):
        print(f"\nComputing: {distance_label}")

        D_channel = compute_avd_distance_matrix(
            features,
            distance_type,
            average_axis,
        )

        metrics = compute_all_metrics(
            D_phys,
            D_channel,
            J,
        )

        feature_results[
            distance_type
        ] = metrics

        del D_channel
        gc.collect()

    all_results[
        feature_label
    ] = feature_results


# Final summary
print("\n" + "=" * 120)
print(
    "FINAL SUMMARY — DICHASUS REV2, ARRAY A, "
    "ANCHOR-ONLY, AVD, FIXED TAPS [509:527]"
)
print("=" * 120)

for (
    configuration_label,
    _,
    _,
    feature_label,
) in CONFIGURATIONS:
    print(f"\nFeature: {feature_label}")

    print(
        f"{'Distance':<40} "
        f"{'TW':>8} "
        f"{'CT':>8} "
        f"{'NPR':>8} "
        f"{'KS':>8} "
        f" "
        f""
    )

    print("-" * 96)

    for distance_type, distance_label in zip(
        DISTANCE_TYPES,
        DISTANCE_LABELS,
    ):
        metrics = all_results[
            feature_label
        ][distance_type]

        print(
            f"{distance_label:<40} "
            f"{metrics['TW']:>8.4f} "
            f"{metrics['CT']:>8.4f} "
            f"{metrics['NPR']:>8.4f} "
            f"{metrics['KS']:>8.4f} "
            f" "
            f""
        )

print("\nAll done!")
