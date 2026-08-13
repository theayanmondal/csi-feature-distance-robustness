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
    "ArrayA/average_over_6_sqasd_18taps/"
)
os.makedirs(OUTPUT_DIR, exist_ok=True)

TFRECORD_FILES = [
    os.path.join(
        BASE_PATH,
        f"dichasus-cf0{i}.tfrecords",
    )
    for i in range(2, 8)
]

OFFSET_FILES = [
    os.path.join(
        BASE_PATH,
        f"reftx-offsets-dichasus-cf0{i}.json",
    )
    for i in range(2, 8)
]

# Physical DICHASUS Array A in its 2 x 4 layout.
ARRAY_A_GRID = np.array(
    [
        [6, 2, 16, 18],
        [28, 5, 10, 14],
    ],
    dtype=np.int32,
)

ARRAY_A_CHANNELS = ARRAY_A_GRID.ravel()
N_ANTENNAS = len(ARRAY_A_CHANNELS)

TARGET_ANCHORS = 4000

# Require 49 successors when selecting anchors.
N_NEIGHBOURS = 49

# Actual temporal window used in this experiment:
# anchor + five immediate consecutive samples.
N_WINDOW = 6

GRID_SIZE = 20
SEED = 42
J = 100

# Fixed interval selected from the rev2 aggregate delay-power analysis.
TAP_START = 509
TAP_STOP = 527
N_TAPS = TAP_STOP - TAP_START

assert N_TAPS == 18
assert N_WINDOW <= N_NEIGHBOURS + 1


# TFRecord parsing
POSITION_DESCRIPTION = {
    "pos-tachy": tf.io.FixedLenFeature(
        [],
        tf.string,
        default_value="",
    ),
}

CSI_DESCRIPTION = {
    "csi": tf.io.FixedLenFeature(
        [],
        tf.string,
        default_value="",
    ),
}


def parse_position(serialized_record):
    record = tf.io.parse_single_example(
        serialized_record,
        POSITION_DESCRIPTION,
    )

    position = tf.io.parse_tensor(
        record["pos-tachy"],
        out_type=tf.float64,
    )

    return tf.ensure_shape(
        position,
        (3,),
    )


def parse_csi(serialized_record):
    record = tf.io.parse_single_example(
        serialized_record,
        CSI_DESCRIPTION,
    )

    csi = tf.io.parse_tensor(
        record["csi"],
        out_type=tf.float32,
    )

    return tf.ensure_shape(
        csi,
        (32, 1024, 2),
    )


def to_complex(csi_iq):
    return (
        csi_iq[..., 0]
        + 1j * csi_iq[..., 1]
    ).astype(np.complex64)


# DICHASUS rev2 preprocessing
def normalize_rev2_sto(csi_shifted):
    """Remove the sample-dependent common phase slope in DICHASUS rev2."""
    adjacent_products = (
        csi_shifted[:, 1:]
        * np.conj(csi_shifted[:, :-1])
    )

    phase_increment = np.float32(
        np.angle(
            np.sum(adjacent_products)
        )
    )

    subcarrier_indices = np.arange(
        csi_shifted.shape[-1],
        dtype=np.float32,
    )

    correction = np.exp(
        -1j
        * phase_increment
        * subcarrier_indices
    ).astype(np.complex64)

    return (
        csi_shifted
        * correction[None, :]
    ).astype(np.complex64)


def apply_calibration(
    csi_rev2_normalized,
    offsets,
):
    """Apply file-specific per-channel STO/CPO calibration."""
    n_channels, n_subcarriers = (
        csi_rev2_normalized.shape
    )

    sto = np.asarray(
        offsets["sto"],
        dtype=np.float32,
    )

    cpo = np.asarray(
        offsets["cpo"],
        dtype=np.float32,
    )

    if sto.shape != (n_channels,):
        raise ValueError(
            f"Expected STO shape ({n_channels},), "
            f"received {sto.shape}."
        )

    if cpo.shape != (n_channels,):
        raise ValueError(
            f"Expected CPO shape ({n_channels},), "
            f"received {cpo.shape}."
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
        csi_rev2_normalized
        * correction
    ).astype(np.complex64)


def shifted_cfr_to_centered_cir(
    cfr_shifted,
):
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
def stratified_sample_exact(
    positions,
    target,
    grid_size,
    rng,
):
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
                & (
                    positions[:, 0]
                    < x_edges[col + 1]
                )
            )

            in_y = (
                (positions[:, 1] >= y_edges[row])
                & (
                    positions[:, 1]
                    < y_edges[row + 1]
                )
            )

            if col == grid_size - 1:
                in_x |= (
                    positions[:, 0]
                    == x_edges[col + 1]
                )

            if row == grid_size - 1:
                in_y |= (
                    positions[:, 1]
                    == y_edges[row + 1]
                )

            indices = np.where(
                in_x & in_y
            )[0]

            if len(indices) > 0:
                cell_members[
                    (row, col)
                ] = indices

    if not cell_members:
        raise RuntimeError(
            "No occupied spatial grid cells were found."
        )

    fair_share = (
        target // len(cell_members)
    )

    selected = []
    remaining = {}

    for cell, indices in cell_members.items():
        count = min(
            fair_share,
            len(indices),
        )

        chosen = rng.choice(
            indices,
            size=count,
            replace=False,
        )

        selected.extend(
            chosen.tolist()
        )

        leftover = np.setdiff1d(
            indices,
            chosen,
        )

        if len(leftover) > 0:
            remaining[cell] = leftover

    shortfall = (
        target - len(selected)
    )

    if shortfall > 0:
        ordered_cells = sorted(
            remaining.items(),
            key=lambda item: len(item[1]),
            reverse=True,
        )

        for _, leftover in ordered_cells:
            if shortfall <= 0:
                break

            count = min(
                shortfall,
                len(leftover),
            )

            chosen = rng.choice(
                leftover,
                size=count,
                replace=False,
            )

            selected.extend(
                chosen.tolist()
            )

            shortfall -= count

    selected = np.sort(
        np.asarray(
            selected,
            dtype=np.int64,
        )
    )

    if len(selected) != target:
        raise RuntimeError(
            f"Could not select exactly {target} anchors; "
            f"selected {len(selected)}."
        )

    return selected


# Pass 1: read positions and select the same 4000 anchors
print("=" * 78)
print("PASS 1: READING POSITIONS AND SELECTING ANCHORS")
print("=" * 78)

file_positions = []

for tfrecord_path in TFRECORD_FILES:
    filename = os.path.basename(
        tfrecord_path
    )

    dataset = tf.data.TFRecordDataset(
        tfrecord_path
    )

    dataset = dataset.map(
        parse_position,
        num_parallel_calls=tf.data.AUTOTUNE,
    ).prefetch(tf.data.AUTOTUNE)

    positions = []

    for position in tqdm(
        dataset,
        desc=f"Positions: {filename}",
    ):
        positions.append(
            position.numpy()[:2]
        )

    positions = np.asarray(
        positions,
        dtype=np.float64,
    )

    file_positions.append(
        positions
    )

    print(
        f"  {filename}: "
        f"{len(positions)} samples"
    )

eligible_positions = []
eligible_file_indices = []
eligible_local_indices = []

for file_index, positions in enumerate(
    file_positions
):
    # Eligibility is based on 49 available successors so that the selected
    # anchors are identical across all three mobility settings.
    last_eligible = (
        len(positions)
        - 1
        - N_NEIGHBOURS
    )

    if last_eligible < 0:
        continue

    n_eligible = (
        last_eligible + 1
    )

    eligible_positions.extend(
        positions[:n_eligible].tolist()
    )

    eligible_file_indices.extend(
        [file_index] * n_eligible
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
        f"Only {len(eligible_positions)} eligible samples "
        f"are available, but {TARGET_ANCHORS} anchors "
        "are required."
    )

rng = np.random.default_rng(
    SEED
)

sampled_indices = stratified_sample_exact(
    eligible_positions,
    TARGET_ANCHORS,
    GRID_SIZE,
    rng,
)

anchor_file_indices = (
    eligible_file_indices[
        sampled_indices
    ]
)

anchor_local_indices = (
    eligible_local_indices[
        sampled_indices
    ]
)

pos_anchors = (
    eligible_positions[
        sampled_indices
    ]
)

print(
    f"\nAnchors selected      : "
    f"{TARGET_ANCHORS}"
)

print(
    f"Anchor eligibility    : "
    f"{N_NEIGHBOURS} successors"
)

print(
    f"Samples per experiment: "
    f"{N_WINDOW} "
    f"(anchor + {N_WINDOW - 1})"
)

print(
    "Per-file anchors     :",
    np.bincount(
        anchor_file_indices,
        minlength=len(TFRECORD_FILES),
    ).tolist(),
)

print(
    f"X range: "
    f"[{pos_anchors[:, 0].min():.3f}, "
    f"{pos_anchors[:, 0].max():.3f}]"
)

print(
    f"Y range: "
    f"[{pos_anchors[:, 1].min():.3f}, "
    f"{pos_anchors[:, 1].max():.3f}]"
)


# Pass 2: read each required CSI sample once and fill all six-slot windows
print("\n" + "=" * 78)
print("PASS 2: LOADING SIX-SAMPLE COMPLEX CFR WINDOWS")
print("=" * 78)

CFR_WIN = np.empty(
    (
        TARGET_ANCHORS,
        N_WINDOW,
        N_ANTENNAS,
        1024,
    ),
    dtype=np.complex64,
)

filled = np.zeros(
    (
        TARGET_ANCHORS,
        N_WINDOW,
    ),
    dtype=bool,
)

for file_index, (
    tfrecord_path,
    offset_path,
) in enumerate(
    zip(
        TFRECORD_FILES,
        OFFSET_FILES,
    )
):
    anchor_slots = np.where(
        anchor_file_indices
        == file_index
    )[0]

    if len(anchor_slots) == 0:
        continue

    local_to_destinations = {}

    for anchor_slot in anchor_slots:
        anchor_local = int(
            anchor_local_indices[
                anchor_slot
            ]
        )

        for window_slot in range(
            N_WINDOW
        ):
            local_index = (
                anchor_local
                + window_slot
            )

            local_to_destinations.setdefault(
                local_index,
                [],
            ).append(
                (
                    int(anchor_slot),
                    window_slot,
                )
            )

    maximum_needed_local = max(
        local_to_destinations
    )

    with open(
        offset_path,
        "r",
    ) as file:
        offsets = json.load(file)

    dataset = tf.data.TFRecordDataset(
        tfrecord_path
    )

    total_samples = len(
        file_positions[file_index]
    )

    for local_index, serialized_record in enumerate(
        tqdm(
            dataset,
            total=total_samples,
            desc=(
                "CSI: "
                + os.path.basename(
                    tfrecord_path
                )
            ),
        )
    ):
        if local_index > maximum_needed_local:
            break

        destinations = (
            local_to_destinations.get(
                local_index
            )
        )

        if destinations is None:
            continue

        csi_iq = parse_csi(
            serialized_record
        ).numpy()

        csi_complex = to_complex(
            csi_iq
        )

        csi_shifted = np.fft.fftshift(
            csi_complex,
            axes=-1,
        )

        csi_rev2_normalized = (
            normalize_rev2_sto(
                csi_shifted
            )
        )

        csi_calibrated = (
            apply_calibration(
                csi_rev2_normalized,
                offsets,
            )
        )

        cfr_array_a = (
            csi_calibrated[
                ARRAY_A_CHANNELS,
                :,
            ]
        )

        for (
            anchor_slot,
            window_slot,
        ) in destinations:
            CFR_WIN[
                anchor_slot,
                window_slot,
            ] = cfr_array_a

            filled[
                anchor_slot,
                window_slot,
            ] = True

if not np.all(filled):
    missing = np.argwhere(
        ~filled
    )

    raise RuntimeError(
        f"Failed to load {len(missing)} "
        "anchor/window entries. "
        f"First missing entries: "
        f"{missing[:10].tolist()}"
    )

print(
    f"\nCFR window shape: "
    f"{CFR_WIN.shape} | "
    f"dtype: {CFR_WIN.dtype}"
)

print(
    "Each window contains the calibrated complex CFR "
    "of the anchor and its five immediate successors."
)


# centred CIR and fixed 18 taps for each temporal slot
print("\n" + "=" * 78)
print("STEP 3: COMPUTING SIX-SAMPLE CIR WINDOWS")
print("=" * 78)

CIR_WIN = np.empty(
    (
        TARGET_ANCHORS,
        N_WINDOW,
        N_ANTENNAS,
        N_TAPS,
    ),
    dtype=np.complex64,
)

for window_slot in tqdm(
    range(N_WINDOW),
    desc="Temporal slots",
):
    cir_full = shifted_cfr_to_centered_cir(
        CFR_WIN[
            :,
            window_slot,
            :,
            :,
        ]
    )

    CIR_WIN[
        :,
        window_slot,
        :,
        :,
    ] = cir_full[
        :,
        :,
        TAP_START:TAP_STOP,
    ]

    del cir_full
    gc.collect()

print(
    f"Fixed tap interval : "
    f"[{TAP_START}:{TAP_STOP}]"
)

print(
    f"CIR window shape   : "
    f"{CIR_WIN.shape}"
)


# complex CFR, CIR, and 2D beam-CIR feature representations
print("\n" + "=" * 78)
print("STEP 4: BUILDING COMPLEX SIX-SAMPLE FEATURES")
print("=" * 78)

# Restore the physical 2 x 4 array for every temporal sample.
CIR_GRID_WIN = CIR_WIN.reshape(
    TARGET_ANCHORS,
    N_WINDOW,
    2,
    4,
    N_TAPS,
)

# 2D spatial transform across the two physical array dimensions.
BEAM_GRID_WIN = np.fft.fftshift(
    np.fft.fft2(
        CIR_GRID_WIN,
        axes=(2, 3),
        norm="ortho",
    ),
    axes=(2, 3),
).astype(np.complex64)

BEAM_CIR_WIN = BEAM_GRID_WIN.reshape(
    TARGET_ANCHORS,
    N_WINDOW,
    N_ANTENNAS,
    N_TAPS,
)


def to_window_feature(
    window_data,
):
    """Reorder axes for feature-distance evaluation."""
    return window_data.transpose(
        0,
        2,
        3,
        1,
    )


C_CFR_FEAT = to_window_feature(
    CFR_WIN
)

C_CIR_FEAT = to_window_feature(
    CIR_WIN
)

C_BEAM_CIR_FEAT = to_window_feature(
    BEAM_CIR_WIN
)

del (
    CFR_WIN,
    CIR_WIN,
    CIR_GRID_WIN,
    BEAM_GRID_WIN,
    BEAM_CIR_WIN,
)

gc.collect()

print(
    f"Complex CFR feature      : "
    f"{C_CFR_FEAT.shape}"
)

print(
    f"Complex CIR feature      : "
    f"{C_CIR_FEAT.shape}"
)

print(
    f"Complex beam-CIR feature : "
    f"{C_BEAM_CIR_FEAT.shape}"
)


# physical distance matrix between anchor positions
print("\n" + "=" * 78)
print("STEP 5: COMPUTING PHYSICAL DISTANCE MATRIX")
print("=" * 78)

position_difference = (
    pos_anchors[:, None, :]
    - pos_anchors[None, :, :]
)

D_PHYS = np.sqrt(
    np.sum(
        position_difference**2,
        axis=-1,
    )
).astype(np.float64)

np.fill_diagonal(
    D_PHYS,
    0.0,
)

del position_difference
gc.collect()

print(
    f"D_phys shape: {D_PHYS.shape} | "
    f"min nonzero="
    f"{D_PHYS[D_PHYS > 0].min():.4f} m | "
    f"max={D_PHYS.max():.4f} m"
)


# SqASD over the six corresponding temporal samples
def compute_sqasd_window_distance_matrix(
    features,
    distance_type,
    average_axis,
):
    """Compute RASD over corresponding temporal samples."""
    n_samples = features.shape[0]
    n_window = features.shape[3]

    if average_axis == "row":
        n_outer = features.shape[1]

        outer_slices = (
            features[
                :,
                outer_index,
                :,
                :,
            ]
            for outer_index
            in range(n_outer)
        )

    elif average_axis == "col":
        n_outer = features.shape[2]

        outer_slices = (
            features[
                :,
                :,
                outer_index,
                :,
            ]
            for outer_index
            in range(n_outer)
        )

    else:
        raise ValueError(
            "average_axis must be "
            "'row' or 'col'."
        )

    total_terms = (
        n_outer
        * n_window
    )

    squared_distance_sum = np.zeros(
        (
            n_samples,
            n_samples,
        ),
        dtype=np.float64,
    )

    for feature_slice in tqdm(
        outer_slices,
        total=n_outer,
        desc=(
            f"  {distance_type} "
            f"({average_axis})"
        ),
        leave=False,
    ):
        # From (N, vector_dimension, N_WINDOW)
        # to   (N, N_WINDOW, vector_dimension).
        temporal_vectors = feature_slice.transpose(
            0,
            2,
            1,
        )

        for window_slot in range(
            n_window
        ):
            vectors = temporal_vectors[
                :,
                window_slot,
                :,
            ].astype(
                np.complex128,
                copy=False,
            )

            norm_squared = np.sum(
                np.abs(vectors) ** 2,
                axis=1,
            )

            inner_product = (
                vectors
                @ vectors.conj().T
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
                vectors
                / safe_norms
            )

            normalized_inner = (
                normalized_vectors
                @ normalized_vectors.conj().T
            )

            absolute_normalized_inner = np.clip(
                np.abs(
                    normalized_inner
                ),
                0.0,
                1.0,
            )

            if distance_type == "euclidean":
                distance_squared = (
                    norm_squared[:, None]
                    + norm_squared[None, :]
                    - 2.0
                    * np.real(
                        inner_product
                    )
                )

                squared_distance = np.maximum(
                    distance_squared,
                    0.0,
                )

            elif distance_type == "norm_euclidean":
                distance_squared = (
                    2.0
                    - 2.0
                    * np.real(
                        normalized_inner
                    )
                )

                squared_distance = np.maximum(
                    distance_squared,
                    0.0,
                )

            elif distance_type == "norm_geodesic_sphere":
                distance = np.arccos(
                    np.clip(
                        np.real(
                            normalized_inner
                        ),
                        -1.0,
                        1.0,
                    )
                )

                squared_distance = (
                    distance**2
                )

            elif distance_type == "global_phase_chordal":
                absolute_inner = np.abs(
                    inner_product
                )

                distance_squared = (
                    0.5
                    * (
                        norm_squared[:, None] ** 2
                        + norm_squared[None, :] ** 2
                    )
                    - absolute_inner**2
                )

                squared_distance = np.maximum(
                    distance_squared,
                    0.0,
                )

            elif distance_type == "global_phase_bw":
                absolute_inner = np.abs(
                    inner_product
                )

                distance_squared = (
                    0.5
                    * (
                        norm_squared[:, None]
                        + norm_squared[None, :]
                    )
                    - absolute_inner
                )

                squared_distance = np.maximum(
                    distance_squared,
                    0.0,
                )


            elif distance_type == "norm_chordal":
                squared_distance = np.maximum(
                    1.0
                    - absolute_normalized_inner**2,
                    0.0,
                )

            elif distance_type == "norm_geodesic_grass":
                distance = np.arccos(
                    absolute_normalized_inner
                )

                squared_distance = (
                    distance**2
                )

            elif distance_type == "norm_bw":
                squared_distance = np.maximum(
                    1.0
                    - absolute_normalized_inner,
                    0.0,
                )

            else:
                raise ValueError(
                    f"Unknown distance type: "
                    f"{distance_type}"
                )

            np.fill_diagonal(
                squared_distance,
                0.0,
            )

            squared_distance_sum += (
                squared_distance
            )

    distance_matrix = np.sqrt(
        squared_distance_sum
        / float(total_terms)
    )

    np.fill_diagonal(
        distance_matrix,
        0.0,
    )

    return distance_matrix


# quality metrics
def rank_matrix(
    distance_matrix,
):
    without_self = (
        distance_matrix.copy()
    )

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


def trustworthiness(
    physical_ranks,
    channel_ranks,
    neighbourhood_size,
):
    n_samples = (
        physical_ranks.shape[0]
    )

    false_neighbours = (
        (
            channel_ranks
            <= neighbourhood_size
        )
        & (
            physical_ranks
            > neighbourhood_size
        )
    )

    penalty = np.where(
        false_neighbours,
        (
            physical_ranks
            - neighbourhood_size
        ),
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

    return float(
        1.0
        - normalization
        * penalty.sum()
    )


def continuity(
    physical_ranks,
    channel_ranks,
    neighbourhood_size,
):
    n_samples = (
        physical_ranks.shape[0]
    )

    missing_neighbours = (
        (
            physical_ranks
            <= neighbourhood_size
        )
        & (
            channel_ranks
            > neighbourhood_size
        )
    )

    penalty = np.where(
        missing_neighbours,
        (
            channel_ranks
            - neighbourhood_size
        ),
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

    return float(
        1.0
        - normalization
        * penalty.sum()
    )


def kruskal_stress(
    physical_distances,
    channel_distances,
):
    upper_triangle = np.triu_indices(
        physical_distances.shape[0],
        k=1,
    )

    physical_vector = (
        physical_distances[
            upper_triangle
        ].astype(np.float64)
    )

    channel_vector = (
        channel_distances[
            upper_triangle
        ].astype(np.float64)
    )

    channel_energy = np.sum(
        channel_vector**2
    )

    physical_energy = np.sum(
        physical_vector**2
    )

    if (
        channel_energy <= 0.0
        or physical_energy <= 0.0
    ):
        return (
            float("nan"),
            float("nan"),
        )

    scale = (
        np.sum(
            physical_vector
            * channel_vector
        )
        / channel_energy
    )

    stress = np.sqrt(
        np.sum(
            (
                physical_vector
                - scale
                * channel_vector
            ) ** 2
        )
        / physical_energy
    )

    return (
        float(stress),
        float(scale),
    )


def neighbourhood_preservation_ratio(
    physical_ranks,
    channel_ranks,
    neighbourhood_size,
):
    physical_neighbours = (
        physical_ranks
        <= neighbourhood_size
    )

    channel_neighbours = (
        channel_ranks
        <= neighbourhood_size
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


# distances and feature configurations
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
        "(complex CFR, Array A, average over 6, SqASD)",
        C_CFR_FEAT,
        "col",
        "antenna_per_frequency",
    ),
    (
        "Antenna-per-Delay "
        "(complex CIR, Array A, fixed 18 taps, average over 6, SqASD)",
        C_CIR_FEAT,
        "col",
        "antenna_per_delay",
    ),
    (
        "Delay-per-Antenna "
        "(complex CIR, Array A, fixed 18 taps, average over 6, SqASD)",
        C_CIR_FEAT,
        "row",
        "delay_per_antenna",
    ),
    (
        "Delay-per-Beam "
        "(complex 2D beam-CIR, Array A, fixed 18 taps, average over 6, SqASD)",
        C_BEAM_CIR_FEAT,
        "row",
        "delay_per_beam",
    ),
]


# main computation
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
        f"Feature shape: "
        f"{features.shape} | "
        f"average_axis="
        f"{average_axis}"
    )

    print("=" * 78)

    feature_results = {}

    for (
        distance_type,
        distance_label,
    ) in zip(
        DISTANCE_TYPES,
        DISTANCE_LABELS,
    ):
        print(
            f"\nComputing: "
            f"{distance_label}"
        )

        D_CHANNEL = (
            compute_sqasd_window_distance_matrix(
                features,
                distance_type,
                average_axis,
            )
        )

        metrics = compute_all_metrics(
            D_PHYS,
            D_CHANNEL,
            J,
        )

        feature_results[
            distance_type
        ] = metrics

        del D_CHANNEL
        gc.collect()

    all_results[
        feature_label
    ] = feature_results


# Final summary
print("\n" + "=" * 120)

print(
    "FINAL SUMMARY — DICHASUS rev2, "
    "Array A, complex values, "
    "average over 6 samples (anchor + 5), "
    "SqASD, fixed taps [509:527]"
)

print("=" * 120)

for (
    _,
    _,
    _,
    feature_label,
) in CONFIGURATIONS:
    print(
        f"\nFeature: "
        f"{feature_label}"
    )

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

    for (
        distance_type,
        distance_label,
    ) in zip(
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
