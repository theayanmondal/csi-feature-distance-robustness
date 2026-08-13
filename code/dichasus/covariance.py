
import gc
import json
import os

import numpy as np
import tensorflow as tf
from tqdm import tqdm


# Paths and experiment settings
BASE_PATH = "data/dichasus"
OUTPUT_DIR = "outputs/dichasus/covariance"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TFRECORD_FILES = [
    os.path.join(BASE_PATH, f"dichasus-cf0{i}.tfrecords")
    for i in range(2, 8)
]
OFFSET_FILES = [
    os.path.join(BASE_PATH, f"reftx-offsets-dichasus-cf0{i}.json")
    for i in range(2, 8)
]

ARRAY_A_GRID = np.array(
    [[6, 2, 16, 18], [28, 5, 10, 14]],
    dtype=np.int32,
)
ARRAY_A_CHANNELS = ARRAY_A_GRID.ravel()
N_ANTENNAS = len(ARRAY_A_CHANNELS)

TARGET_ANCHORS = 4000
ELIGIBILITY_SUCCESSORS = 49
GRID_SIZE = 20
SEED = 42
J = 100

TAP_START = 509
TAP_STOP = 527
N_TAPS = TAP_STOP - TAP_START
WINDOW_SIZES = (1, 6, 50)

LOG_EIGENVALUE_RELATIVE_FLOOR = 1e-6
ABSOLUTE_EIGENVALUE_FLOOR = 1e-12
DISTANCE_ROW_BLOCK = 500
BURES_PAIR_BLOCK = 256

assert N_ANTENNAS == 8
assert N_TAPS == 18


# TFRecord parsing and DICHASUS rev2 preprocessing
POSITION_DESCRIPTION = {
    "pos-tachy": tf.io.FixedLenFeature([], tf.string, default_value=""),
}
CSI_DESCRIPTION = {
    "csi": tf.io.FixedLenFeature([], tf.string, default_value=""),
}


def parse_position(serialized_record):
    record = tf.io.parse_single_example(serialized_record, POSITION_DESCRIPTION)
    position = tf.io.parse_tensor(record["pos-tachy"], out_type=tf.float64)
    return tf.ensure_shape(position, (3,))


def parse_csi(serialized_record):
    record = tf.io.parse_single_example(serialized_record, CSI_DESCRIPTION)
    csi = tf.io.parse_tensor(record["csi"], out_type=tf.float32)
    return tf.ensure_shape(csi, (32, 1024, 2))


def to_complex(csi_iq):
    return (csi_iq[..., 0] + 1j * csi_iq[..., 1]).astype(np.complex64)


def normalize_rev2_sto(csi_shifted):
    adjacent_products = csi_shifted[:, 1:] * np.conj(csi_shifted[:, :-1])
    phase_increment = np.float32(np.angle(np.sum(adjacent_products)))
    k = np.arange(csi_shifted.shape[-1], dtype=np.float32)
    correction = np.exp(-1j * phase_increment * k).astype(np.complex64)
    return (csi_shifted * correction[None, :]).astype(np.complex64)


def apply_calibration(csi_rev2_normalized, offsets):
    n_channels, n_subcarriers = csi_rev2_normalized.shape
    sto = np.asarray(offsets["sto"], dtype=np.float32)
    cpo = np.asarray(offsets["cpo"], dtype=np.float32)

    if sto.shape != (n_channels,) or cpo.shape != (n_channels,):
        raise ValueError(
            f"Expected STO/CPO shape ({n_channels},), got {sto.shape}/{cpo.shape}."
        )

    k = np.arange(n_subcarriers, dtype=np.float32)
    phase = sto[:, None] * (2.0 * np.pi * k[None, :] / n_subcarriers) + cpo[:, None]
    return (csi_rev2_normalized * np.exp(1j * phase)).astype(np.complex64)


def shifted_cfr_to_centered_cir(cfr_shifted):
    return np.fft.fftshift(
        np.fft.ifft(
            np.fft.ifftshift(cfr_shifted, axes=-1),
            axis=-1,
        ),
        axes=-1,
    ).astype(np.complex64)


def hermitian_part(matrix):
    return 0.5 * (matrix + matrix.conj().T)


def sample_spatial_covariance(csi_iq, offsets):
    csi_shifted = np.fft.fftshift(to_complex(csi_iq), axes=-1)
    csi_rev2 = normalize_rev2_sto(csi_shifted)
    csi_calibrated = apply_calibration(csi_rev2, offsets)
    cfr_array_a = csi_calibrated[ARRAY_A_CHANNELS, :]
    cir_full = shifted_cfr_to_centered_cir(cfr_array_a)
    cir_taps = cir_full[:, TAP_START:TAP_STOP].astype(np.complex128, copy=False)

    if cir_taps.shape != (N_ANTENNAS, N_TAPS):
        raise RuntimeError(f"Unexpected CIR shape: {cir_taps.shape}")

    covariance = (cir_taps @ cir_taps.conj().T) / float(N_TAPS)
    return hermitian_part(covariance)


# Anchor selection
def stratified_sample_exact(positions, target, grid_size, rng):
    x_edges = np.linspace(positions[:, 0].min(), positions[:, 0].max(), grid_size + 1)
    y_edges = np.linspace(positions[:, 1].min(), positions[:, 1].max(), grid_size + 1)

    cells = {}
    for row in range(grid_size):
        for col in range(grid_size):
            in_x = (positions[:, 0] >= x_edges[col]) & (positions[:, 0] < x_edges[col + 1])
            in_y = (positions[:, 1] >= y_edges[row]) & (positions[:, 1] < y_edges[row + 1])
            if col == grid_size - 1:
                in_x |= positions[:, 0] == x_edges[col + 1]
            if row == grid_size - 1:
                in_y |= positions[:, 1] == y_edges[row + 1]
            indices = np.where(in_x & in_y)[0]
            if len(indices) > 0:
                cells[(row, col)] = indices

    fair_share = target // len(cells)
    selected = []
    remaining = {}

    for cell, indices in cells.items():
        count = min(fair_share, len(indices))
        chosen = rng.choice(indices, size=count, replace=False)
        selected.extend(chosen.tolist())
        leftover = np.setdiff1d(indices, chosen)
        if len(leftover) > 0:
            remaining[cell] = leftover

    shortfall = target - len(selected)
    if shortfall > 0:
        for _, leftover in sorted(remaining.items(), key=lambda item: len(item[1]), reverse=True):
            if shortfall <= 0:
                break
            count = min(shortfall, len(leftover))
            chosen = rng.choice(leftover, size=count, replace=False)
            selected.extend(chosen.tolist())
            shortfall -= count

    if len(selected) != target:
        raise RuntimeError(f"Selected {len(selected)} anchors, expected {target}.")

    return np.sort(np.asarray(selected, dtype=np.int64))


def select_anchors():
    eligible_positions = []
    eligible_file_indices = []
    eligible_local_indices = []

    for file_index, tfrecord_path in enumerate(TFRECORD_FILES):
        positions = []
        dataset = tf.data.TFRecordDataset(tfrecord_path)
        for position in tqdm(
            dataset.map(parse_position, num_parallel_calls=tf.data.AUTOTUNE),
            desc=f"Positions {os.path.basename(tfrecord_path)}",
            leave=False,
        ):
            positions.append(position.numpy()[:2])

        positions = np.asarray(positions, dtype=np.float64)
        n_eligible = max(0, len(positions) - ELIGIBILITY_SUCCESSORS)

        eligible_positions.extend(positions[:n_eligible].tolist())
        eligible_file_indices.extend([file_index] * n_eligible)
        eligible_local_indices.extend(range(n_eligible))

    eligible_positions = np.asarray(eligible_positions, dtype=np.float64)
    eligible_file_indices = np.asarray(eligible_file_indices, dtype=np.int32)
    eligible_local_indices = np.asarray(eligible_local_indices, dtype=np.int32)

    rng = np.random.default_rng(SEED)
    sampled = stratified_sample_exact(
        eligible_positions,
        TARGET_ANCHORS,
        GRID_SIZE,
        rng,
    )

    return (
        eligible_positions[sampled],
        eligible_file_indices[sampled],
        eligible_local_indices[sampled],
    )


# Spatial covariance construction
def compute_spatial_covariances(anchor_file_index, anchor_local_index, window_size):
    accumulator = np.zeros(
        (TARGET_ANCHORS, N_ANTENNAS, N_ANTENNAS),
        dtype=np.complex128,
    )
    counts = np.zeros(TARGET_ANCHORS, dtype=np.int16)

    for file_index, (tfrecord_path, offset_path) in enumerate(zip(TFRECORD_FILES, OFFSET_FILES)):
        slots = np.where(anchor_file_index == file_index)[0]
        if len(slots) == 0:
            continue

        local_to_destinations = {}
        for anchor_slot in slots:
            start = int(anchor_local_index[anchor_slot])
            for offset in range(window_size):
                local_index = start + offset
                local_to_destinations.setdefault(local_index, []).append(anchor_slot)

        with open(offset_path, "r", encoding="utf-8") as handle:
            offsets = json.load(handle)

        required = set(local_to_destinations)
        loaded = 0
        dataset = tf.data.TFRecordDataset(tfrecord_path)

        for local_index, csi_tensor in enumerate(
            tqdm(
                dataset.map(parse_csi, num_parallel_calls=tf.data.AUTOTUNE),
                desc=f"Covariance window={window_size}, {os.path.basename(tfrecord_path)}",
                leave=False,
            )
        ):
            if local_index not in required:
                continue

            covariance = sample_spatial_covariance(csi_tensor.numpy(), offsets)
            for anchor_slot in local_to_destinations[local_index]:
                accumulator[anchor_slot] += covariance
                counts[anchor_slot] += 1

            loaded += 1
            if loaded == len(required):
                break

        if loaded != len(required):
            raise RuntimeError(
                f"{os.path.basename(tfrecord_path)}: loaded {loaded} of "
                f"{len(required)} required samples."
            )

        del local_to_destinations, required
        gc.collect()

    invalid = np.where(counts != window_size)[0]
    if len(invalid) > 0:
        raise RuntimeError(
            f"{len(invalid)} anchors did not receive exactly {window_size} "
            f"contributions. First invalid anchors: {invalid[:10].tolist()}"
        )

    covariance = accumulator / float(window_size)
    covariance = 0.5 * (covariance + covariance.conj().swapaxes(1, 2))
    return covariance.astype(np.complex64)


def trace_normalize(covariance):
    traces = np.real(np.trace(covariance, axis1=1, axis2=2))
    if np.any(traces <= 0):
        raise RuntimeError("Encountered a non-positive covariance trace.")
    return (covariance / traces[:, None, None]).astype(np.complex64)


# Covariance distances
def hermitize_batch(covariance_matrices):
    """Return the Hermitian part of each matrix."""
    return (
        0.5
        * (
            covariance_matrices
            + covariance_matrices.conj().swapaxes(1, 2)
        )
    ).astype(np.complex64)


def flatten_covariances(covariance_matrices):
    """Flatten a batch of covariance matrices."""
    n_samples, dimension, _ = covariance_matrices.shape
    return covariance_matrices.reshape(
        n_samples,
        dimension * dimension,
    ).astype(np.complex64, copy=False)


def nonnegative_eigendecomposition(covariance_matrices):
    """Eigendecompose Hermitian matrices and clip numerical negative eigenvalues."""
    eigenvalues, eigenvectors = np.linalg.eigh(
        covariance_matrices.astype(np.complex128, copy=False)
    )
    eigenvalues = np.maximum(eigenvalues, 0.0)
    return (
        eigenvalues.astype(np.float64),
        eigenvectors.astype(np.complex128),
    )


def log_eigendecomposition(covariance_matrices):
    """Apply the trace-relative eigenvalue floor used for the matrix logarithm."""
    covariance_matrices = hermitize_batch(covariance_matrices)

    eigenvalues, eigenvectors = np.linalg.eigh(
        covariance_matrices.astype(np.complex128, copy=False)
    )

    dimension = covariance_matrices.shape[1]
    traces = np.real(
        np.trace(covariance_matrices, axis1=1, axis2=2)
    ).astype(np.float64)

    average_eigenvalue = traces / float(dimension)
    eigenvalue_floor = np.maximum(
        LOG_EIGENVALUE_RELATIVE_FLOOR * average_eigenvalue,
        ABSOLUTE_EIGENVALUE_FLOOR,
    )

    eigenvalues = np.maximum(
        eigenvalues,
        eigenvalue_floor[:, None],
    )

    return (
        eigenvalues.astype(np.float64),
        eigenvectors.astype(np.complex128),
    )


def covariance_euclidean(covariance_matrices):
    """Compute Frobenius distances in row blocks."""
    flattened = flatten_covariances(covariance_matrices)
    squared_norms = np.sum(
        np.abs(flattened) ** 2,
        axis=1,
    ).astype(np.float64)

    n_samples = covariance_matrices.shape[0]
    distance_matrix = np.empty(
        (n_samples, n_samples),
        dtype=np.float32,
    )

    for start in tqdm(
        range(0, n_samples, DISTANCE_ROW_BLOCK),
        desc="    Euclidean rows",
        leave=False,
    ):
        stop = min(start + DISTANCE_ROW_BLOCK, n_samples)
        block = flattened[start:stop]

        inner_product = np.real(
            block @ flattened.conj().T
        ).astype(np.float64)

        squared_distance = (
            squared_norms[start:stop, None]
            + squared_norms[None, :]
            - 2.0 * inner_product
        )

        distance_matrix[start:stop] = np.sqrt(
            np.maximum(squared_distance, 0.0)
        ).astype(np.float32)

    distance_matrix = 0.5 * (
        distance_matrix + distance_matrix.T
    )
    np.fill_diagonal(distance_matrix, 0.0)
    return distance_matrix


def covariance_log_euclidean(covariance_matrices):
    """Compute Log-Euclidean covariance distances."""
    eigenvalues, eigenvectors = log_eigendecomposition(covariance_matrices)
    log_eigenvalues = np.log(eigenvalues)

    log_covariances = np.einsum(
        "nij,nj,nkj->nik",
        eigenvectors,
        log_eigenvalues,
        eigenvectors.conj(),
        optimize=True,
    ).astype(np.complex64)

    del eigenvalues, eigenvectors, log_eigenvalues
    gc.collect()

    distance_matrix = covariance_euclidean(log_covariances)

    del log_covariances
    gc.collect()
    return distance_matrix


def covariance_bures(covariance_matrices):
    """Compute Bures-Wasserstein covariance distances in pair blocks."""
    covariance_matrices = hermitize_batch(covariance_matrices)
    n_samples, dimension, _ = covariance_matrices.shape

    eigenvalues, eigenvectors = nonnegative_eigendecomposition(
        covariance_matrices
    )
    square_root_eigenvalues = np.sqrt(eigenvalues)

    square_root_matrices = np.einsum(
        "nij,nj,nkj->nik",
        eigenvectors,
        square_root_eigenvalues,
        eigenvectors.conj(),
        optimize=True,
    ).astype(np.complex64)

    traces = np.real(
        np.trace(covariance_matrices, axis1=1, axis2=2)
    ).astype(np.float64)

    del eigenvalues, eigenvectors, square_root_eigenvalues
    gc.collect()

    distance_matrix = np.zeros(
        (n_samples, n_samples),
        dtype=np.float32,
    )

    for first_index in tqdm(
        range(n_samples),
        desc=f"    Bures-Wasserstein ({dimension}x{dimension})",
        leave=False,
    ):
        square_root_first = square_root_matrices[first_index].astype(
            np.complex128,
            copy=False,
        )

        for second_start in range(
            first_index,
            n_samples,
            BURES_PAIR_BLOCK,
        ):
            second_stop = min(
                second_start + BURES_PAIR_BLOCK,
                n_samples,
            )

            second_covariances = covariance_matrices[
                second_start:second_stop
            ].astype(np.complex128, copy=False)

            middle_matrices = np.einsum(
                "ab,nbc,cd->nad",
                square_root_first,
                second_covariances,
                square_root_first,
                optimize=True,
            )
            middle_matrices = 0.5 * (
                middle_matrices
                + middle_matrices.conj().swapaxes(1, 2)
            )

            middle_eigenvalues = np.linalg.eigvalsh(middle_matrices)
            middle_eigenvalues = np.maximum(middle_eigenvalues, 0.0)

            squared_distance = (
                traces[first_index]
                + traces[second_start:second_stop]
                - 2.0
                * np.sum(
                    np.sqrt(middle_eigenvalues),
                    axis=1,
                )
            )

            distances = np.sqrt(
                np.maximum(squared_distance, 0.0)
            ).astype(np.float32)

            distance_matrix[
                first_index,
                second_start:second_stop,
            ] = distances
            distance_matrix[
                second_start:second_stop,
                first_index,
            ] = distances

    np.fill_diagonal(distance_matrix, 0.0)

    del square_root_matrices
    gc.collect()
    return distance_matrix


# Performance metrics
def physical_distance_matrix(positions):
    diff = positions[:, None, :] - positions[None, :, :]
    distance = np.sqrt(np.sum(diff**2, axis=-1))
    np.fill_diagonal(distance, 0.0)
    return distance


def rank_matrix(distance):
    no_self = distance.copy()
    np.fill_diagonal(no_self, np.inf)
    return np.argsort(np.argsort(no_self, axis=1), axis=1) + 1


def trustworthiness(r_phys, r_chan, k):
    n = r_phys.shape[0]
    mask = (r_chan <= k) & (r_phys > k)
    penalty = np.where(mask, r_phys - k, 0)
    scale = 2.0 / (n * k * (2 * n - 3 * k - 1))
    return float(1.0 - scale * penalty.sum())


def continuity(r_phys, r_chan, k):
    n = r_phys.shape[0]
    mask = (r_phys <= k) & (r_chan > k)
    penalty = np.where(mask, r_chan - k, 0)
    scale = 2.0 / (n * k * (2 * n - 3 * k - 1))
    return float(1.0 - scale * penalty.sum())


def neighbourhood_preservation_ratio(r_phys, r_chan, k):
    intersection = ((r_phys <= k) & (r_chan <= k)).sum(axis=1)
    return float(intersection.mean() / k)


def kruskal_stress(d_phys, d_chan):
    idx = np.triu_indices(d_phys.shape[0], k=1)
    target = d_phys[idx]
    observed = d_chan[idx]
    denom = np.sum(observed**2)
    if denom <= 0:
        return float("nan")
    scale = np.sum(target * observed) / denom
    return float(np.sqrt(np.sum((target - scale * observed) ** 2) / np.sum(target**2)))


def compute_all_metrics(D_phys, D_chan, J):
    R_phys = rank_matrix(D_phys)
    R_chan = rank_matrix(D_chan)
    tw = trustworthiness(R_phys, R_chan, J)
    ct = continuity(R_phys, R_chan, J)
    ks = kruskal_stress(D_phys, D_chan)
    np_r = neighbourhood_preservation_ratio(R_phys, R_chan, J)
    return dict(TW=tw, CT=ct, KS=ks, NPR=np_r)


# Main
def evaluate_setting(anchor_positions, anchor_file_index, anchor_local_index, d_phys, window_size):
    covariance_raw = compute_spatial_covariances(
        anchor_file_index,
        anchor_local_index,
        window_size,
    )
    covariance_norm = trace_normalize(covariance_raw)

    settings = {
        "raw": covariance_raw,
        "trace_normalized": covariance_norm,
    }
    distance_functions = {
        "euclidean": covariance_euclidean,
        "log_euclidean": covariance_log_euclidean,
        "bures_wasserstein": covariance_bures,
    }

    results = {}
    save_dict = {
        "pos_anchors": anchor_positions,
        "metric_names": np.array(["TW", "CT", "KS", "NPR"]),
    }

    for normalization, covariance in settings.items():
        for distance_name, distance_fn in distance_functions.items():
            key = f"{normalization}_{distance_name}"
            print(f"Computing {key} ...")
            d_chan = distance_fn(covariance)
            metrics = compute_all_metrics(d_phys, d_chan, J)
            results[key] = metrics
            save_dict[f"D_{key}"] = d_chan
            save_dict[f"metrics_{key}"] = np.array(
                [metrics["TW"], metrics["CT"], metrics["KS"], metrics["NPR"]],
                dtype=np.float64,
            )

    print(f"\nSpatial covariance summary — window={window_size}")
    print(f"{'Setting / Distance':<42} {'TW':>8} {'CT':>8} {'NPR':>8} {'KS':>8}")
    print("-" * 80)
    for key, metrics in results.items():
        print(
            f"{key:<42} {metrics['TW']:>8.4f} {metrics['CT']:>8.4f} "
            f"{metrics['NPR']:>8.4f} {metrics['KS']:>8.4f}"
        )

    output_path = os.path.join(OUTPUT_DIR, f"spatial_covariance_window{window_size}.npz")
    np.savez(output_path, **save_dict)
    print(f"Saved: {output_path}\n")


def main():
    anchor_positions, anchor_file_index, anchor_local_index = select_anchors()
    d_phys = physical_distance_matrix(anchor_positions)

    print(f"Anchors: {TARGET_ANCHORS}")
    print(f"Array A channels: {ARRAY_A_CHANNELS.tolist()}")
    print(f"Fixed CIR taps: [{TAP_START}:{TAP_STOP}] ({N_TAPS} taps)")
    print("Covariance feature: spatial covariance from CIR taps")

    for window_size in WINDOW_SIZES:
        evaluate_setting(
            anchor_positions,
            anchor_file_index,
            anchor_local_index,
            d_phys,
            window_size,
        )


if __name__ == "__main__":
    main()
