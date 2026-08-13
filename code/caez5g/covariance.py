
import gc
import os

import numpy as np
from tqdm import tqdm


# Paths and experiment settings
DATA_DIR = "data/caez5g/mobility_dataset"
METADATA_PATH = os.path.join(DATA_DIR, "metadata.npz")
OUTPUT_DIR = "outputs/caez5g/covariance"
os.makedirs(OUTPUT_DIR, exist_ok=True)

ORU_IDX = 0
TAP_START = 1602
TAP_STOP = 1670
N_TAPS = TAP_STOP - TAP_START
N_ANTENNAS = 4
N_DMRS = 3
J = 100

WINDOW_SIZES = (1, 6, 50)
LOG_EIGENVALUE_RELATIVE_FLOOR = 1e-6
ABSOLUTE_EIGENVALUE_FLOOR = 1e-12
DISTANCE_ROW_BLOCK = 500
BURES_PAIR_BLOCK = 256

assert N_TAPS == 68


# Channel and covariance construction
def full_cfr_to_centered_cir(cfr):
    """Convert the full CAEZ-5G CFR to centred CIR."""
    return np.fft.fftshift(
        np.fft.ifft(cfr.astype(np.complex128), axis=1, norm="ortho"),
        axes=1,
    ).astype(np.complex64)


def hermitian_part(matrix):
    return 0.5 * (matrix + matrix.conj().T)


def sample_spatial_covariance(ch):
    """Compute spatial covariance from the retained CIR taps."""
    cir_full = full_cfr_to_centered_cir(ch)
    cir_taps = cir_full[:, TAP_START:TAP_STOP, :]  # (4, 68, 3)

    if cir_taps.shape != (N_ANTENNAS, N_TAPS, N_DMRS):
        raise RuntimeError(f"Unexpected CIR shape: {cir_taps.shape}")

    covariance = np.zeros((N_ANTENNAS, N_ANTENNAS), dtype=np.complex128)
    for dmrs_idx in range(N_DMRS):
        h = cir_taps[:, :, dmrs_idx].astype(np.complex128, copy=False)
        covariance += h @ h.conj().T

    covariance /= float(N_DMRS * N_TAPS)
    return hermitian_part(covariance)


def build_window_indices(metadata, window_size):
    anchor_pos = metadata["anchor_pos_in_needed"].astype(np.int64)
    n_anchors = len(anchor_pos)

    indices = np.empty((n_anchors, window_size), dtype=np.int64)
    indices[:, 0] = anchor_pos

    if window_size > 1:
        neighbours = metadata["window_indices_in_needed"].astype(np.int64)
        indices[:, 1:] = neighbours[:, : window_size - 1]

    return indices


def compute_spatial_covariances(metadata, window_size):
    """Compute spatial covariance for the selected temporal window."""
    window_indices = build_window_indices(metadata, window_size)
    shard_map = metadata["shard_map"].astype(np.int64)
    pos_in_shard = metadata["pos_in_shard"].astype(np.int64)
    n_shards = int(metadata["n_shards"])
    n_anchors = window_indices.shape[0]

    accumulator = np.zeros(
        (n_anchors, N_ANTENNAS, N_ANTENNAS),
        dtype=np.complex128,
    )
    counts = np.zeros(n_anchors, dtype=np.int16)

    # Each needed sample can contribute to multiple overlapping anchor windows.
    for shard_id in range(n_shards):
        local_to_destinations = {}

        for anchor_idx in range(n_anchors):
            for needed_idx in window_indices[anchor_idx]:
                if int(shard_map[needed_idx]) != shard_id:
                    continue
                row = int(pos_in_shard[needed_idx])
                local_to_destinations.setdefault(row, []).append(anchor_idx)

        if not local_to_destinations:
            continue

        shard_path = os.path.join(DATA_DIR, f"ch_est_shard_{shard_id:03d}.npz")
        shard_data = np.load(shard_path)["ch_est"]

        for row, destinations in tqdm(
            sorted(local_to_destinations.items()),
            desc=f"Covariance window={window_size}, shard={shard_id:03d}",
            leave=False,
        ):
            ch = shard_data[row, ORU_IDX][:, 0, :, :]  # (4, 3276, 3)
            covariance = sample_spatial_covariance(ch)

            for anchor_idx in destinations:
                accumulator[anchor_idx] += covariance
                counts[anchor_idx] += 1

        del shard_data

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


def npr(r_phys, r_chan, k):
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
    np_r = npr(R_phys, R_chan, J)
    return dict(TW=tw, CT=ct, KS=ks, NPR=np_r)


# Main
def evaluate_setting(metadata, d_phys, window_size):
    covariance_raw = compute_spatial_covariances(metadata, window_size)
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
        "pos_anchors": metadata["pos_anchors"],
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
    metadata = np.load(METADATA_PATH, allow_pickle=True)
    positions = metadata["pos_anchors"].astype(np.float64)
    d_phys = physical_distance_matrix(positions)

    print(f"Anchors: {len(positions)}")
    print(f"Fixed CIR taps: [{TAP_START}:{TAP_STOP}] ({N_TAPS} taps)")
    print("Covariance feature: spatial covariance from CIR taps")

    for window_size in WINDOW_SIZES:
        evaluate_setting(metadata, d_phys, window_size)


if __name__ == "__main__":
    main()
