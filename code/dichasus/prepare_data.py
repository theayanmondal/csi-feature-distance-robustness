import os

import numpy as np
import tensorflow as tf
from tqdm import tqdm


# Paths and settings
BASE_PATH = "data/dichasus"
OUTPUT_PATH = os.path.join(BASE_PATH, "anchor_metadata.npz")

TFRECORD_FILES = [
    os.path.join(BASE_PATH, f"dichasus-cf0{i}.tfrecords")
    for i in range(2, 8)
]

TARGET_ANCHORS = 4000
N_NEIGHBOURS = 49
GRID_SIZE = 20
SEED = 42


# Position parsing
POSITION_DESCRIPTION = {
    "pos-tachy": tf.io.FixedLenFeature([], tf.string, default_value=""),
}


def parse_position(serialized_record):
    record = tf.io.parse_single_example(serialized_record, POSITION_DESCRIPTION)
    position = tf.io.parse_tensor(record["pos-tachy"], out_type=tf.float64)
    return tf.ensure_shape(position, (3,))


def stratified_sample_exact(positions, target, grid_size, rng):
    """Select exactly `target` spatially stratified samples."""
    x_edges = np.linspace(positions[:, 0].min(), positions[:, 0].max(), grid_size + 1)
    y_edges = np.linspace(positions[:, 1].min(), positions[:, 1].max(), grid_size + 1)

    cell_members = {}
    for row in range(grid_size):
        for col in range(grid_size):
            in_x = ((positions[:, 0] >= x_edges[col]) &
                    (positions[:, 0] < x_edges[col + 1]))
            in_y = ((positions[:, 1] >= y_edges[row]) &
                    (positions[:, 1] < y_edges[row + 1]))

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
        chosen = rng.choice(indices, size=count, replace=False)
        selected.extend(chosen.tolist())
        leftover = np.setdiff1d(indices, chosen)
        if len(leftover) > 0:
            remaining[cell] = leftover

    shortfall = target - len(selected)
    if shortfall > 0:
        ordered_cells = sorted(remaining.items(), key=lambda item: len(item[1]), reverse=True)
        for _, leftover in ordered_cells:
            if shortfall <= 0:
                break
            count = min(shortfall, len(leftover))
            chosen = rng.choice(leftover, size=count, replace=False)
            selected.extend(chosen.tolist())
            shortfall -= count

    selected = np.sort(np.asarray(selected, dtype=np.int64))
    if len(selected) != target:
        raise RuntimeError(f"Could not select exactly {target} anchors; selected {len(selected)}.")
    return selected


# Build anchor metadata
all_positions = []
file_lengths = []

for path in TFRECORD_FILES:
    positions = []
    dataset = tf.data.TFRecordDataset(path).map(parse_position)
    for position in tqdm(dataset, desc=os.path.basename(path)):
        positions.append(position.numpy()[:2])
    positions = np.asarray(positions, dtype=np.float64)
    all_positions.append(positions)
    file_lengths.append(len(positions))

eligible_positions = []
eligible_file_indices = []
eligible_local_indices = []

for file_index, positions in enumerate(all_positions):
    last_eligible = len(positions) - 1 - N_NEIGHBOURS
    if last_eligible < 0:
        continue
    n_eligible = last_eligible + 1
    eligible_positions.extend(positions[:n_eligible].tolist())
    eligible_file_indices.extend([file_index] * n_eligible)
    eligible_local_indices.extend(range(n_eligible))

eligible_positions = np.asarray(eligible_positions, dtype=np.float64)
eligible_file_indices = np.asarray(eligible_file_indices, dtype=np.int32)
eligible_local_indices = np.asarray(eligible_local_indices, dtype=np.int32)

if len(eligible_positions) < TARGET_ANCHORS:
    raise RuntimeError(
        f"Only {len(eligible_positions)} eligible samples are available, "
        f"but {TARGET_ANCHORS} anchors are required."
    )

sampled = stratified_sample_exact(
    eligible_positions,
    TARGET_ANCHORS,
    GRID_SIZE,
    np.random.default_rng(SEED),
)

anchor_file_indices = eligible_file_indices[sampled]
anchor_local_indices = eligible_local_indices[sampled]
pos_anchors = eligible_positions[sampled]

np.savez(
    OUTPUT_PATH,
    anchor_file_indices=anchor_file_indices,
    anchor_local_indices=anchor_local_indices,
    pos_anchors=pos_anchors,
    file_lengths=np.asarray(file_lengths, dtype=np.int64),
    tfrecord_files=np.asarray([os.path.basename(p) for p in TFRECORD_FILES]),
)

print(f"Saved: {OUTPUT_PATH}")
print(f"Anchors selected: {len(pos_anchors)}")
print("Per-file anchor counts:", np.bincount(anchor_file_indices, minlength=len(TFRECORD_FILES)).tolist())
