import numpy as np
import os
import pickle
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# Paths
input_path = 'data/caez5g/raw'
npz_path   = 'data/caez5g/reference/caez_5g_indoor_full.npz'
output_dir = 'data/caez5g/mobility_dataset/'
os.makedirs(output_dir, exist_ok=True)

# Parameters
N_ANCHORS  = 4000
WINDOW_SIZE = 99
GRID_SIZE   = 10
SEED        = 42
SHARD_SIZE  = 2000
N_WORKERS   = 8

# Load full .npz
print("Loading full .npz...")
npz_data       = np.load(npz_path, allow_pickle=True)
npz_timestamps = npz_data['timestamps']
npz_pos        = npz_data['pos']
N_total        = len(npz_timestamps)
print(f"Total samples : {N_total}")
print(f"X: [{npz_pos[:,0].min():.3f}, {npz_pos[:,0].max():.3f}]")
print(f"Y: [{npz_pos[:,1].min():.3f}, {npz_pos[:,1].max():.3f}]")

# Sort everything by timestamp
print("\nSorting all samples by timestamp...")
sort_order     = np.argsort(npz_timestamps, kind='stable')
ts_sorted      = npz_timestamps[sort_order]
pos_sorted     = npz_pos[sort_order]
rank_in_sorted = np.empty(N_total, dtype=np.int64)
rank_in_sorted[sort_order] = np.arange(N_total)
print(f"Timestamps {'already sorted' if bool(np.all(np.diff(npz_timestamps) >= 0)) else 'sorted'}.")

# Build timestamp → filename map
print("\nScanning pickle files...")
pickle_files = sorted([f for f in os.listdir(input_path) if f.endswith('.pickle')])
print(f"Pickle files found: {len(pickle_files)}")
ts_to_fname = {float(f.split('_')[0]): f for f in pickle_files}

# Build 10×10 grid, collect per-cell populations
print(f"\nBuilding {GRID_SIZE}×{GRID_SIZE} grid and collecting cell populations...")
x_min, x_max = npz_pos[:, 0].min(), npz_pos[:, 0].max()
y_min, y_max = npz_pos[:, 1].min(), npz_pos[:, 1].max()
x_edges = np.linspace(x_min, x_max, GRID_SIZE + 1)
y_edges = np.linspace(y_min, y_max, GRID_SIZE + 1)

cell_indices = {}
for row in range(GRID_SIZE):
    for col in range(GRID_SIZE):
        in_x = (npz_pos[:, 0] >= x_edges[col]) & \
               (npz_pos[:, 0] <  x_edges[col + 1])
        in_y = (npz_pos[:, 1] >= y_edges[row]) & \
               (npz_pos[:, 1] <  y_edges[row + 1])
        if col == GRID_SIZE - 1:
            in_x |= (npz_pos[:, 0] == x_edges[col + 1])
        if row == GRID_SIZE - 1:
            in_y |= (npz_pos[:, 1] == y_edges[row + 1])
        idxs = np.where(in_x & in_y)[0]
        cell_indices[(row, col)] = idxs

populated_cells = {k: v for k, v in cell_indices.items() if len(v) > 0}
empty_cells     = {k: v for k, v in cell_indices.items() if len(v) == 0}
total_pop       = sum(len(v) for v in populated_cells.values())

print(f"Populated cells : {len(populated_cells)} / {GRID_SIZE*GRID_SIZE}")
print(f"Empty cells     : {len(empty_cells)}")
print(f"Total samples   : {total_pop}")

# Build eligible pools
print(f"\nBuilding eligible pools (rank + {WINDOW_SIZE} < {N_total})...")
eligible_mask = (rank_in_sorted + WINDOW_SIZE) < N_total
print(f"Eligible samples: {eligible_mask.sum()} / {N_total}")

eligible_cell_indices = {}
for (row, col), idxs in populated_cells.items():
    elig = idxs[eligible_mask[idxs]]
    if len(elig) > 0:
        eligible_cell_indices[(row, col)] = elig

ineligible_cells = [k for k in populated_cells if k not in eligible_cell_indices]
if ineligible_cells:
    print(f"Cells fully ineligible: {ineligible_cells}")
print(f"Eligible cells  : {len(eligible_cell_indices)} / {GRID_SIZE*GRID_SIZE}")

# Distribute N_ANCHORS uniformly
print(f"\nDistributing {N_ANCHORS} anchors uniformly across eligible cells...")
rng              = np.random.default_rng(seed=SEED)
n_eligible_cells = len(eligible_cell_indices)
base_quota       = N_ANCHORS // n_eligible_cells
remainder        = N_ANCHORS  % n_eligible_cells

sorted_cells = sorted(eligible_cell_indices.keys(),
                      key=lambda k: len(eligible_cell_indices[k]), reverse=True)

quotas   = {}
leftover = 0
for i, cell in enumerate(sorted_cells):
    desired      = base_quota + (1 if i < remainder else 0)
    actual       = min(desired, len(eligible_cell_indices[cell]))
    quotas[cell] = actual
    leftover    += desired - actual

print(f"  Base quota/cell : {base_quota},  remainder : {remainder}")
print(f"  Shortfall after first pass : {leftover}")

surplus_cells = sorted(
    [(k, len(eligible_cell_indices[k]) - quotas[k]) for k in quotas
     if len(eligible_cell_indices[k]) > quotas[k]],
    key=lambda x: -x[1]
)
n_surplus = len(surplus_cells)
for i, (cell, surplus) in enumerate(surplus_cells):
    if leftover == 0:
        break
    fair_share    = -(-leftover // (n_surplus - i))
    extra         = min(fair_share, surplus)
    quotas[cell] += extra
    leftover     -= extra

print(f"  Leftover after redistribution : {leftover}")
assert sum(quotas.values()) == N_ANCHORS
assert leftover == 0

print(f"\n{'Cell':>10} | {'Eligible':>10} | {'Quota':>6}")
print("-" * 35)
for cell in sorted(quotas.keys()):
    print(f"  {str(cell):>8} | {len(eligible_cell_indices[cell]):>10} | {quotas[cell]:>6}")
print("-" * 35)
print(f"  {'TOTAL':>8} | {sum(len(v) for v in eligible_cell_indices.values()):>10} | {sum(quotas.values()):>6}")

# Sample anchors
print(f"\nSampling anchors...")
anchor_orig_indices = []
cell_counts = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)

for (row, col), quota in quotas.items():
    idxs   = eligible_cell_indices[(row, col)]
    chosen = rng.choice(idxs, size=quota, replace=False)
    anchor_orig_indices.extend(chosen.tolist())
    cell_counts[row, col] = quota

anchor_orig_indices = np.array(anchor_orig_indices)
N_anchors_valid     = len(anchor_orig_indices)
assert N_anchors_valid == N_ANCHORS
print(f"Anchors sampled : {N_anchors_valid}")
print(f"Cell counts — min: {cell_counts[cell_counts > 0].min()}, max: {cell_counts.max()}")

# Build mobility windows
print(f"\nBuilding mobility windows (next {WINDOW_SIZE} points by timestamp)...")
anchor_ranks          = rank_in_sorted[anchor_orig_indices]
window_sorted_indices = anchor_ranks[:, None] + np.arange(1, WINDOW_SIZE + 1)[None, :]
assert (window_sorted_indices.max(axis=1) < N_total).all()
print("All windows in bounds.")

# Collect all unique samples needed
print("\nCollecting unique samples needed...")
window_orig_indices = sort_order[window_sorted_indices]
all_needed_orig     = np.sort(np.array(
    list(set(anchor_orig_indices.tolist()) |
         set(window_orig_indices.flatten().tolist())),
    dtype=np.int64
))
N_needed   = len(all_needed_orig)
ts_needed  = npz_timestamps[all_needed_orig]
pos_needed = npz_pos[all_needed_orig]
print(f"Unique samples needed: {N_needed}")

orig_to_needed = np.empty(N_total, dtype=np.int64)
orig_to_needed[all_needed_orig] = np.arange(N_needed)

# Build compact metadata arrays
anchor_pos_in_needed     = orig_to_needed[anchor_orig_indices].astype(np.int32)
window_indices_in_needed = orig_to_needed[window_orig_indices].astype(np.int32)
pos_anchors              = npz_pos[anchor_orig_indices]
ts_anchors               = npz_timestamps[anchor_orig_indices]

# Load all needed pickles in parallel → save as sharded .npz
print(f"\nLoading {N_needed} pickle files in parallel ({N_WORKERS} workers)...")
print(f"Shard size: {SHARD_SIZE} samples → {int(np.ceil(N_needed / SHARD_SIZE))} shards")

N_CELLS  = 4
CH_SHAPE = (4, 1, 3276, 3)

def load_one(args):
    local_idx, fname = args
    fpath = os.path.join(input_path, fname)
    with open(fpath, 'rb') as f:
        data = pickle.load(f)
    ch       = np.stack([data['ch_est'][c][0] for c in range(N_CELLS)], axis=0)  # (4,4,1,3276,3)
    cell_ids = np.array(data['cellIds'], dtype=np.uint16)                         # (4,)
    return local_idx, ch, cell_ids

fnames_needed = [ts_to_fname[ts] for ts in ts_needed]
n_shards      = int(np.ceil(N_needed / SHARD_SIZE))

# Initialize with -1 to detect failed loads.
shard_map    = np.full(N_needed, -1, dtype=np.int32)
pos_in_shard = np.full(N_needed, -1, dtype=np.int32)

for shard_idx in range(n_shards):
    start = shard_idx * SHARD_SIZE
    end   = min(start + SHARD_SIZE, N_needed)
    n     = end - start

    ch_buf      = np.empty((n, N_CELLS) + CH_SHAPE, dtype=np.complex64)
    cell_id_buf = np.empty((n, N_CELLS),             dtype=np.uint16)

    tasks = [(i, fnames_needed[start + i]) for i in range(n)]

    missing = []
    with ThreadPoolExecutor(max_workers=N_WORKERS) as executor:
        futures = {executor.submit(load_one, t): t for t in tasks}
        for fut in tqdm(as_completed(futures), total=n,
                        desc=f"Shard {shard_idx+1:03d}/{n_shards}"):
            try:
                local_idx, ch, cell_ids        = fut.result()
                ch_buf[local_idx]              = ch
                cell_id_buf[local_idx]         = cell_ids
                shard_map[start + local_idx]    = shard_idx
                pos_in_shard[start + local_idx] = local_idx
            except Exception as e:
                task = futures[fut]
                missing.append((task[0], task[1], str(e)))

    if missing:
        raise RuntimeError(
            f"Failed to load {len(missing)} samples in shard {shard_idx}: "
            f"{missing[:3]}"
        )

    shard_path = os.path.join(output_dir, f'ch_est_shard_{shard_idx:03d}.npz')
    np.savez(shard_path, ch_est=ch_buf, cell_ids=cell_id_buf)
    print(f"  Saved: {shard_path}  shape={ch_buf.shape}  size={ch_buf.nbytes/1e9:.2f} GB")

    del ch_buf, cell_id_buf

print(f"\nAll {n_shards} shards saved.")

# Save metadata .npz
meta_path = os.path.join(output_dir, 'metadata.npz')
np.savez(
    meta_path,
    pos_anchors              = pos_anchors,
    ts_anchors               = ts_anchors,
    anchor_orig_indices      = anchor_orig_indices,
    anchor_pos_in_needed     = anchor_pos_in_needed,
    window_indices_in_needed = window_indices_in_needed,
    window_orig_indices      = window_orig_indices,
    ts_needed                = ts_needed,
    pos_needed               = pos_needed,
    cell_counts              = cell_counts,
    grid_shape               = np.array([GRID_SIZE, GRID_SIZE]),
    window_size              = np.int32(WINDOW_SIZE),
    n_anchors_target         = np.int32(N_ANCHORS),
    n_shards                 = np.int32(n_shards),
    shard_size               = np.int32(SHARD_SIZE),
    shard_map                = shard_map,
    pos_in_shard             = pos_in_shard,
)
print(f"Metadata saved → {meta_path}")

print(f"""
Done.
  Total raw samples:          {N_total}
  Grid:                       {GRID_SIZE}×{GRID_SIZE}
  Anchors sampled:            {N_anchors_valid}
  Unique samples packed:      {N_needed}
  Shards written:             {n_shards}  (SHARD_SIZE={SHARD_SIZE})
  Output folder:              {output_dir}
  Upload to Triton:           metadata.npz  +  ch_est_shard_*.npz
""")