# CSI Feature and Distance Robustness for Indoor Positioning

Code accompanying the paper **“On the Robustness of CSI Features and Distance Metrics for Indoor Positioning: Evidence from CAEZ-5G and DICHASUS.”**

The repository contains the CAEZ-5G and DICHASUS experiments used to compare CSI feature representations and distance metrics for indoor positioning.

## Datasets

The experiments use the publicly available CAEZ-5G and DICHASUS datasets. The datasets are not included in this repository and can be obtained from their official sources.

- **CAEZ-5G:** [CAEZ: CSI Acquisition at ETH Zurich](https://iip.ethz.ch/datasets/caez.html). We use the CAEZ-5G indoor measurements. Please cite:

  > R. Wiesmayr, F. Zumegen, S. Taner, C. Dick, and C. Studer, “CSI-Based User Positioning, Channel Charting, and Device Classification with an NVIDIA 5G Testbed,” *Asilomar Conference on Signals, Systems, and Computers*, 2025.

- **DICHASUS:** [dichasus-cf0x](https://dichasus.inue.uni-stuttgart.de/datasets/data/dichasus-cf0x/). We use the `dichasus-cf02` to `dichasus-cf07` measurements. Please cite:

  > F. Euchner and M. Gauger, “CSI Dataset dichasus-cf0x: Distributed Antenna Setup in Industrial Environment, Day 1,” *DaRUS*, 2022. DOI: [10.18419/DARUS-2854](https://doi.org/10.18419/DARUS-2854).


## Repository layout

```text
code/
├── compare_results.py
├── caez5g/
│   ├── prepare_data.py
│   ├── anchor_avd.py
│   ├── anchor_rasd.py
│   ├── anchor_l1.py
│   ├── avg6_avd.py
│   ├── avg6_rasd.py
│   ├── avg6_l1.py
│   ├── avg50_avd.py
│   ├── avg50_rasd.py
│   ├── avg50_l1.py
│   ├── magnitude.py
│   └── covariance.py
└── dichasus/
    ├── prepare_data.py
    ├── anchor_avd.py
    ├── anchor_rasd.py
    ├── anchor_l1.py
    ├── avg6_avd.py
    ├── avg6_rasd.py
    ├── avg6_l1.py
    ├── avg50_avd.py
    ├── avg50_rasd.py
    ├── avg50_l1.py
    ├── magnitude.py
    └── covariance.py
```

The two dataset folders use matching filenames for corresponding experiments.
`compare_results.py` produces the comparative NPR and ranking results reported across the two datasets.

## CSI representations

The vector experiments use four feature representations:

- spatial CSI vector at each subcarrier (`antenna_per_freq`),
- spatial CSI vector at each delay tap (`antenna_per_delay`),
- delay-domain CSI vector at each antenna (`delay_per_antenna`),
- delay-domain CSI vector at each beam (`delay_per_beam`).

Magnitude-only experiments use the same feature orientations. For the delay-per-beam magnitude feature, the spatial transform is applied to the complex CIR before taking the magnitude.

The covariance experiments use spatial covariance matrices formed from the retained CIR taps. Raw and trace-normalised covariance matrices are evaluated with Euclidean, Log-Euclidean, and Bures-Wasserstein distances. The Log-Euclidean calculation uses the eigenvalue floor

```text
max(1e-6 * trace(R) / M, 1e-12)
```

where `M` is the covariance dimension. Euclidean and Bures-Wasserstein distances are computed in memory-controlled blocks.

## Performance measures

The scripts report:

- Trustworthiness (TW),
- Continuity (CT),
- Neighbourhood Preservation Ratio (NPR),
- Kruskal stress (KS).

The neighbourhood size is `J = 100` and both datasets use 4,000 anchor samples.

## Experimental settings

### DICHASUS

- DICHASUS rev2 STO normalisation followed by file-specific STO/CPO calibration.
- Array A channel layout: `[6, 2, 16, 18; 28, 5, 10, 14]`.
- 20 × 20 spatial stratification with seed 42.
- CIR interval `509:527` (18 taps).
- Average-over-6: anchor + 5 consecutive samples.
- Average-over-50: anchor + 49 consecutive samples.

Expected files:

```text
data/dichasus/
├── dichasus-cf02.tfrecords
├── ...
├── dichasus-cf07.tfrecords
├── reftx-offsets-dichasus-cf02.json
├── ...
└── reftx-offsets-dichasus-cf07.json
```

### CAEZ-5G

- ORU 0.
- Every 12th subcarrier is retained for CFR-based vector features (273 subcarriers).
- CIR is computed from the full 3276-subcarrier CFR before tap selection.
- CIR interval `1602:1670` (68 taps).
- Three DMRS estimates per CSI sample.
- Average-over-6: anchor + 5 consecutive samples.
- Average-over-50: anchor + 49 consecutive samples.

The preparation script stores 99 successors for each selected anchor. The experiment scripts use the first 5 or 49 successors according to the selected averaging window.

Expected files before preprocessing:

```text
data/caez5g/
├── raw/
└── reference/
    └── caez_5g_indoor_full.npz
```

Prepare the CAEZ-5G mobility data from the repository root:

```bash
python code/caez5g/prepare_data.py
```

Prepared files are written to `data/caez5g/mobility_dataset/`.

## Running experiments

Run scripts from the repository root. For example:

```bash
python code/dichasus/anchor_avd.py
python code/caez5g/avg50_avd.py
```

The datasets and generated metric files are not included in the repository.

## Comparative results

The final comparisons across CAEZ-5G and DICHASUS can be obtained from the experiment metric files using:

```bash
python code/compare_results.py
```

## Dependencies

```bash
pip install -r requirements.txt
```

## Citation

Citation metadata are provided in `CITATION.cff`.
