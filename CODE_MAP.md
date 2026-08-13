# Code map

| File | Experiment |
|---|---|
| `prepare_data.py` | Dataset preparation and anchor selection |
| `anchor_avd.py` | Anchor-only AVD |
| `anchor_rasd.py` | Anchor-only RASD |
| `anchor_l1.py` | Anchor-only L1,1 and L1,2 |
| `avg6_avd.py` | Average-over-6 AVD |
| `avg6_rasd.py` | Average-over-6 RASD |
| `avg6_l1.py` | Average-over-6 L1,1 and L1,2 |
| `avg50_avd.py` | Average-over-50 AVD |
| `avg50_rasd.py` | Average-over-50 RASD |
| `avg50_l1.py` | Average-over-50 L1,1 and L1,2 |
| `magnitude.py` | Magnitude-only CSI experiments |
| `covariance.py` | Spatial covariance experiments |

The vector experiments use `antenna_per_freq`, `antenna_per_delay`, `delay_per_antenna`, and `delay_per_beam` in both datasets.

The covariance scripts form spatial covariance matrices from the retained CIR taps and evaluate Euclidean, Log-Euclidean, and Bures-Wasserstein distances for raw and trace-normalised covariance matrices.

Performance measures are TW, CT, NPR, and KS.
