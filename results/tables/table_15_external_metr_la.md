**Table XV. Probabilistic forecasting on METR-LA, 207 loop detectors on the Los Angeles freeway network, with the protocol of Section V. The last column is the share of continuous ranked probability score lost between zero and 40 percent added observation loss, which is the robustness claim of this paper tested on a corpus the authors did not generate.**

| Model                      | MAE             | MAPE (%)         | CRPS            | PICP95          | ECE             |   CRPS at 40% loss |   Score lost (%) |
|:---------------------------|:----------------|:-----------------|:----------------|:----------------|:----------------|-------------------:|-----------------:|
| DyHGT-MARL mixture density | 4.769 +/- 0.059 | 14.652 +/- 0.236 | 3.327 +/- 0.057 | 0.956 +/- 0.010 | 0.020 +/- 0.003 |              3.55  |              6.7 |
| Temporal Transformer       | 4.333 +/- 0.071 | 12.740 +/- 0.210 | 4.006 +/- 0.060 | 0.930 +/- 0.001 | 0.119 +/- 0.002 |              4.234 |              5.7 |
| DCRNN, reduced             | 4.440 +/- 0.004 | 12.988 +/- 0.419 | 4.027 +/- 0.024 | 0.927 +/- 0.001 | 0.111 +/- 0.002 |              4.238 |              5.2 |
| GRU                        | 4.445 +/- 0.005 | 13.158 +/- 0.093 | 4.053 +/- 0.042 | 0.927 +/- 0.000 | 0.114 +/- 0.000 |              4.287 |              5.8 |
| Ridge autoregression       | 4.945           | 14.372           | 4.241           | 0.927           | 0.109           |              4.471 |              5.4 |
| Historical average         | 4.695           | 14.945           | 4.346           | 0.936           | 0.118           |              4.346 |              0   |
| STGCN, reduced             | 5.778 +/- 0.014 | 16.575 +/- 0.376 | 4.784 +/- 0.026 | 0.919 +/- 0.001 | 0.082 +/- 0.002 |              4.908 |              2.6 |
