**Table XIV. The primary benchmark against the revised stress regime, in which faults show as neighbourhood disagreement rather than telemetry clamping, the node series are decorrelated, and latent validity and false accept cost differ by zone. Topology, splits and evaluation code are unchanged.**

| Quantity                                          |   Primary benchmark |   Stress regime |
|:--------------------------------------------------|--------------------:|----------------:|
| Mean pairwise node correlation, harmonized series |              0.9345 |          0.5501 |
| Latent validity, core zone                        |              0.9803 |          0.975  |
| Latent validity, peripheral zone                  |              0.9793 |          0.8719 |
| Latent validity gap between zones                 |              0.001  |          0.1031 |
| Fault F1, multilayer perceptron                   |              0.988  |          0.459  |
| Fault F1, gradient boosting                       |              0.982  |          0.394  |
| Fault F1, graph feature forest                    |              0.889  |          0.302  |
| Fault F1, proposed encoder                        |              0.658  |          0.241  |
| Forecast CRPS, proposed encoder                   |              1.65   |          2.885  |
| Forecast CRPS, climatological baseline            |              1.705  |          3.118  |
| Forecast CRPS, spatio temporal baseline           |              2.417  |          3.4    |
| Forecast CRPS, linear baseline                    |              2.802  |          3.369  |
| Unconstrained policy, equal opportunity gap       |              0.0006 |          0.0298 |
| Constrained policy, equal opportunity gap         |              0.0013 |          0.0136 |
| Constrained policy, abstention rate               |              0.1352 |          0.374  |
| Constrained policy, revenue recovered percent     |             88.96   |         75.12   |
| Coupled fault head, revenue recovered percent     |             87.73   |         79.45   |
| Coupled fault head, decision error rate           |              0.0258 |          0.0377 |
| Coupled fault head, equal opportunity gap         |              0.0041 |          0.0273 |
