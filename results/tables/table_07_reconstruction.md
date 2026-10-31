**Table VII. Reconstruction of masked travel time observations on the test block. Methods that use future observations are smoothers and are marked as not causal.**

| Method                           |   MAE minutes |   RMSE minutes |   Masked cells scored | Causal, uses no future observation   |
|:---------------------------------|--------------:|---------------:|----------------------:|:-------------------------------------|
| linear_interpolation             |         1.288 |          1.768 |                  3003 | False                                |
| global_mean                      |        10.482 |         12.179 |                  3003 | True                                 |
| last_observation_carried_forward |         2.023 |          2.718 |                  3003 | True                                 |
| knn_imputer                      |         2.272 |          2.93  |                  3003 | False                                |
| dyhgt_marl[topology_closure]     |         2.733 |          3.594 |                   596 | True                                 |
| hour_of_week_climatology         |         2.765 |          3.564 |                  3003 | True                                 |
| dyhgt_marl[MNAR_signal]          |         2.822 |          3.755 |                   130 | True                                 |
| dyhgt_marl                       |         2.88  |          3.68  |                  2986 | True                                 |
| graph_diffusion                  |         3.092 |          3.954 |                  3003 | False                                |
| dyhgt_marl[gateway_outage]       |         4.79  |          5.464 |                    60 | True                                 |
