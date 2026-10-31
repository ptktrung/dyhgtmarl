**Table X. Computational cost of each component. Training times were measured on a single CPU core and are not a hardware benchmark of a deployed system.**

| Component                                       | Cost measure              |   Nominal model value |   Measured on this machine | Note                                     |
|:------------------------------------------------|:--------------------------|----------------------:|---------------------------:|:-----------------------------------------|
| policy: centralized_constrained                 | milliseconds per decision |                 1.051 |                     0.0087 | nominal values are simulator assumptions |
| policy: centralized_utility_max                 | milliseconds per decision |                 1.099 |                     0.0094 | nominal values are simulator assumptions |
| policy: decentralized_constrained               | milliseconds per decision |                 0.806 |                     0.0076 | nominal values are simulator assumptions |
| policy: decentralized_constrained_no_abstention | milliseconds per decision |                 1.076 |                     0.0083 | nominal values are simulator assumptions |
| policy: decentralized_utility_max               | milliseconds per decision |                 0.954 |                     0.0076 | nominal values are simulator assumptions |
| policy: rule_based_legacy                       | milliseconds per decision |                 0.4   |                     0.0145 | nominal values are simulator assumptions |
| fault model: dyhgt_marl                         | seconds to fit one seed   |                       |                   131.1    | single CPU core, no GPU                  |
| fault model: hist_gradient_boosting             | seconds to fit one seed   |                       |                     0.7    | single CPU core, no GPU                  |
| fault model: logistic_regression                | seconds to fit one seed   |                       |                     0.1    | single CPU core, no GPU                  |
| fault model: mlp                                | seconds to fit one seed   |                       |                     2.4    | single CPU core, no GPU                  |
| fault model: random_forest                      | seconds to fit one seed   |                       |                    17.9    | single CPU core, no GPU                  |
| fault model: random_forest_graph_features       | seconds to fit one seed   |                       |                    39.3    | single CPU core, no GPU                  |
| forecast model: dcrnn                           | seconds to fit one seed   |                       |                    23.1    | single CPU core, no GPU                  |
| forecast model: dyhgt                           | seconds to fit one seed   |                       |                   273      | single CPU core, no GPU                  |
| forecast model: gru                             | seconds to fit one seed   |                       |                    30      | single CPU core, no GPU                  |
| forecast model: ha                              | seconds to fit one seed   |                       |                     0      | single CPU core, no GPU                  |
| forecast model: ridge                           | seconds to fit one seed   |                       |                     0      | single CPU core, no GPU                  |
| forecast model: stgcn                           | seconds to fit one seed   |                       |                    29      | single CPU core, no GPU                  |
| forecast model: transformer                     | seconds to fit one seed   |                       |                   107.8    | single CPU core, no GPU                  |
