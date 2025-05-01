from __future__ import annotations

from hpobench.benchmarks.nas.nasbench_201 import NASBench_201Data

# from hpobench.benchmarks.nas.nasbench_101 import NASBench_101DataManager

# Load nasbench201 data
datasets = ["cifar10-valid", "cifar100", "ImageNet16-120"]
for dataset in datasets:
    data_manager = NASBench_201Data(dataset=dataset)
    data_manager.load()

# # Load nasbench101 data
# datasets = ["cifar10-valid", "cifar100", "ImageNet16-120"]
# for dataset in datasets:
#     data_manager = NASBench_101DataManager(dataset=dataset)
#     data_manager.load()