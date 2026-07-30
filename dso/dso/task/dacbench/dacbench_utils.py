from dacbench.benchmarks import FunctionApproximationBenchmark
from dacbench.envs import FunctionApproximationEnv
from pathlib import Path

import ConfigSpace as CS  # noqa: N817
import ConfigSpace.hyperparameters as CSH

import numpy as np
import pandas as pd

def create_function_approximation_dim1_float_env(seed):
    # loads default config
    bench = FunctionApproximationBenchmark()

    #override with benchmark configs like dimension 1 from paper but make it continuous
    bench.config.omit_instance_type = True
    bench.config.instance_set_path = "dacbench_instances/function_approximation/sigmoid_1D3M_train.csv"
    bench.config.test_set_path = "dacbench_instances/function_approximation/sigmoid_1D3M_test.csv"
    bench.config.discrete = [False]
    cfg_space = CS.ConfigurationSpace()
    dim1 = CSH.UniformFloatHyperparameter(
        name="value_dim_1", lower=0, upper=2
    )
    cfg_space.add(dim1)
    bench.config.config_space = cfg_space
    bench.config.benchmark_info["state_description"] = [
        "Remaining Budget",
        "Shift (dimension 1)",
        "Slope (dimension 1)",
        "Action",
    ]
    bench.config.observation_space_args = [
        np.array([-np.inf for _ in range(4)]),
        np.array([np.inf for _ in range(4)]),
    ]
    bench.config.seed = seed
    bench.read_instance_set()
    bench.read_instance_set(test=True)
    return FunctionApproximationEnv(bench.config)
