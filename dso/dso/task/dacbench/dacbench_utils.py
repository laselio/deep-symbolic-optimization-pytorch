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

def create_theory_env(instance_set : str, test : bool = False):
    from dacbench.benchmarks.theory_benchmark import TheoryBenchmark

    config = {

    # continuous action space
    "discrete_action": False,
    "min_action": 1.0,
    "max_action": 50.0,
    # simple reward
    "reward_choice": "imp",
    "problem": "LeadingOne",
    "instance_set_path": instance_set,
    }

    benchmark = TheoryBenchmark(config=config)
    env = benchmark.get_environment(test_env=test)
    print(env.action_space)
    print(env.config["reward_range"])
    print(env.config["cutoff"])
    return env