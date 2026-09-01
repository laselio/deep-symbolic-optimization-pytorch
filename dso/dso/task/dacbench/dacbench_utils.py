from dacbench.benchmarks import FunctionApproximationBenchmark
from dacbench.envs import FunctionApproximationEnv

import ConfigSpace as CS  # noqa: N817
import ConfigSpace.hyperparameters as CSH

import numpy as np
import gymnasium as gym

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

def create_theory_env(instance_set : str, instance_size: float, test : bool = False):
    from dacbench.benchmarks.theory_benchmark import TheoryBenchmark

    config = {

    # continuous action space
    "discrete_action": False,
    "min_action": 1.0,
    "max_action": instance_size,
    # simple reward
    "reward_choice": "minus_evals",
    "problem": "LeadingOne",
    "instance_set_path": instance_set,
    }

    benchmark = TheoryBenchmark(config=config)
    env = benchmark.get_environment(test_env=test)
    print(env.action_space)
    print(env.config["reward_range"])
    print(env.config["cutoff"])
    return env

# per default instance_set and test_set are the same.
def create_toy_sgd_env(instance_set: str = "toysgd_default.csv" , test_set: str = "toysgd_default.csv", fixed_momentum: float = -5 ):
    from dacbench.benchmarks.toysgd_benchmark import ToySGDBenchmark
    from dacbench.abstract_benchmark import objdict

    config = objdict({
        "instance_set_path": instance_set,
        "test_set_path": test_set
    })

    benchmark = ToySGDBenchmark(config=config)
    env = benchmark.get_environment()
    env = ScalarActionWrapper(env=env, fixed_momentum=fixed_momentum)
    env = ToySGDObsFixWrapper(env=env)
    print(env.action_space)
    print(env.config["reward_range"])
    print(env.config["cutoff"])

    return env

class ScalarActionWrapper(gym.ActionWrapper):
    def __init__(self, env, fixed_momentum=-5.0):
        super().__init__(env)
        self.config = env.config
        self.fixed_momentum = fixed_momentum
        self.env = env
        self.initial_seed = env.initial_seed
        self.instance = env.instance

        self.action_space = gym.spaces.Box(
            low=env.action_space.low[:1],
            high=env.action_space.high[:1],
            dtype=np.float32,
        )

    def action(self, action):
        return np.array(
            [action[0], self.fixed_momentum],
            dtype=np.float32
        )
    def get_inst_id(self):
        return self.env.get_inst_id()


 # scalar log-learning-rate
# make it return the learning rate and the fixed momentum
# Observation wrapper der alles in float 32 casted
# __get attribute does not work, cast all relevant attributes through debugging

class ToySGDObsFixWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.config =env.config
        self.env = env
        self.initial_seed = env.initial_seed
        self.instance = env.instance

        #add env attributes
    def observation(self, obs):
        return {
            "remaining_budget": np.array(
                [obs["remaining_budget"]],
                dtype=np.float32
            ),
            "gradient": np.asarray(
                obs["gradient"],
                dtype=np.float32
            ),
            "learning_rate": np.array(
                [obs["learning_rate"]],
                dtype=np.float32
            ),
            "momentum": np.array(
                [obs["momentum"]],
                dtype=np.float32
            ),
        }

    def get_inst_id(self):
        return self.env.get_inst_id()

