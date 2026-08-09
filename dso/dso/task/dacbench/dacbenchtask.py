from gymnasium import spaces
import numpy as np

from dacbench.logger import Logger
from dacbench.wrappers import PerformanceTrackingWrapper
from pathlib import Path

from dso.program import Program, from_str_tokens
from dso.library import Library, DiscreteAction, MultiDiscreteAction
from dso.functions import create_tokens, create_state_checkers
from sympy import pprint
from dso.task import HierarchicalTask

from dso.task.dacbench import dacbench_utils as util
from gymnasium.spaces import Dict
from gymnasium.wrappers import FlattenObservation

REWARD_SEED_SHIFT = int(1e6)  # Reserve the first million seeds for evaluation


class Action:
    """
    This serves as an interface between the actions in DSO and gymnasium. Depending
    on the action space, the corresponding type of symbolic action is computed and returned.

    Parameters
    ----------
    space : gymnasium.Space
        Action space for the control problem.
        Supported option: Box, Discrete, MultiDiscrete.
    """

    def __init__(self, space):
        self.is_discrete = isinstance(space, spaces.Discrete)
        self.is_multi_discrete = isinstance(space, spaces.MultiDiscrete)
        self.shape = (1,) if self.is_discrete else space.shape
        self.symbolic_actions = {}
        self.model = None

        self.n_actions = self.shape[0]
        self.action_dim = None

        # set upper and lower bounds for action value
        self.low = None
        self.high = None
        if isinstance(space, spaces.Box):
            self.low = space.low
            self.high = space.high

    def set_action_spec(self, action_spec, algorithm, anchor, env_name):
        """Set anchor model, previously learned symbolic actions, and
        action_dim of the action being learned according to action_spec."""

        # TODO: Load the anchor model according to benchmark if action space is multi-dimensional
        self.model = None

        for i, spec in enumerate(action_spec):
            # Action taken from anchor policy
            if spec == "anchor":
                continue

            # Action dimension being learned
            if spec is None:
                self.action_dim = i

            # Pre-specified symbolic policy
            elif isinstance(spec, list) or isinstance(spec, str):
                p = from_str_tokens(spec, skip_cache=True)
                self.symbolic_actions[i] = p
            else:
                assert (
                    False
                ), "Action specifications must be None, a \
                                str/list of tokens, or 'anchor'."

    def __call__(self, p, obs):
        """Depending on action_spec, returns action from Program p, anchor
        model, and previously learned symbolic actions according to obs."""
        if self.is_multi_discrete:
            action = self._get_action_from_program(p, obs)
        else:
            # Compute anchor actions
            if self.model is not None:
                action, _ = self.model.predict(obs)
            else:
                action = np.zeros(self.shape, dtype=np.float32)

            # Replace fixed symbolic actions
            for j, fixed_p in self.symbolic_actions.items():
                action[j] = self._get_action_from_program(fixed_p, obs)

            # Replace symbolic action with current program
            if self.action_dim is not None:
                action[self.action_dim] = self._get_action_from_program(p, obs)

        return self._to_gym_action(action)

    def _get_action_from_program(self, p, obs):
        """Helper function to get action from Program p according to obs,
        since Program.execute() requires 2D arrays, but we only want 1D."""
        action = p.execute(np.array([obs]))[0]
        return np.asarray(action)

    def _to_gym_action(self, action):
        """Returns the correct object expected by gymnasium based on action space."""
        # Replace Nans with zero and clip infinites
        action[np.isnan(action)] = 0.0
        if self.is_discrete:
            return int(action[0])
        elif self.is_multi_discrete:
            return action.astype(int)
        return np.clip(action, self.low, self.high)


def create_decision_tree_tokens(
        n_obs, obs_threshold_sets, action_space, ref_action=None
):
    """
    Create a list of tokens for learning decision trees. The action space
    must be either Discrete or MultiDiscrete. The returning list contains
    only action tokens and StateCheckers.

    Parameters
    ----------
    n_obs : int
        Number of observations (or state variables).

    obs_threshold_sets : list or list of lists
        If it is a list of constants [t1, t2, ..., tn], tj's are thresholds
        for all state variables. If it is a list of lists of constants
        [[t11, t12, t1n], [t21, t22, ..., t2m], ...], the i-th list contains
        thresholds for state variable xi. The sizes of the threshold lists
        can be different for different state variables.

    action_space : gymnasium.Space
        Action space for the control problem.

    Returns
    -------
    tokens : list of Tokens
        a list of Tokens in the library.
    """
    tokens = []

    if isinstance(action_space, spaces.Discrete):
        for a in range(action_space.n):
            tokens.append(DiscreteAction(a))
    else:
        assert isinstance(action_space, spaces.MultiDiscrete)
        if ref_action is None:
            ref_action = np.zeros(len(action_space.nvec), dtype=np.int32)
        tokens.append(MultiDiscreteAction(ref_action))
        for d in range(len(action_space.nvec)):
            for a in range(action_space.nvec[d]):
                tokens.append(MultiDiscreteAction(a, d))

    state_checker_tokens = create_state_checkers(n_obs, obs_threshold_sets)
    tokens.extend(state_checker_tokens)

    return tokens


class DACBenchTask(HierarchicalTask):
    """
    Class for the control task. Discrete objects are expressions, which are
    evaluated by directly using them as control policies in a reinforcement
    learning environment.
    """

    def __init__(
            self,
            function_set,
            env_name,
            action_spec,
            experiment_name,
            logging_dir: str,
            instance_set: str,
            test_set: str,
            max_action: float,
            algorithm=None,
            anchor=None,
            n_episodes_train=5,
            n_episodes_test=1000,
            protected=False,
            reward_scale=True,
            decision_tree_threshold_set=None,
            ref_action=None,
    ):
        """
        Parameters
        ----------

        function_set : list
            List of allowable functions.

        env_name : str
            Name of dacbench environment, e.g. "FunctionApproximation-v0".


        action_spec : list
            List of action specifications: None, "anchor", or a list of tokens.

        experiment_name : str
            Name of the experiment, used for the log file

        logging_dir : str
            Path to logging directory.

        instance_set: str
            Only used in TheoryBenchmark for now, File name of the instance_set used for training

        test_set: str
            Only used in TheoryBenchmark for now, File name of the instance_set used for testing

        max_action: float
            Used only in case of TheoryBenchmark for now, controls upper bound of action interval
        algorithm : str or None
            Name of algorithm corresponding to anchor path, or None to use
            default anchor for given environment.

        anchor : str or None
            Path to anchor model, or None to use default anchor for given
            environment.

        n_episodes_train : int
            Number of episodes to run during training.

        n_episodes_test : int
            Number of episodes to run during testing.

        protected : bool
            Whether or not to use protected operators.

        decision_tree_threshold_set : list
            A set of constants {tj} for constructing nodes (xi < tj) in decision
            trees.
        """

        super(HierarchicalTask).__init__()

        # Set member variables used by member functions
        self.n_episodes_train = n_episodes_train
        self.n_episodes_test = n_episodes_test
        self.stochastic = True
        self.reward_scale = reward_scale

        # Create the environment based on dacbench benchmark, add Wrappers for box space conversion and episode statistics
        self.eval_env = None
        self.has_test_set = True
        if env_name == "FunctionApproximationBenchmark":
            self.env = util.create_function_approximation_dim1_float_env(seed=0)
            # setting n_episodes_test to cover the full test instance_sets
            self.n_episodes_test = len(self.env.test_set)
        elif env_name == "TheoryBenchmark":
            self.env = util.create_theory_env(instance_set=instance_set, instance_size=max_action)
            self.eval_env = util.create_theory_env(instance_set=test_set, instance_size=max_action, test=True)
            self.eval_env.instance_updates = "round_robin"
            self.has_test_set = False
            self.n_episodes_test = 400
        elif env_name == "ToySGDBenchmark":
            self.env = util.create_toy_sgd_env(instance_set=instance_set, test_set=test_set)
            self.n_episodes_test = 400
        else:
            print("Invalid Benchmark Name")
            return -1
        self.env_name = env_name

        # Determine reward scaling

        self.r_min, self.r_max = self.env.config["reward_range"]

        self.env.instance_updates = "round_robin"

        pprint("Instance: {}".format(self.env.instance))
        print(self.env.action_space)
        # hooking env up with logging
        self.logger = Logger(experiment_name=experiment_name, output_path=Path(logging_dir))
        self.performance_logger = self.logger.add_module(PerformanceTrackingWrapper)
        self.logger.set_env(self.env)

        self.env = PerformanceTrackingWrapper(self.env, logger=self.performance_logger)
        print(self.env.observation_space)
        if isinstance(self.env.observation_space, Dict):
            print("Flattening")
            self.env = FlattenObservation(self.env)
            if self.eval_env is not None:
                self.eval_env = FlattenObservation(self.eval_env)
        print(self.env.observation_space)

        print(type(self.env.observation_space))

        print(self.env.observation_space.shape)

        print(self.env.action_space)
        print(type(self.env.action_space))

        self.action = Action(self.env.action_space)

        print(f"Minimum Reward: {self.r_min}, Maximum Reward:{self.r_max}")

        # Set the library based on the action space shape (do this now in case there are symbolic actions)
        n_input_var = self.env.observation_space.shape[0]

        if self.action.is_discrete or self.action.is_multi_discrete:
            print(
                "WARNING: The provided function_set will be ignored because "
                "action space of {} is {}.".format(env_name, self.env.action_space)
            )
            tokens = create_decision_tree_tokens(
                n_input_var,
                decision_tree_threshold_set,
                self.env.action_space,
                ref_action,
            )
        else:
            tokens = create_tokens(
                n_input_var, function_set, protected, decision_tree_threshold_set
            )
        self.library = Library(tokens)
        Program.library = self.library

        # Configuration assertions (taken from original dso)
        assert (
                len(self.env.observation_space.shape) == 1
        ), "Only support vector observation spaces."
        n_actions = self.action.n_actions
        assert n_actions == len(
            action_spec
        ), "Received spec for {} action \
               dimensions; expected {}.".format(
            len(action_spec), n_actions
        )
        if not self.action.is_multi_discrete:
            assert (
                    len([v for v in action_spec if v is None]) <= 1
            ), "No more than 1 action_spec element can be None."
        assert int(algorithm is None) + int(anchor is None) in [
            0,
            2,
        ], "Either none or both of (algorithm, anchor) must be None."

        # Generate symbolic policies and determine action dimension
        self.action.set_action_spec(action_spec, algorithm, anchor, env_name)

        # Define name based on environment and learned action dimension
        self.name = env_name
        if self.action.action_dim is not None:
            self.name += "_a{}".format(self.action.action_dim)

    def run_episodes(self, p, n_episodes, evaluate):
        """Runs n_episodes episodes and returns each episodic reward."""

        # Run the episodes and use the gymnasium RecordEpisodeStatistics info
        # when available, to avoid manual accumulation.
        r_episodes = np.zeros(
            n_episodes, dtype=np.float64
        )  # Episodic rewards for each episode

        if evaluate and self.eval_env is not None:
            print("Evaluating on separate Evaluation Env")
            self.env = self.eval_env

        for i in range(n_episodes):

            # Always use random seeds
            obs, _ = self.env.reset()

            done = False
            episode_reward = 0.0
            info = {}
            while not done:
                action = self.action(p, obs)
                obs, r, terminated, truncated, info = self.env.step(action)
                if evaluate:
                    self.logger.next_step()
                done = terminated or truncated
                episode_reward += r

            if "episode" in info and "r" in info["episode"]:
                r_episodes[i] = info["episode"]["r"]
            else:
                r_episodes[i] = episode_reward

            if evaluate:
                self.logger.next_episode()

        return r_episodes

    def reward_function(self, p, optimizing=False):

        # Run the episodes
        r_episodes = self.run_episodes(p, self.n_episodes_train, evaluate=False)

        # print("program:", p)
        # print("r_episodes:", r_episodes)

        # Return the mean
        r_avg = np.mean(r_episodes)

        # Scale rewards to [0, 1] if reward_scale == true
        if self.reward_scale and self.r_min is not None:
            r_avg = (r_avg - self.r_min) / (self.r_max - self.r_min)  # type: ignore

        return r_avg

    def evaluate(self, p):
        if self.has_test_set:
            self.env.unwrapped.use_test_set()
            self.env.unwrapped.instance_index = -1

        # Run the episodes
        r_episodes = self.run_episodes(p, self.n_episodes_test, evaluate=True)

        if self.has_test_set:
            self.env.unwrapped.use_training_set()
        # Compute eval statistics
        r_avg_test = np.mean(r_episodes)

        info = {
            "r_avg_test": r_avg_test
        }
        return info