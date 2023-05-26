from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from mani_skill2.utils.wrappers import RecordEpisode
from stable_baselines3.common.utils import set_random_seed

num_envs = 2 # you can increases this and decrease the n_steps parameter if you have more cores to speed up training
env_id = "LiftCube-v0"
obs_mode = "state"
control_mode = "pd_ee_delta_pose"
reward_mode = "dense"

# define an SB3 style make_env function for evaluation
def make_env(env_id: str, max_episode_steps: int = None, record_dir: str = None):
    def _init() -> gym.Env:
        # NOTE: Import envs here so that they are registered with gym in subprocesses
        import mani_skill2.envs
        env = gym.make(env_id, obs_mode=obs_mode, reward_mode=reward_mode, control_mode=control_mode,)
        # For training, we regard the task as a continuous task with infinite horizon.
        # you can use the ContinuousTaskWrapper here for that
        if max_episode_steps is not None:
            env = ContinuousTaskWrapper(env, max_episode_steps)
        if record_dir is not None:
            env = SuccessInfoWrapper(env)
            env = RecordEpisode(
                env, record_dir, info_on_video=True, render_mode="cameras"
            )
        return env
    return _init

# create one eval environment
eval_env = SubprocVecEnv([make_env(env_id, record_dir="logs/videos") for i in range(1)])
eval_env = VecMonitor(eval_env) # attach this so SB3 can log reward metrics
eval_env.seed(0)
eval_env.reset()

# create num_envs training environments
# we also specify max_episode_steps=100 to speed up training
env = SubprocVecEnv([make_env(env_id, max_episode_steps=100) for i in range(num_envs)])
env = VecMonitor(env)
env.seed(0)
obs = env.reset()


### EvalCallback and Checkpoint Callback
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

# SB3 uses callback functions to create evaluation and checkpoints

# Evaluation: periodically evaluate the agent without noise and save results to the logs folder
eval_callback = EvalCallback(eval_env, best_model_save_path="./logs/",
                             log_path="./logs/", eval_freq=32000,
                             deterministic=True, render=False)

checkpoint_callback = CheckpointCallback(
    save_freq=32000,
    save_path="./logs/",
    name_prefix="rl_model",
    save_replay_buffer=True,
    save_vecnormalize=True,
)


###
from stable_baselines3 import PPO

set_random_seed(0) # set SB3's global seed to 0
rollout_steps = 3200

# create our module
policy_kwargs = dict(net_arch=[256, 256])
model = PPO(
    "MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=1,
    n_steps=rollout_steps // num_envs, batch_size=400,
    n_epochs=15,
    tensorboard_log="./logs",
    gamma=0.85,
    target_kl=0.05
)


# Train with PPO
model.learn(400_000, callback=[checkpoint_callback, eval_callback])
model.save("./logs/latest_model")

# optionally load back the module that was saved
model = model.load("./logs/latest_model")
