from gymnasium.envs.registration import register
from gymnasium.wrappers import TimeLimit

register(
    id="Quadrotor-v1",
    entry_point="env.quad.quad_rotor:QuadRateEnv",  # Adjust as per your directory structure
)

register(
    id="Quadrotor-Still-v1",
    entry_point="env.quad.quad_rotor_still:QuadStillEnv",  # Adjust as per your directory structure
)

def make_bicycle_env():
    from env.bicycle.bicycle_model import KinematicBicycleEnv
    env = KinematicBicycleEnv()
    # env = TimeLimit(env, max_episode_steps=1000)  # Apply TimeLimit wrapper
    return env

register(
    id="Bicycle-v1",
    entry_point=make_bicycle_env,  # Adjust as per your directory structure
)

register(
    id="CustomCart-v1",
    entry_point="env.cartpole.cost_pend:CustomInvertedPendulumEnv",  # Adjust as per your directory structure
)