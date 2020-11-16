#train_utils.py
import numpy as np
import torch
import tqdm.notebook as tqdm

import dream_template.config as cfg
from dream_template import grid, city, policy, relabel, rl, utils


def run_episode(env, policy, experience_observers=None, test=False):
  """Runs a single episode on the environment following the policy.

  Args:
    env (gym.Environment): environment to run on.
    policy (Policy): policy to follow.
    experience_observers (list[Callable] | None): each observer is called with
      with each experience at each timestep.

  Returns:
    episode (list[Experience]): experiences from the episode.
    renders (list[object | None]): renderings of the episode, only rendered if
      test=True. Otherwise, returns list of Nones.
  """
  # Optimization: rendering takes a lot of time.
  def maybe_render(env, action, reward, timestep):
    if test:
      render = env.render()
      render.write_text("Action: {}".format(str(action)))
      render.write_text("Reward: {}".format(reward))
      render.write_text("Timestep: {}".format(timestep))
      return render
    return None

  if experience_observers is None:
    experience_observers = []

  episode = []
  state = env.reset()
  timestep = 0
  renders = [maybe_render(env, None, 0, timestep)]
  hidden_state = None
  while True:
    action, next_hidden_state = policy.act(
        state, hidden_state, test=test)
    next_state, reward, done, info = env.step(action)
    timestep += 1
    renders.append(
        maybe_render(env, grid.Action(action), reward, timestep))
    experience = rl.Experience(
        state, action, reward, next_state, done, info, hidden_state,
        next_hidden_state)
    episode.append(experience)
    for observer in experience_observers:
      observer(experience)

    state = next_state
    hidden_state = next_hidden_state
    if done:
      return episode, renders


def get_env_class(environment_type):
  """Returns the environment class specified by the type.

  Args:
    environment_type (str): a valid environment type.

  Returns:
    environment_class (type): type specified.
  """
  if environment_type == "vanilla":
    return city.CityGridEnv
  elif environment_type == "distraction":
    return city.DistractionGridEnv
  elif environment_type == "map":
    return city.MapGridEnv
  elif environment_type == "cooking":
    return cooking.CookingGridEnv
  elif environment_type == "miniworld_sign":
    # Dependencies on OpenGL, so only load if absolutely necessary
    from envs.miniworld import sign
    return sign.MiniWorldSign
  else:
    raise ValueError(
        "Unsupported environment type: {}".format(environment_type))


def get_instruction_agent(instruction_config, instruction_env):
  if instruction_config.get("type") == "learned":
    return DQNAgent.from_config(instruction_config, instruction_env)
  else:
    raise ValueError(
        "Invalid instruction agent: {}".format(instruction_config.get("type")))


def get_exploration_agent(exploration_config, exploration_env):
  if exploration_config.get("type") == "learned":
    return DQNAgent.from_config(exploration_config, exploration_env)
  elif exploration_config.get("type") == "random":
    return policy.RandomPolicy(exploration_env.action_space)
  elif exploration_config.get("type") == "none":
    return policy.ConstantActionPolicy(grid.Action.end_episode)
  else:
    raise ValueError("Invalid exploration agent: {}".format(
      exploration_config.get("type")))


def log_episode(exploration_episode, exploration_rewards, distances, path):
  with open(path, "w+") as f:
    f.write("Env ID: {}\n".format(exploration_episode[0].state.env_id))
    for t, (exp, exploration_reward, distance) in enumerate(
        zip(exploration_episode, exploration_rewards, distances)):
      f.write("=" * 80 + "\n")
      f.write("Timestep: {}\n".format(t))
      f.write("State: {}\n".format(exp.state.observation))
      f.write("Action: {}\n".format(grid.Action(exp.action).name))
      f.write("Reward: {}\n".format(exploration_reward))
      f.write("Distance: {}\n".format(distance))
      f.write("Next state: {}\n".format(exp.next_state.observation))
      f.write("=" * 80 + "\n")
      f.write("\n"
