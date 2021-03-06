from collections import defaultdict
from typing import List

import numpy as np
from config import ENV_PATH, TARGET_SCORE, EPOCHS, BATCH_SIZE
from ppo.buffer import TrajectoryProcessor, TrajectoryPrep
from utils import ReacherEnv
from ppo.agent import PPO
import sys
import matplotlib.pyplot as plt
from pathlib import Path

import os

os.makedirs("./plots", exist_ok=True)


def train_agent(agent:PPO, env:ReacherEnv, target_score:float=None):
    episodes_so_far = 0
    best_score = -np.inf
    train_info = defaultdict(lambda: [])
    print(f"Episode {episodes_so_far}: last 100 episodes score mean = {0:2f}", end="")
    last_100_episodes_mean_score = 0
    while last_100_episodes_mean_score < target_score:
        episodes_so_far += 1
        episode_info = train_for_an_episode(agent, env)
        train_info["episode_scores"] += [episode_info["average_score"]]
        last_100_episodes_window = max(episodes_so_far - 100, 0)
        last_100_episodes_mean_score = np.mean(train_info["episode_scores"][last_100_episodes_window:])
        train_info["last_100_episodes_mean_score"] += [last_100_episodes_mean_score]
        print(f"\rEpisode {episodes_so_far}: last 100 episodes score mean = "
              f"{last_100_episodes_mean_score:2f}", end="")
        if episode_info["average_score"] > best_score:
            agent.save()
            best_score = episode_info["average_score"]
        plot_score(train_info["last_100_episodes_mean_score"], train_info["episode_scores"])

    print(f"\rEpisode {episodes_so_far}: last 100 episodes score mean = "
          f"{last_100_episodes_mean_score:2f}")

    return train_info


def train_for_an_episode(agent:PPO, env:ReacherEnv):
    dones = [False]*env.num_agents
    episode_info = {}
    obs = env.reset(train_mode=True)
    trajectory_processor = TrajectoryProcessor()
    trajectory_prep = TrajectoryPrep()
    scores = [0]*env.num_agents
    while not any(dones):
        actions_dists = [agent.take_action(ob) for ob in obs]
        values = [agent.compute_value(ob).detach().cpu().numpy() for ob in obs]
        actions = [action.detach().cpu().numpy()  for action, _ in actions_dists]
        log_probs = [dist.log_prob(action).detach().cpu().numpy() for action, dist in actions_dists]
        next_obs, rewards, dones, info = env.step(actions)
        scores = [s+r for s,r in zip(scores, rewards)]
        trajectory_prep.add(obs, actions, rewards, values, dones, log_probs)
        obs = next_obs
    next_value = [agent.compute_value(ob).detach().cpu().numpy() for ob in next_obs]
    trajectory_processor.process(*trajectory_prep.get(), next_value)
    for data in trajectory_processor.iterate(EPOCHS, BATCH_SIZE):
        episode_info["critic_loss"], episode_info["policy_loss"] = agent.update(*data)

    episode_info["average_score"] = np.mean(scores)

    return episode_info


def parse_args():
    valid_commands = ["train", "test", "eval"]
    args = sys.argv
    assert len(args) == 2 and args[0] == "main.py"
    if args[1] not in valid_commands:
        raise ValueError(f"command not identified! use either of valid_commands")

    return args[1]


def evaluate(agent:PPO, env:ReacherEnv):
    dones = [False]*env.num_agents
    scores = [0]*env.num_agents
    obs = env.reset(train_mode=False)
    while not any(dones):
        actions = [agent.take_action(ob)[0].detach().cpu().numpy() for ob in obs]
        next_obs, rewards, dones, info = env.step(actions)
        scores = [s+r for s,r in zip(scores, rewards)]
        obs = next_obs

    return np.mean(scores)


def plot_score(last_100_episode_score_mean:List[float], episode_scores:List[float]):
    filename = "score_plot"
    plt.figure(figsize=(30, 10))
    for plot, score, title in zip((121, 122), (last_100_episode_score_mean, episode_scores),
                                  ("last_100_episode_score_mean", "episode_scores")):
        plt.subplot(plot)
        plt.plot(score)
        plt.title(title)
        plt.ylabel(title)
        plt.xlabel("episodes")
        plt.savefig(f"./plots/{filename}.png")
    plt.close()


if __name__ == "__main__":
    command = parse_args()
    env_file_path = ENV_PATH
    env = ReacherEnv(env_file_path)
    agent = PPO()
    if command == "train":
        info = train_agent(agent, env, target_score=TARGET_SCORE)
    else:
        assert Path("./checkpoint/checkpoint.pt").exists()
        agent.restore("./checkpoint/checkpoint.pt")
        score = evaluate(agent, env)
        print(score)
