"""
Part 5: Evaluation Procedure for DQN on CartPole-v1

This notebook cell adds fixed-interval greedy evaluation to DQN training.

It:
- evaluates the current policy without exploration
- runs a fixed number of test episodes
- records average evaluation reward
- logs evaluation metrics during training

Assumes DQNAgent and DQNConfig are already defined in the notebook.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import copy
import logging
import os
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from dqn_components import DQNAgent, DQNConfig


def _get_logger() -> logging.Logger:
    """
    Logging setup for this script.

    Creates a logs/ directory next to this file and configures:
      - logs/evaluation.log : dedicated log for evaluation.py

    The logger also echoes to the console so output remains visible in Colab.
    Safe to call multiple times.
    """
    log_dir = Path(__file__).resolve().parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("evaluation")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s — %(message)s"
        )

        file_handler = logging.FileHandler(log_dir / "evaluation.log", mode="a")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        logger.propagate = False

    return logger


LOG = _get_logger()


@dataclass
class TrainingHistory:
    """Stores training and evaluation metrics across episodes."""

    episode_rewards: list[float] = field(default_factory=list)
    moving_average_rewards: list[float] = field(default_factory=list)
    episode_losses: list[float] = field(default_factory=list)
    epsilons: list[float] = field(default_factory=list)
    replay_sizes: list[int] = field(default_factory=list)

    evaluation_episodes: list[int] = field(default_factory=list)
    evaluation_mean_rewards: list[float] = field(default_factory=list)
    evaluation_max_rewards: list[float] = field(default_factory=list)
    evaluation_min_rewards: list[float] = field(default_factory=list)

    best_moving_average: float = 0.0
    best_episode_reward: float = 0.0
    best_evaluation_mean_reward: float = 0.0


class DQNTrainer:
    """DQN trainer with periodic greedy evaluation."""

    def __init__(
        self,
        config: DQNConfig,
        num_episodes: int = 500,
        target_update_every_episodes: int = 3,
        moving_average_window: int = 20,
        evaluation_interval: int = 25,
        num_eval_episodes: int = 10,
        early_stop_eval_reward: float = 475.0,
    ) -> None:
        self.config = config
        self.num_episodes = num_episodes
        self.target_update_every_episodes = target_update_every_episodes
        self.moving_average_window = moving_average_window
        self.evaluation_interval = evaluation_interval
        self.num_eval_episodes = num_eval_episodes
        self.early_stop_eval_reward = early_stop_eval_reward

        self.project_root = Path(__file__).resolve().parent
        self.checkpoint_dir = self.project_root / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.best_model_path = self.checkpoint_dir / "evaluation_best_model.pt"

        self.env = gym.make("CartPole-v1")
        self.eval_env = gym.make("CartPole-v1")
        self.agent = DQNAgent(config)
        self.history = TrainingHistory()

        self.best_q_state_dict = copy.deepcopy(self.agent.q_network.state_dict())
        self.best_target_state_dict = copy.deepcopy(
            self.agent.target_network.state_dict()
        )

        LOG.info(
            "Evaluation trainer initialized | episodes=%d eval_interval=%d eval_episodes=%d",
            self.num_episodes,
            self.evaluation_interval,
            self.num_eval_episodes,
        )

    def _train_step_with_clipping(self) -> float | None:
        """Perform one training step with gradient clipping."""
        if not self.agent.ready_to_train():
            return None

        loss = self.agent.compute_dqn_loss()

        self.agent.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.agent.q_network.parameters(),
            max_norm=10.0,
        )
        self.agent.optimizer.step()

        self.agent.training_steps += 1
        return float(loss.item())

    def evaluate_greedily(
        self,
        num_eval_episodes: int,
    ) -> tuple[float, float, float, list[float]]:
        """Evaluate the current policy without exploration."""
        rewards: list[float] = []

        for eval_index in range(num_eval_episodes):
            state, info = self.eval_env.reset(seed=self.config.seed + 10_000 + eval_index)
            terminated = False
            truncated = False
            total_reward = 0.0

            while not (terminated or truncated):
                action = self.agent.select_action(state=state, explore=False)
                next_state, reward, terminated, truncated, step_info = self.eval_env.step(
                    action
                )

                state = next_state
                total_reward += reward

            rewards.append(total_reward)

        reward_array = np.array(rewards, dtype=np.float32)
        mean_reward = float(reward_array.mean())
        max_reward = float(reward_array.max())
        min_reward = float(reward_array.min())

        return mean_reward, max_reward, min_reward, rewards

    def train(self) -> TrainingHistory:
        """Run full training with periodic evaluation."""
        for episode_index in range(1, self.num_episodes + 1):
            state, info = self.env.reset(seed=self.config.seed + episode_index)
            terminated = False
            truncated = False
            total_reward = 0.0
            episode_losses: list[float] = []

            while not (terminated or truncated):
                action = self.agent.select_action(state=state, explore=True)

                next_state, reward, terminated, truncated, step_info = self.env.step(
                    action
                )
                done = terminated or truncated

                self.agent.store_transition(
                    state=state,
                    action=action,
                    reward=reward,
                    next_state=next_state,
                    done=done,
                    info=step_info,
                )

                loss = self._train_step_with_clipping()
                if loss is not None:
                    episode_losses.append(loss)

                state = next_state
                total_reward += reward

            self.agent.update_epsilon()

            if episode_index % self.target_update_every_episodes == 0:
                self.agent.update_target_network()

            average_episode_loss = (
                float(np.mean(episode_losses)) if episode_losses else 0.0
            )

            self.history.episode_rewards.append(total_reward)
            moving_average_reward = float(
                np.mean(self.history.episode_rewards[-self.moving_average_window :])
            )
            self.history.moving_average_rewards.append(moving_average_reward)
            self.history.episode_losses.append(average_episode_loss)
            self.history.epsilons.append(self.agent.epsilon)
            self.history.replay_sizes.append(len(self.agent.replay_buffer))

            if total_reward > self.history.best_episode_reward:
                self.history.best_episode_reward = total_reward

            if moving_average_reward > self.history.best_moving_average:
                self.history.best_moving_average = moving_average_reward

            message = (
                f"Episode {episode_index:03d} | "
                f"Reward: {total_reward:6.1f} | "
                f"AvgReward({self.moving_average_window}): "
                f"{moving_average_reward:7.2f} | "
                f"Loss: {average_episode_loss:10.6f} | "
                f"Epsilon: {self.agent.epsilon:.4f} | "
                f"Replay: {len(self.agent.replay_buffer):6d}"
            )
            print(message)
            LOG.info(message)

            if episode_index % self.evaluation_interval == 0:
                eval_mean, eval_max, eval_min, eval_rewards = self.evaluate_greedily(
                    num_eval_episodes=self.num_eval_episodes
                )

                self.history.evaluation_episodes.append(episode_index)
                self.history.evaluation_mean_rewards.append(eval_mean)
                self.history.evaluation_max_rewards.append(eval_max)
                self.history.evaluation_min_rewards.append(eval_min)

                eval_message = (
                    f"Evaluation @ Episode {episode_index:03d} | "
                    f"Mean Reward: {eval_mean:7.2f} | "
                    f"Max Reward: {eval_max:6.1f} | "
                    f"Min Reward: {eval_min:6.1f}"
                )
                print(f"  {eval_message}")
                LOG.info(eval_message)

                if eval_mean > self.history.best_evaluation_mean_reward:
                    self.history.best_evaluation_mean_reward = eval_mean
                    self.best_q_state_dict = copy.deepcopy(
                        self.agent.q_network.state_dict()
                    )
                    self.best_target_state_dict = copy.deepcopy(
                        self.agent.target_network.state_dict()
                    )
                    self.agent.save(str(self.best_model_path))
                    LOG.info(
                        "New best evaluation mean %.2f at episode %d. Checkpoint saved to %s",
                        eval_mean,
                        episode_index,
                        self.best_model_path,
                    )

                if eval_mean >= self.early_stop_eval_reward:
                    stop_message = (
                        "Early stopping triggered: evaluation mean reward "
                        f"reached {eval_mean:.2f}."
                    )
                    print(f"\n{stop_message}")
                    LOG.info(stop_message)
                    break

        self.agent.q_network.load_state_dict(self.best_q_state_dict)
        self.agent.target_network.load_state_dict(self.best_target_state_dict)

        return self.history

    def close(self) -> None:
        """Close environments cleanly."""
        self.env.close()
        self.eval_env.close()
        LOG.info("Training and evaluation environments closed.")


def summarize_training_and_evaluation(history: TrainingHistory) -> None:
    """Print a concise final summary of training and evaluation."""
    print("\nFinal summary")
    print("-" * 60)
    print(f"Training episodes completed:     {len(history.episode_rewards)}")
    print(f"Best training episode reward:    {history.best_episode_reward:.2f}")
    print(f"Best moving average reward:      {history.best_moving_average:.2f}")

    if history.evaluation_mean_rewards:
        best_eval = max(history.evaluation_mean_rewards)
        final_eval = history.evaluation_mean_rewards[-1]
        print(f"Best evaluation mean reward:     {best_eval:.2f}")
        print(f"Final evaluation mean reward:    {final_eval:.2f}")
        print(f"Evaluation checkpoints run:      {len(history.evaluation_mean_rewards)}")
        LOG.info(
            "Final summary | training_episodes=%d best_train_reward=%.2f best_mavg=%.2f best_eval=%.2f final_eval=%.2f eval_runs=%d",
            len(history.episode_rewards),
            history.best_episode_reward,
            history.best_moving_average,
            best_eval,
            final_eval,
            len(history.evaluation_mean_rewards),
        )
    else:
        print("No evaluation checkpoints were run.")
        LOG.info(
            "Final summary | training_episodes=%d best_train_reward=%.2f best_mavg=%.2f no_eval_runs",
            len(history.episode_rewards),
            history.best_episode_reward,
            history.best_moving_average,
        )


if __name__ == "__main__":
    config = DQNConfig(
        state_dim=4,
        action_dim=2,
        hidden_dim_1=128,
        hidden_dim_2=128,
        learning_rate=5e-4,
        gamma=0.99,
        batch_size=64,
        replay_buffer_size=100_000,
        min_replay_size=2_000,
        target_update_every=1_000,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.985,
        seed=42,
    )

    num_episodes = int(os.environ.get("DQN_EVAL_TRAIN_EPISODES", "500"))
    target_update_every_episodes = int(
        os.environ.get("DQN_EVAL_TARGET_UPDATE_EPISODES", "3")
    )
    moving_average_window = int(os.environ.get("DQN_EVAL_MOVING_AVG_WINDOW", "20"))
    evaluation_interval = int(os.environ.get("DQN_EVAL_INTERVAL", "25"))
    num_eval_episodes = int(os.environ.get("DQN_NUM_EVAL_EPISODES", "10"))
    early_stop_eval_reward = float(os.environ.get("DQN_EVAL_EARLY_STOP", "475.0"))

    trainer = DQNTrainer(
        config=config,
        num_episodes=num_episodes,
        target_update_every_episodes=target_update_every_episodes,
        moving_average_window=moving_average_window,
        evaluation_interval=evaluation_interval,
        num_eval_episodes=num_eval_episodes,
        early_stop_eval_reward=early_stop_eval_reward,
    )

    try:
        training_history = trainer.train()
        summarize_training_and_evaluation(training_history)
    finally:
        trainer.close()
