import argparse
import json
import os
import random
from collections import defaultdict

from edge_ids_train import DEFAULT_TRAIN_PATH, build_feature_frame
from systemModel import LOCAL, OFFLOAD, NIDSOffloadingEnv, STATE_NAMES


MODEL_PATH = "offload_q_table.json"


class QLearningOffloadAgent:
    """Small saved RL policy for the two-VM demo.

    The original DQN/Double/Dueling files are kept in the repo. This agent is
    intentionally lightweight so the edge VM can load a trained policy without
    TensorFlow setup problems during the first prototype.
    """

    def __init__(self, alpha=0.12, gamma=0.90, epsilon=0.15):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = defaultdict(lambda: [0.0, 0.0])

    def discretize(self, state):
        edge_cpu, edge_ram, queue, latency, bandwidth, rtt, flow_size = state
        return (
            self._bin(edge_cpu, [35, 55, 70, 85]),
            self._bin(edge_ram, [40, 60, 75, 90]),
            self._bin(queue, [15, 35, 70, 100]),
            self._bin(latency, [10, 25, 45, 80]),
            self._bin(bandwidth, [25, 50, 80, 92]),
            self._bin(rtt, [25, 55, 100, 140]),
            self._bin(flow_size, [300, 900, 1500, 4500]),
        )

    @staticmethod
    def _bin(value, thresholds):
        for index, threshold in enumerate(thresholds):
            if value <= threshold:
                return index
        return len(thresholds)

    def choose_action(self, state, explore=True):
        key = self.discretize(state)
        if explore and random.random() < self.epsilon:
            return random.choice([LOCAL, OFFLOAD])
        values = self.q_table[key]
        return OFFLOAD if values[OFFLOAD] > values[LOCAL] else LOCAL

    def learn(self, state, action, reward, next_state):
        key = self.discretize(state)
        next_key = self.discretize(next_state)
        old_value = self.q_table[key][action]
        best_next = max(self.q_table[next_key])
        self.q_table[key][action] = old_value + self.alpha * (
            reward + self.gamma * best_next - old_value
        )

    def save(self, path=MODEL_PATH):
        model_dir = os.path.dirname(path)
        if model_dir:
            os.makedirs(model_dir, exist_ok=True)
        payload = {
            "state_names": STATE_NAMES,
            "alpha": self.alpha,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "q_table": {"|".join(map(str, key)): value for key, value in self.q_table.items()},
        }
        with open(path, "w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2)

    @classmethod
    def load(cls, path=MODEL_PATH):
        with open(path, "r", encoding="utf-8-sig") as file:
            payload = json.load(file)
        agent = cls(
            alpha=float(payload.get("alpha", 0.12)),
            gamma=float(payload.get("gamma", 0.90)),
            epsilon=float(payload.get("epsilon", 0.0)),
        )
        agent.epsilon = 0.0
        for key, value in payload["q_table"].items():
            agent.q_table[tuple(int(part) for part in key.split("|"))] = [float(value[0]), float(value[1])]
        return agent


def load_training_flows(path, max_rows=None):
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError(
            "pandas is required to train the offloading policy from real flows. "
            "Install dependencies with: pip install -r requirements-edge-ids.txt"
        ) from exc

    df = pd.read_csv(path)
    flows = build_feature_frame(df)
    if max_rows:
        flows = flows.head(max_rows)
    if flows.empty:
        raise ValueError(f"No flow rows found in {path}")
    return flows


def train_agent(episodes=450, steps_per_episode=180, seed=40, flows_csv=DEFAULT_TRAIN_PATH, max_train_rows=None):
    random.seed(seed)
    agent = QLearningOffloadAgent()
    flows = load_training_flows(flows_csv, max_train_rows)

    for episode in range(episodes):
        env = NIDSOffloadingEnv(seed=seed + episode)
        env.reset()
        sampled = flows.sample(
            n=steps_per_episode,
            replace=steps_per_episode > len(flows),
            random_state=seed + episode,
        )
        flow_rows = [row for _, row in sampled.iterrows()]
        state = env.getstate(flow_rows[0])
        for index, flow in enumerate(flow_rows):
            action = agent.choose_action(state, explore=True)
            _, reward, _ = env.step_flow(action, flow)
            if index + 1 < len(flow_rows):
                next_state = env.getstate(flow_rows[index + 1])
            else:
                next_state = env.getstate(flow)
            agent.learn(state, action, reward, next_state)
            state = next_state

        agent.epsilon = max(0.02, agent.epsilon * 0.995)

    return agent


def main():
    parser = argparse.ArgumentParser(description="Train the adaptive NIDS offloading RL policy.")
    parser.add_argument("--episodes", type=int, default=450)
    parser.add_argument("--steps", type=int, default=180)
    parser.add_argument("--flows-csv", default=DEFAULT_TRAIN_PATH, help="CSV of real flow features used for RL training.")
    parser.add_argument("--max-train-rows", type=int, default=None)
    parser.add_argument("--model", default=MODEL_PATH)
    args = parser.parse_args()

    agent = train_agent(args.episodes, args.steps, flows_csv=args.flows_csv, max_train_rows=args.max_train_rows)
    agent.save(args.model)
    print(f"Saved trained offloading policy to {args.model}")


if __name__ == "__main__":
    main()
