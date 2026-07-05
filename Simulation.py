import argparse
import csv
import json
import os

from systemModel import LOCAL, NIDSOffloadingEnv
from Training import MODEL_PATH, QLearningOffloadAgent, train_agent


def run_strategy(strategy, agent=None, packets=600, seed=40):
    env = NIDSOffloadingEnv(seed=seed)
    state = env.reset()

    for _ in range(packets):
        if strategy == "local":
            action = LOCAL
        elif strategy == "threshold":
            action = env.threshold_action()
        elif strategy == "rl":
            action = agent.choose_action(state, explore=False)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        state, _, _ = env.step(action)

    return env.summary()


def ensure_model(model_path):
    if os.path.exists(model_path):
        try:
            agent = QLearningOffloadAgent.load(model_path)
            if len(agent.q_table) > 0:
                return agent
            print("Existing model is empty. Retraining...")
        except (KeyError, ValueError, OSError, json.JSONDecodeError, NameError):
            print("Existing model is invalid. Retraining...")

    print("No trained model found. Training before simulation...")
    agent = train_agent()
    agent.save(model_path)
    return agent


def print_summary(results):
    print("\nAdaptive NIDS offloading comparison")
    print("-" * 76)
    print(f"{'strategy':<14} {'processed':>10} {'offloaded':>10} {'drops':>8} {'drop_rate':>10} {'avg_lat':>10} {'reward':>10}")
    print("-" * 76)
    for name, summary in results.items():
        print(
            f"{name:<14} "
            f"{summary['processed']:>10} "
            f"{summary['offloaded']:>10} "
            f"{summary['dropped']:>8} "
            f"{summary['drop_rate']:>10.3f} "
            f"{summary['avg_latency_ms']:>10.2f} "
            f"{summary['total_reward']:>10.2f}"
        )


def save_results(results, path):
    result_dir = os.path.dirname(path)
    if result_dir:
        os.makedirs(result_dir, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["strategy", "processed", "local", "offloaded", "dropped", "drop_rate", "avg_latency_ms", "total_reward", "cpu_saved"])
        for name, summary in results.items():
            writer.writerow([
                name,
                summary["processed"],
                summary["local"],
                summary["offloaded"],
                summary["dropped"],
                summary["drop_rate"],
                summary["avg_latency_ms"],
                summary["total_reward"],
                summary["cpu_saved"],
            ])


def main():
    parser = argparse.ArgumentParser(description="Compare local-only, threshold, and RL adaptive NIDS offloading.")
    parser.add_argument("--packets", type=int, default=600)
    parser.add_argument("--seed", type=int, default=40)
    parser.add_argument("--model", default=MODEL_PATH)
    parser.add_argument("--out", default="nids_offloading_results.csv")
    args = parser.parse_args()

    agent = ensure_model(args.model)
    results = {
        "local": run_strategy("local", packets=args.packets, seed=args.seed),
        "threshold": run_strategy("threshold", packets=args.packets, seed=args.seed),
        "rl": run_strategy("rl", agent=agent, packets=args.packets, seed=args.seed),
    }
    print_summary(results)
    save_results(results, args.out)
    print(f"\nSaved results to {args.out}")


if __name__ == "__main__":
    main()
