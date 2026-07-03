import argparse
import json
import os
import socket
import time

from systemModel import LOCAL, OFFLOAD, NIDSOffloadingEnv
from Training import MODEL_PATH, QLearningOffloadAgent, train_agent


def send_to_cloud(host, port, timeout):
    def client(packet_dict):
        with socket.create_connection((host, port), timeout=timeout) as sock:
            request = json.dumps(packet_dict).encode("utf-8") + b"\n"
            sock.sendall(request)
            response = b""
            while not response.endswith(b"\n"):
                chunk = sock.recv(4096)
                if not chunk:
                    break
                response += chunk
        return json.loads(response.decode("utf-8"))

    return client


def load_or_train_model(model_path, train_if_missing):
    if os.path.exists(model_path):
        try:
            agent = QLearningOffloadAgent.load(model_path)
            if len(agent.q_table) > 0:
                return agent
            message = f"Model file is empty: {model_path}."
        except Exception as exc:
            message = f"Model file is invalid: {model_path}. {exc}"
        if not train_if_missing:
            raise RuntimeError(f"{message} Run 'python Training.py' first.")
    if not train_if_missing:
        raise FileNotFoundError(
            f"Model file not found: {model_path}. Run 'python Training.py' first."
        )
    print("No trained model found. Training a small policy now...")
    agent = train_agent(episodes=250, steps_per_episode=140)
    agent.save(model_path)
    return agent


def run_edge_demo(args):
    cloud_client = send_to_cloud(args.cloud_host, args.cloud_port, args.timeout)
    env = NIDSOffloadingEnv(seed=args.seed, cloud_client=cloud_client)

    if args.strategy == "rl":
        agent = load_or_train_model(args.model, args.train_if_missing)
    else:
        agent = None

    state = env.reset()
    print(f"Edge demo started with strategy={args.strategy}")
    print(f"Cloud target: {args.cloud_host}:{args.cloud_port}")

    for _ in range(args.packets):
        if args.strategy == "local":
            action = LOCAL
        elif args.strategy == "threshold":
            action = env.threshold_action()
        else:
            action = agent.choose_action(state, explore=False)

        try:
            next_state, reward, result = env.step(action)
        except OSError as exc:
            print(f"Cloud send failed for packet {env.last_packet.packet_id}: {exc}")
            next_state, reward, result = env.step(LOCAL)

        action_name = "OFFLOAD" if action == OFFLOAD else "LOCAL"
        print(
            f"packet={result.get('packet_id')} action={action_name} "
            f"processor={result.get('processor')} status={result.get('status')} "
            f"latency_ms={result.get('latency_ms', 0):.2f} reward={reward:.2f}"
        )
        state = next_state
        time.sleep(args.interval)

    print("Summary:")
    for key, value in env.summary().items():
        print(f"  {key}: {value}")


def main():
    parser = argparse.ArgumentParser(description="Edge IDS + RL offloading sender for VM1.")
    parser.add_argument("--cloud-host", required=True, help="IP address of VM2 running cloud.py")
    parser.add_argument("--cloud-port", type=int, default=9000)
    parser.add_argument("--strategy", choices=["local", "threshold", "rl"], default="rl")
    parser.add_argument("--packets", type=int, default=50)
    parser.add_argument("--interval", type=float, default=0.05)
    parser.add_argument("--timeout", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=40)
    parser.add_argument("--model", default=MODEL_PATH)
    parser.add_argument("--train-if-missing", action="store_true")
    args = parser.parse_args()
    run_edge_demo(args)


if __name__ == "__main__":
    main()
