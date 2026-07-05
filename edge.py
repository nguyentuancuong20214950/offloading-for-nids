import argparse
import csv
import json
import os
import socket
import time

from systemModel import LOCAL, OFFLOAD, NIDSOffloadingEnv
from Training import MODEL_PATH, QLearningOffloadAgent, train_agent


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_UNSW_TEST_PATH = os.path.join(
    BASE_DIR,
    "ids_data",
    "unsw_nb15",
    "Training and Testing Sets",
    "UNSW_NB15_testing-set.csv",
)
DEFAULT_EDGE_IDS_LOG = os.path.join(BASE_DIR, "results", "edge_ids", "edge_ids_offloading_demo.csv")
DEFAULT_EDGE_IDS_MODEL_DIR = os.path.join(BASE_DIR, "models", "edge_ids")


def load_pandas():
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError(
            "pandas is required for --strategy ids-rl. "
            "Install edge IDS dependencies with: pip install -r requirements-edge-ids.txt"
        ) from exc
    return pd


def ensure_parent_dir(path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def append_csv_row(path, fieldnames, row):
    ensure_parent_dir(path)
    file_exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def send_to_cloud(host, port, timeout):
    def client(payload):
        with socket.create_connection((host, port), timeout=timeout) as sock:
            request = json.dumps(payload).encode("utf-8") + b"\n"
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


def normalize_attack_name(value):
    pd = load_pandas()
    if pd.isna(value):
        return "Normal"
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "-", "normal"}:
        return "Normal"
    return text


def flow_to_cloud_payload(flow_id, flow, reason, true_attack_cat, edge_prediction=None, edge_confidence=None):
    pd = load_pandas()
    clean_flow = {}
    for key, value in flow.items():
        if pd.isna(value):
            clean_flow[key] = None
        elif hasattr(value, "item"):
            clean_flow[key] = value.item()
        else:
            clean_flow[key] = value

    return {
        "type": "flow",
        "flow_id": flow_id,
        "flow": clean_flow,
        "reason": reason,
        "true_attack_cat": true_attack_cat,
        "edge_prediction": edge_prediction,
        "edge_confidence": edge_confidence,
    }


def estimate_flow_size(flow):
    sbytes = float(flow.get("sbytes") or 0)
    dbytes = float(flow.get("dbytes") or 0)
    return max(64, int(sbytes + dbytes))


def make_state_for_flow(env, flow):
    size = estimate_flow_size(flow)
    return [
        round(env.edge_cpu, 3),
        round(env.edge_ram, 3),
        float(env.packet_queue),
        round(env.processing_latency, 3),
        round(env.bandwidth_used, 3),
        round(env.rtt, 3),
        float(size),
    ]


def advance_flow_metrics(env, action, latency_ms):
    reward = env.calculate_reward(action, latency_ms, dropped=False)
    env.stats["processed"] += 1
    env.stats["total_reward"] += reward
    env.stats["total_latency"] += latency_ms
    if action == OFFLOAD:
        env.stats["offloaded"] += 1
        env.stats["cpu_saved"] += 8.0
    else:
        env.stats["local"] += 1
    env._advance_metrics(action, latency_ms)
    return reward


def run_ids_rl_demo(args):
    pd = load_pandas()
    from edge_ids_predictor import EdgeIDSPredictor

    if not os.path.exists(args.flows_csv):
        raise FileNotFoundError(f"UNSW-NB15 test CSV not found: {args.flows_csv}")

    cloud_client = send_to_cloud(args.cloud_host, args.cloud_port, args.timeout)
    agent = load_or_train_model(args.model, args.train_if_missing)
    ids = EdgeIDSPredictor(model_path=args.ids_model, label_encoder_path=args.ids_label_encoder, schema_path=args.ids_schema)
    env = NIDSOffloadingEnv(seed=args.seed)
    env.reset()

    flows = pd.read_csv(args.flows_csv)
    if args.flows:
        flows = flows.head(args.flows)

    log_fields = [
        "timestamp",
        "flow_id",
        "true_attack_cat",
        "rl_decision",
        "edge_ids_used",
        "edge_prediction",
        "edge_confidence",
        "ids_escalated_to_cloud",
        "final_location",
        "reason",
        "latency_ms",
        "reward",
    ]

    print("Edge IDS/RL live-flow demo started")
    print(f"Cloud target: {args.cloud_host}:{args.cloud_port}")
    print(f"UNSW live-flow source: {args.flows_csv}")
    print(f"Edge log: {args.edge_log}")

    for index, row in flows.iterrows():
        flow_id = int(row["id"]) if "id" in row and not pd.isna(row["id"]) else index + 1
        true_attack_cat = normalize_attack_name(row.get("attack_cat", "unknown"))
        flow = row.to_dict()
        state = make_state_for_flow(env, flow)
        action = agent.choose_action(state, explore=False)
        start = time.time()

        edge_ids_used = False
        edge_prediction = None
        edge_confidence = None
        ids_escalated = False

        if action == OFFLOAD:
            reason = "rl_resource_offload"
            response = cloud_client(flow_to_cloud_payload(flow_id, flow, reason, true_attack_cat))
            latency_ms = float(response.get("latency_ms", (time.time() - start) * 1000.0))
            final_location = "cloud"
            rl_decision = "cloud"
            reward = advance_flow_metrics(env, OFFLOAD, latency_ms)
        else:
            rl_decision = "local"
            edge_ids_used = True
            prediction = ids.predict(row)
            edge_prediction = prediction["attack_category"]
            edge_confidence = prediction.get("confidence")
            local_latency_ms = max(0.1, (time.time() - start) * 1000.0)

            if edge_prediction != "Normal":
                reason = "edge_predicted_attack"
                ids_escalated = True
            elif edge_confidence is not None and edge_confidence < args.confidence_threshold:
                reason = "low_confidence_normal"
                ids_escalated = True
            else:
                reason = "confident_local_normal"

            if ids_escalated:
                cloud_start = time.time()
                response = cloud_client(
                    flow_to_cloud_payload(
                        flow_id,
                        flow,
                        reason,
                        true_attack_cat,
                        edge_prediction=edge_prediction,
                        edge_confidence=edge_confidence,
                    )
                )
                cloud_latency_ms = float(response.get("latency_ms", (time.time() - cloud_start) * 1000.0))
                latency_ms = local_latency_ms + cloud_latency_ms
                final_location = "cloud"
                reward = advance_flow_metrics(env, OFFLOAD, latency_ms)
            else:
                latency_ms = local_latency_ms
                final_location = "edge"
                reward = advance_flow_metrics(env, LOCAL, latency_ms)

        print(
            "Flow {flow_id} | RL={rl_decision} | edge={edge_prediction} "
            "conf={edge_confidence} | final={final_location} | reason={reason} | true={true_attack_cat}".format(
                flow_id=flow_id,
                rl_decision=rl_decision,
                edge_prediction=edge_prediction or "skipped",
                edge_confidence="n/a" if edge_confidence is None else f"{edge_confidence:.3f}",
                final_location=final_location,
                reason=reason,
                true_attack_cat=true_attack_cat,
            )
        )

        try:
            append_csv_row(
                args.edge_log,
                log_fields,
                {
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "flow_id": flow_id,
                    "true_attack_cat": true_attack_cat,
                    "rl_decision": rl_decision,
                    "edge_ids_used": edge_ids_used,
                    "edge_prediction": edge_prediction,
                    "edge_confidence": edge_confidence,
                    "ids_escalated_to_cloud": ids_escalated,
                    "final_location": final_location,
                    "reason": reason,
                    "latency_ms": latency_ms,
                    "reward": reward,
                },
            )
        except OSError as exc:
            print(f"Edge log write skipped: {exc}")
        time.sleep(args.interval)

    print("Summary:")
    for key, value in env.summary().items():
        print(f"  {key}: {value}")


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
    parser.add_argument("--strategy", choices=["local", "threshold", "rl", "ids-rl"], default="rl")
    parser.add_argument("--packets", type=int, default=50)
    parser.add_argument("--flows", type=int, default=50, help="Number of UNSW-NB15 rows to replay in ids-rl mode.")
    parser.add_argument("--flows-csv", default=DEFAULT_UNSW_TEST_PATH, help="UNSW-NB15 testing CSV used as live-flow input.")
    parser.add_argument("--confidence-threshold", type=float, default=0.90)
    parser.add_argument("--edge-log", default=DEFAULT_EDGE_IDS_LOG)
    parser.add_argument("--ids-model", default=os.path.join(DEFAULT_EDGE_IDS_MODEL_DIR, "best_edge_ids_model.joblib"))
    parser.add_argument("--ids-label-encoder", default=os.path.join(DEFAULT_EDGE_IDS_MODEL_DIR, "label_encoder.joblib"))
    parser.add_argument("--ids-schema", default=os.path.join(DEFAULT_EDGE_IDS_MODEL_DIR, "feature_schema.json"))
    parser.add_argument("--interval", type=float, default=0.05)
    parser.add_argument("--timeout", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=40)
    parser.add_argument("--model", default=MODEL_PATH)
    parser.add_argument("--train-if-missing", action="store_true")
    args = parser.parse_args()
    if args.strategy == "ids-rl":
        run_ids_rl_demo(args)
    else:
        run_edge_demo(args)


if __name__ == "__main__":
    main()
