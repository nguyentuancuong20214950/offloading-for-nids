import argparse
import csv
import json
import os
import socketserver
import time

from cloud_ids_predictor import (
    CloudIDSPredictor,
    DEFAULT_LABEL_ENCODER_PATH,
    DEFAULT_MODEL_PATH,
    DEFAULT_SCHEMA_PATH,
)


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


def analyze_cloud_flow(request, predictor):
    flow = request.get("flow", {})
    start = time.time()
    prediction = predictor.predict(flow)
    elapsed_ms = (time.time() - start) * 1000.0

    return {
        "flow_id": request.get("flow_id"),
        "status": "processed",
        "processor": "cloud",
        "cloud_result": prediction.get("attack_category"),
        "cloud_confidence": prediction.get("confidence"),
        "cloud_class_probabilities": prediction.get("class_probabilities"),
        "true_attack_cat": request.get("true_attack_cat", "unknown"),
        "edge_prediction": request.get("edge_prediction"),
        "edge_confidence": request.get("edge_confidence"),
        "reason": request.get("reason", "rl_offload"),
        "processing_delay_ms": round(elapsed_ms, 3),
    }


class CloudRequestHandler(socketserver.StreamRequestHandler):
    def handle(self):
        raw = self.rfile.readline().decode("utf-8").strip()
        if not raw:
            return

        try:
            request = json.loads(raw)
            if request.get("type") != "flow":
                raise ValueError("cloud.py only accepts real flow requests with type='flow'.")
            result = analyze_cloud_flow(request, self.server.ids_predictor)
        except Exception as exc:
            result = {
                "status": "error",
                "processor": "cloud",
                "error": str(exc),
            }

        self.log_request(result)
        self.wfile.write((json.dumps(result) + "\n").encode("utf-8"))

    def log_request(self, result):
        if getattr(self.server, "verbose", False):
            print(
                "Cloud handled flow={flow_id} reason={reason} status={status}".format(
                    flow_id=result.get("flow_id"),
                    reason=result.get("reason"),
                    status=result.get("status"),
                )
            )

        log_path = getattr(self.server, "log_path", None)
        if not log_path:
            return

        row = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "source_ip": self.client_address[0],
            "flow_id": result.get("flow_id"),
            "reason": result.get("reason"),
            "edge_prediction": result.get("edge_prediction"),
            "edge_confidence": result.get("edge_confidence"),
            "true_attack_cat": result.get("true_attack_cat"),
            "cloud_result": result.get("cloud_result"),
            "cloud_confidence": result.get("cloud_confidence"),
            "status": result.get("status"),
            "processing_time_ms": result.get("processing_delay_ms"),
        }
        try:
            append_csv_row(log_path, self.server.log_fields, row)
        except OSError as exc:
            print(f"Cloud log write skipped: {exc}")


class ReusableTCPServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True


def main():
    parser = argparse.ArgumentParser(description="Cloud IDS backend for the two-VM NIDS offloading demo.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--log", default="results/cloud_ids/cloud_requests_log.csv")
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--label-encoder", default=DEFAULT_LABEL_ENCODER_PATH)
    parser.add_argument("--schema", default=DEFAULT_SCHEMA_PATH)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    ids_predictor = CloudIDSPredictor(args.model, args.label_encoder, args.schema)
    print("Loaded cloud IDS model: {}".format(args.model))

    with ReusableTCPServer((args.host, args.port), CloudRequestHandler) as server:
        server.log_path = args.log
        server.verbose = args.verbose
        server.ids_predictor = ids_predictor
        server.log_fields = [
            "timestamp",
            "source_ip",
            "flow_id",
            "reason",
            "edge_prediction",
            "edge_confidence",
            "true_attack_cat",
            "cloud_result",
            "cloud_confidence",
            "status",
            "processing_time_ms",
        ]
        print(f"Cloud IDS listening on {args.host}:{args.port}")
        print(f"Cloud request log: {args.log}")
        server.serve_forever()


if __name__ == "__main__":
    main()
