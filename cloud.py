import argparse
import csv
import json
import os
import socketserver
import time

from systemModel import Packet


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


def estimate_flow_size(flow):
    for key in ("packet_size", "size"):
        if key in flow:
            return max(1, int(float(flow.get(key) or 1)))
    sbytes = float(flow.get("sbytes") or 0)
    dbytes = float(flow.get("dbytes") or 0)
    return max(1, int(sbytes + dbytes))


def simulate_cloud_ids(packet):
    processing_delay = 0.015 + min(0.060, packet.size / 120000.0)
    time.sleep(processing_delay)
    return {
        "packet_id": packet.packet_id,
        "status": "processed",
        "processor": "cloud",
        "classification": "simulated",
        "processing_delay_ms": round(processing_delay * 1000.0, 3),
    }


def simulate_cloud_flow_analysis(request):
    flow = request.get("flow", {})
    flow_size = estimate_flow_size(flow)
    processing_delay = 0.020 + min(0.080, flow_size / 160000.0)
    time.sleep(processing_delay)
    return {
        "flow_id": request.get("flow_id"),
        "status": "processed",
        "processor": "cloud",
        "cloud_result": "deep_analysis_completed",
        "true_attack_cat": request.get("true_attack_cat", "unknown"),
        "edge_prediction": request.get("edge_prediction"),
        "edge_confidence": request.get("edge_confidence"),
        "reason": request.get("reason", "rl_offload"),
        "processing_delay_ms": round(processing_delay * 1000.0, 3),
    }


class CloudRequestHandler(socketserver.StreamRequestHandler):
    def handle(self):
        raw = self.rfile.readline().decode("utf-8").strip()
        if not raw:
            return

        try:
            request = json.loads(raw)
            if request.get("type") == "flow":
                result = simulate_cloud_flow_analysis(request)
            else:
                packet = Packet.from_dict(request)
                result = simulate_cloud_ids(packet)
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
                "Cloud handled flow={flow_id} packet={packet_id} reason={reason} status={status}".format(
                    flow_id=result.get("flow_id"),
                    packet_id=result.get("packet_id"),
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
            "flow_id": result.get("flow_id", result.get("packet_id")),
            "reason": result.get("reason"),
            "edge_prediction": result.get("edge_prediction"),
            "edge_confidence": result.get("edge_confidence"),
            "true_attack_cat": result.get("true_attack_cat"),
            "cloud_result": result.get("cloud_result", result.get("classification")),
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
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    with ReusableTCPServer((args.host, args.port), CloudRequestHandler) as server:
        server.log_path = args.log
        server.verbose = args.verbose
        server.log_fields = [
            "timestamp",
            "source_ip",
            "flow_id",
            "reason",
            "edge_prediction",
            "edge_confidence",
            "true_attack_cat",
            "cloud_result",
            "status",
            "processing_time_ms",
        ]
        print(f"Cloud IDS listening on {args.host}:{args.port}")
        print(f"Cloud request log: {args.log}")
        server.serve_forever()


if __name__ == "__main__":
    main()
