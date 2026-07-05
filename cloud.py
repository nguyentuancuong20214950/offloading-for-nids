import argparse
import json
import socketserver
import time

from systemModel import Packet


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


class CloudRequestHandler(socketserver.StreamRequestHandler):
    def handle(self):
        raw = self.rfile.readline().decode("utf-8").strip()
        if not raw:
            return

        try:
            packet = Packet.from_dict(json.loads(raw))
            result = simulate_cloud_ids(packet)
        except Exception as exc:
            result = {
                "status": "error",
                "processor": "cloud",
                "error": str(exc),
            }

        self.wfile.write((json.dumps(result) + "\n").encode("utf-8"))


class ReusableTCPServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True


def main():
    parser = argparse.ArgumentParser(description="Cloud IDS backend for the two-VM NIDS offloading demo.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9000)
    args = parser.parse_args()

    with ReusableTCPServer((args.host, args.port), CloudRequestHandler) as server:
        print(f"Cloud IDS listening on {args.host}:{args.port}")
        server.serve_forever()


if __name__ == "__main__":
    main()
