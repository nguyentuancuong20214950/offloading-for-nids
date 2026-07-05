import random
import time
from dataclasses import asdict, dataclass


LOCAL = 0
OFFLOAD = 1

STATE_NAMES = [
    "edge_cpu",
    "edge_ram",
    "packet_queue",
    "processing_latency",
    "bandwidth_used",
    "rtt",
    "packet_size",
]


@dataclass
class Packet:
    packet_id: int
    size: int
    port: int
    created_at: float
    label: str = "unknown"

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        return cls(
            packet_id=int(data["packet_id"]),
            size=int(data["size"]),
            port=int(data.get("port", 0)),
            created_at=float(data.get("created_at", time.time())),
            label=data.get("label", "unknown"),
        )


class NIDSOffloadingEnv:
    """Simulated edge/cloud NIDS environment.

    The environment keeps metrics simulated so weak-edge conditions can be
    reproduced on any laptop or VM. In edge.py, offload actions can still call a
    real TCP cloud backend through the cloud_client callback.
    """

    def __init__(self, seed=40, max_queue=120, cloud_client=None):
        self.random = random.Random(seed)
        self.max_queue = max_queue
        self.cloud_client = cloud_client
        self.packet_counter = 0
        self.reset()

    def reset(self):
        self.edge_cpu = self.random.uniform(25.0, 55.0)
        self.edge_ram = self.random.uniform(30.0, 55.0)
        self.packet_queue = self.random.randint(0, 35)
        self.processing_latency = self.random.uniform(4.0, 20.0)
        self.bandwidth_used = self.random.uniform(5.0, 35.0)
        self.rtt = self.random.uniform(12.0, 55.0)
        self.last_packet = self.generate_packet()
        self.stats = {
            "processed": 0,
            "local": 0,
            "offloaded": 0,
            "dropped": 0,
            "total_reward": 0.0,
            "total_latency": 0.0,
            "cpu_saved": 0.0,
        }
        return self.getstate()

    def generate_packet(self):
        self.packet_counter += 1
        size = int(self.random.triangular(64, 9000, 1000))
        port = self.random.choice([22, 53, 80, 123, 443, 445, 8080, 3389])
        return Packet(self.packet_counter, size, port, time.time())

    def getstate(self):
        packet_size = self.last_packet.size if self.last_packet else 0
        return [
            round(self.edge_cpu, 3),
            round(self.edge_ram, 3),
            float(self.packet_queue),
            round(self.processing_latency, 3),
            round(self.bandwidth_used, 3),
            round(self.rtt, 3),
            float(packet_size),
        ]

    def threshold_action(self):
        overloaded = (
            self.edge_cpu > 70.0
            or self.edge_ram > 75.0
            or self.packet_queue > 70
            or self.processing_latency > 45.0
        )
        network_ok = self.bandwidth_used < 85.0 and self.rtt < 120.0
        return OFFLOAD if overloaded and network_ok else LOCAL

    def process_local(self, packet):
        size_factor = packet.size / 1500.0
        queue_factor = self.packet_queue / max(1.0, self.max_queue)
        latency = 5.0 + (size_factor * 6.0) + (queue_factor * 40.0)
        cpu_cost = min(24.0, 4.0 + size_factor * 3.0)
        ram_cost = min(10.0, 1.5 + size_factor)
        self.edge_cpu = min(100.0, self.edge_cpu + cpu_cost)
        self.edge_ram = min(100.0, self.edge_ram + ram_cost)
        self.processing_latency = 0.75 * self.processing_latency + 0.25 * latency
        return {
            "packet_id": packet.packet_id,
            "status": "processed",
            "processor": "edge",
            "latency_ms": latency,
            "classification": "simulated",
        }

    def process_cloud_simulated(self, packet):
        upload_ms = (packet.size / 1024.0) * 1.8
        cloud_ms = 12.0 + (packet.size / 1500.0) * 2.5
        latency = self.rtt + upload_ms + cloud_ms
        bandwidth_cost = min(28.0, packet.size / 650.0)
        self.bandwidth_used = min(100.0, self.bandwidth_used + bandwidth_cost)
        return {
            "packet_id": packet.packet_id,
            "status": "processed",
            "processor": "cloud",
            "latency_ms": latency,
            "classification": "simulated",
        }

    def offload_to_cloud(self, packet):
        if self.cloud_client is None:
            return self.process_cloud_simulated(packet)
        start = time.time()
        result = self.cloud_client(packet.to_dict())
        result["latency_ms"] = max(0.1, (time.time() - start) * 1000.0)
        result.setdefault("processor", "cloud")
        result.setdefault("status", "processed")
        return result

    def step(self, action):
        packet = self.last_packet
        drop = self.packet_queue >= self.max_queue

        if drop:
            result = {
                "packet_id": packet.packet_id,
                "status": "dropped",
                "processor": "none",
                "latency_ms": 0.0,
                "classification": "not_processed",
            }
            reward = -120.0
            self.stats["dropped"] += 1
        elif action == OFFLOAD:
            result = self.offload_to_cloud(packet)
            reward = self.calculate_reward(action, result["latency_ms"], dropped=False)
            self.stats["offloaded"] += 1
            self.stats["cpu_saved"] += 8.0
        else:
            result = self.process_local(packet)
            reward = self.calculate_reward(action, result["latency_ms"], dropped=False)
            self.stats["local"] += 1

        self.stats["processed"] += 1 if result["status"] == "processed" else 0
        self.stats["total_reward"] += reward
        self.stats["total_latency"] += result["latency_ms"]
        self._advance_metrics(action, result["latency_ms"])
        self.last_packet = self.generate_packet()
        return self.getstate(), reward, result

    def calculate_reward(self, action, latency_ms, dropped):
        if dropped:
            return -120.0

        overload = max(0.0, self.edge_cpu - 70.0) + max(0.0, self.edge_ram - 75.0)
        queue_penalty = max(0.0, self.packet_queue - 70.0)
        bandwidth_penalty = max(0.0, self.bandwidth_used - 80.0)
        success_reward = 40.0
        cpu_saved = 14.0 if action == OFFLOAD and self.edge_cpu > 60.0 else 0.0
        unnecessary_offload = 18.0 if action == OFFLOAD and self.edge_cpu < 50.0 and self.packet_queue < 25 else 0.0

        return (
            success_reward
            + cpu_saved
            - (0.45 * latency_ms)
            - (1.2 * overload)
            - (0.7 * queue_penalty)
            - (0.8 * bandwidth_penalty)
            - unnecessary_offload
        )

    def _advance_metrics(self, action, latency_ms):
        arrival_pressure = self.random.randint(0, 8)
        processed = 2 if latency_ms < 35 else 1
        self.packet_queue = max(0, min(self.max_queue, self.packet_queue + arrival_pressure - processed))

        cooling = self.random.uniform(3.0, 8.0)
        if action == OFFLOAD:
            self.edge_cpu = max(10.0, self.edge_cpu - cooling)
            self.edge_ram = max(18.0, self.edge_ram - self.random.uniform(1.0, 4.0))
        else:
            self.edge_cpu = max(10.0, self.edge_cpu - cooling * 0.35)
            self.edge_ram = max(18.0, self.edge_ram - self.random.uniform(0.2, 1.0))

        self.bandwidth_used = max(0.0, self.bandwidth_used - self.random.uniform(3.0, 9.0))
        self.rtt = max(5.0, min(180.0, self.rtt + self.random.uniform(-4.0, 5.5)))

    def summary(self):
        processed = max(1, self.stats["processed"])
        total = processed + self.stats["dropped"]
        return {
            "processed": self.stats["processed"],
            "local": self.stats["local"],
            "offloaded": self.stats["offloaded"],
            "dropped": self.stats["dropped"],
            "drop_rate": self.stats["dropped"] / max(1, total),
            "avg_latency_ms": self.stats["total_latency"] / processed,
            "total_reward": self.stats["total_reward"],
            "cpu_saved": self.stats["cpu_saved"],
        }
