import random


LOCAL = 0
OFFLOAD = 1
DROP_PENALTY = 300.0

STATE_NAMES = [
    "edge_cpu",
    "edge_ram",
    "flow_queue",
    "processing_latency",
    "bandwidth_used",
    "rtt",
    "flow_size",
]


def estimate_flow_size(flow):
    if flow is None:
        return 64
    if isinstance(flow, (int, float)):
        return max(64, int(float(flow)))
    if hasattr(flow, "to_dict"):
        flow = flow.to_dict()
    if not isinstance(flow, dict):
        return 64
    for key in ("flow_size", "size", "bytes"):
        if key in flow and flow.get(key) not in (None, ""):
            return max(64, int(float(flow.get(key))))
    sbytes = float(flow.get("sbytes") or 0)
    dbytes = float(flow.get("dbytes") or 0)
    spkts = float(flow.get("spkts") or 0)
    dpkts = float(flow.get("dpkts") or 0)
    byte_total = sbytes + dbytes
    if byte_total > 0:
        return max(64, int(byte_total))
    flow_packet_count = spkts + dpkts
    return max(64, int(flow_packet_count * 512))


class NIDSOffloadingEnv:
    """Edge/cloud NIDS offloading environment for real flow rows.

    Flows are provided by the caller as extracted feature rows. The
    environment models edge resource pressure and reward dynamics around those
    real flows instead of generating synthetic inputs.
    """

    def __init__(self, seed=40, max_queue=120):
        self.random = random.Random(seed)
        self.max_queue = max_queue
        self.reset()

    def reset(self):
        self.edge_cpu = self.random.uniform(25.0, 55.0)
        self.edge_ram = self.random.uniform(30.0, 55.0)
        self.flow_queue = self.random.randint(0, 35)
        self.processing_latency = self.random.uniform(4.0, 20.0)
        self.bandwidth_used = self.random.uniform(5.0, 35.0)
        self.rtt = self.random.uniform(12.0, 55.0)
        self.stats = {
            "processed": 0,
            "local": 0,
            "offloaded": 0,
            "dropped": 0,
            "total_reward": 0.0,
            "total_latency": 0.0,
            "cpu_saved": 0.0,
        }
        return self.getstate(64)

    def getstate(self, flow=None):
        flow_size = estimate_flow_size(flow)
        return [
            round(self.edge_cpu, 3),
            round(self.edge_ram, 3),
            float(self.flow_queue),
            round(self.processing_latency, 3),
            round(self.bandwidth_used, 3),
            round(self.rtt, 3),
            float(flow_size),
        ]

    def threshold_action(self):
        overloaded = (
            self.edge_cpu > 70.0
            or self.edge_ram > 75.0
            or self.flow_queue > 70
            or self.processing_latency > 45.0
        )
        network_ok = self.bandwidth_used < 85.0 and self.rtt < 120.0
        return OFFLOAD if overloaded and network_ok else LOCAL

    def estimate_local_latency(self, flow):
        flow_size = estimate_flow_size(flow)
        size_factor = flow_size / 1500.0
        queue_factor = self.flow_queue / max(1.0, self.max_queue)
        latency = 5.0 + (size_factor * 6.0) + (queue_factor * 40.0)
        cpu_cost = min(24.0, 4.0 + size_factor * 3.0)
        ram_cost = min(10.0, 1.5 + size_factor)
        self.edge_cpu = min(100.0, self.edge_cpu + cpu_cost)
        self.edge_ram = min(100.0, self.edge_ram + ram_cost)
        self.processing_latency = 0.75 * self.processing_latency + 0.25 * latency
        return latency

    def estimate_cloud_latency(self, flow):
        flow_size = estimate_flow_size(flow)
        upload_ms = (flow_size / 1024.0) * 1.8
        cloud_ms = 12.0 + (flow_size / 1500.0) * 2.5
        latency = self.rtt + upload_ms + cloud_ms
        bandwidth_cost = min(28.0, flow_size / 650.0)
        self.bandwidth_used = min(100.0, self.bandwidth_used + bandwidth_cost)
        return latency

    def step_flow(self, action, flow, observed_latency_ms=None):
        drop = self.flow_queue >= self.max_queue
        flow_size = estimate_flow_size(flow)

        if drop:
            result = {
                "status": "dropped",
                "processor": "none",
                "flow_size": flow_size,
                "latency_ms": 0.0,
            }
            reward = self.calculate_reward(action, 0.0, dropped=True)
            self.stats["dropped"] += 1
        elif action == OFFLOAD:
            latency_ms = observed_latency_ms
            if latency_ms is None:
                latency_ms = self.estimate_cloud_latency(flow)
            result = {
                "status": "processed",
                "processor": "cloud",
                "flow_size": flow_size,
                "latency_ms": latency_ms,
            }
            reward = self.calculate_reward(action, result["latency_ms"], dropped=False)
            self.stats["offloaded"] += 1
            self.stats["cpu_saved"] += 8.0
        else:
            latency_ms = self.estimate_local_latency(flow)
            result = {
                "status": "processed",
                "processor": "edge",
                "flow_size": flow_size,
                "latency_ms": latency_ms,
            }
            reward = self.calculate_reward(action, result["latency_ms"], dropped=False)
            self.stats["local"] += 1

        self.stats["processed"] += 1 if result["status"] == "processed" else 0
        self.stats["total_reward"] += reward
        self.stats["total_latency"] += result["latency_ms"]
        self._advance_metrics(action, result["latency_ms"])
        return self.getstate(flow), reward, result

    def calculate_reward(self, action, latency_ms, dropped):
        if dropped:
            return -DROP_PENALTY

        overload = max(0.0, self.edge_cpu - 70.0) + max(0.0, self.edge_ram - 75.0)
        queue_penalty = max(0.0, self.flow_queue - 70.0)
        drop_risk_penalty = max(0.0, self.flow_queue - (0.80 * self.max_queue))
        bandwidth_penalty = max(0.0, self.bandwidth_used - 80.0)
        success_reward = 40.0
        cpu_saved = 14.0 if action == OFFLOAD and self.edge_cpu > 60.0 else 0.0
        unnecessary_offload = 18.0 if action == OFFLOAD and self.edge_cpu < 50.0 and self.flow_queue < 25 else 0.0

        return (
            success_reward
            + cpu_saved
            - (0.45 * latency_ms)
            - (1.2 * overload)
            - (1.4 * queue_penalty)
            - (2.0 * drop_risk_penalty)
            - (0.8 * bandwidth_penalty)
            - unnecessary_offload
        )

    def _advance_metrics(self, action, latency_ms):
        arrival_pressure = self.random.randint(0, 8)
        processed = 2 if latency_ms < 35 else 1
        self.flow_queue = max(0, min(self.max_queue, self.flow_queue + arrival_pressure - processed))

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
