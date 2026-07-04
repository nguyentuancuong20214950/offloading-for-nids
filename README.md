# Adaptive Offloading for NIDS

This project adapts the original MEC task-offloading code into a prototype for
adaptive offloading in a Network Intrusion Detection System (NIDS).

The goal is to decide, for each packet/flow inspection task, whether the edge
device should process it locally or offload it to a cloud IDS backend.

## Architecture

The project is designed for a two-VM demo:

- VM1: Edge IDS + RL agent
  - Generates simulated traffic for now.
  - Observes simulated edge/network state.
  - Chooses local processing or cloud offloading.
  - Sends offloaded packets to VM2 over TCP.
- VM2: Cloud IDS backend
  - Receives offloaded packets.
  - Simulates heavier IDS processing.
  - Marks the packet as processed and returns the result.

The IDS classification is simulated for now. Later, the local and cloud
processing functions can be replaced with real IDS models or packet-capture
logic.

## State Vector

The RL policy observes these simulated values:

- `edge_cpu`
- `edge_ram`
- `packet_queue`
- `processing_latency`
- `bandwidth_used`
- `rtt`
- `packet_size`

## Actions

- `0`: process locally at the edge
- `1`: offload to cloud IDS

When action `1` is selected in `edge.py`, the packet is actually sent to the
cloud VM over a TCP socket and a result is received back.

## Reward

The reward encourages:

- successful packet inspection
- lower processing latency
- fewer packet drops
- lower edge overload
- reasonable bandwidth use
- CPU saving when the edge is busy

It penalizes unnecessary offloading when the edge is not busy.

## Files

- `systemModel.py`: packet model, simulated NIDS offloading environment, reward logic
- `Training.py`: trains and saves the RL offloading policy
- `Simulation.py`: compares local-only, threshold, and RL adaptive offloading
- `edge.py`: run on VM1, loads trained model and sends offloaded packets to cloud
- `cloud.py`: run on VM2, receives packets and returns simulated IDS results
- `DQN.py`, `Double.py`, `Dueling.py`: original deep RL files kept for reference/future work

## Quick Local Simulation

Train the RL policy:

```bash
python Training.py
```

Run comparison:

```bash
python Simulation.py
```

The comparison result is saved to:

```text
nids_offloading_results.csv
```

## Two-VM Demo

### 1. Copy the repo to both VMs

Both VMs should have Python 3 installed. This prototype only uses the Python
standard library.

### 2. Train the policy on VM1

On the edge VM:

```bash
python Training.py
```

This creates:

```text
offload_q_table.json
```

### 3. Start the cloud backend on VM2

On the cloud VM:

```bash
python cloud.py --host 0.0.0.0 --port 9000
```

Make sure VM1 can reach VM2 on port `9000`.

### 4. Run the edge sender on VM1

Replace `<VM2_IP>` with the cloud VM IP address:

```bash
python edge.py --cloud-host <VM2_IP> --cloud-port 9000 --strategy rl --packets 50
```

You can also run baseline strategies:

```bash
python edge.py --cloud-host <VM2_IP> --strategy local --packets 50
python edge.py --cloud-host <VM2_IP> --strategy threshold --packets 50
```

## Current Scope

Implemented now:

- simulated packet/flow tasks
- simulated edge metrics
- trained RL offloading policy saved to disk
- real TCP send/receive for offloaded packets in the two-VM demo
- simulated local IDS and cloud IDS processing
- local-only, threshold, and RL comparison

Future work:

- replace generated packets with real captured traffic or pcap replay
- replace simulated IDS processing with real lightweight/cloud IDS models
- optionally migrate the saved policy to DQN/Double DQN/Dueling DQN
- Currently, when the cloud backend (cloud.py) is started on VM2, it only displays a simple message such as “Cloud IDS listening on port 9000”. While this confirms the service is running, it does not provide visibility into whether packets are being received, processed, or results are being sent back to the edge.

For easier debugging and demonstration, should include:

Logging incoming requests (e.g., timestamp, source IP, packet size).

Displaying processing status on screen (e.g., “Received packet #42, offloaded to RL module”).

Returning confirmation messages to the edge client and logging them.

Optional verbose/debug mode to toggle detailed runtime information.

This will make it clearer that the system is actively handling traffic, not just passively listening.
