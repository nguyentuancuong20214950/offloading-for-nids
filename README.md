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

For a more visible demo with request logging:

```bash
python cloud.py --host 0.0.0.0 --port 9000 --verbose --log results/cloud_ids/cloud_requests_log.csv
```

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

### 5. Run the RL-first IDS flow demo

After training the edge IDS, `edge.py` can replay rows from the UNSW-NB15
testing CSV as simulated live flows:

```bash
python edge.py --cloud-host <VM2_IP> --cloud-port 9000 --strategy ids-rl --flows 50 --confidence-threshold 0.90
```

The order is:

1. RL/offloading policy decides first from edge resource/network state.
2. If RL chooses cloud, the flow is sent to VM2 immediately and edge IDS is skipped.
3. If RL chooses local, the lightweight edge IDS classifies the flow.
4. If the local IDS predicts an attack, the flow is escalated to cloud.
5. If the local IDS predicts `Normal` with confidence below the threshold, the flow is also escalated.
6. Only high-confidence local `Normal` results stay at the edge.

Edge-side demo logs are saved to:

```text
results/edge_ids/edge_ids_offloading_demo.csv
```

Cloud-side request logs are saved to:

```text
results/cloud_ids/cloud_requests_log.csv
```

## Current Scope

Implemented now:

- simulated packet/flow tasks
- simulated edge metrics
- trained RL offloading policy saved to disk
- real TCP send/receive for offloaded packets in the two-VM demo
- simulated local IDS and cloud IDS processing
- local-only, threshold, and RL comparison
- lightweight multi-class edge IDS training on UNSW-NB15
- saved edge IDS models and visual comparison reports

## Edge IDS Training

The edge IDS is trained offline with the official processed UNSW-NB15 split:

```text
ids_data/unsw_nb15/Training and Testing Sets/UNSW_NB15_training-set.csv
ids_data/unsw_nb15/Training and Testing Sets/UNSW_NB15_testing-set.csv
```

Use these two files instead of the raw `UNSW-NB15_1.csv`, `UNSW-NB15_2.csv`,
etc. The processed split already includes the `attack_cat` column used for
multi-class labels.

The existing `data/` folder is kept for offloading simulation artifacts. IDS
datasets are kept under `ids_data/` so the two parts of the project stay
separate.

Train the lightweight edge IDS models:

```bash
pip install -r requirements-edge-ids.txt
python edge_ids_train.py
```

This trains and saves three models:

```text
models/edge_ids/random_forest.joblib
models/edge_ids/extra_trees.joblib
models/edge_ids/logistic_regression.joblib
models/edge_ids/best_edge_ids_model.joblib
```

The selected best model is chosen by macro F1 score because UNSW-NB15 attack
classes are imbalanced.

Training also saves reusable metadata:

```text
models/edge_ids/label_encoder.joblib
models/edge_ids/feature_schema.json
models/edge_ids/best_model_info.json
```

Comparison outputs are saved here:

```text
results/edge_ids/edge_ids_model_comparison.csv
results/edge_ids/classification_report_random_forest.csv
results/edge_ids/classification_report_extra_trees.csv
results/edge_ids/classification_report_logistic_regression.csv
results/edge_ids/confusion_matrix_random_forest.png
results/edge_ids/confusion_matrix_extra_trees.png
results/edge_ids/confusion_matrix_logistic_regression.png
```

For a smaller machine, train with a stratified sample:

```bash
python edge_ids_train.py --max-train-rows 50000
```

## Edge IDS Runtime Input

UNSW-NB15 is used for training and evaluation only. In the full VM demo, real
traffic should be captured or generated, converted into flow features, and then
passed to the trained edge IDS model.

The runtime flow extractor must produce the same feature columns saved in:

```text
models/edge_ids/feature_schema.json
```

Later, a flow extractor such as CICFlowMeter may be used, but its output columns
may not match UNSW-NB15 directly. If that happens, add a feature-mapping layer
before calling `edge_ids_predictor.py`.

Run predictions on a CSV of extracted flow features:

```bash
python edge_ids_predictor.py --input-csv extracted_flows.csv --output-csv edge_ids_predictions.csv
```

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

- shape reward: tune parameter so RL learns smater policies 
