# Adaptive Offloading for NIDS

This project adapts the original MEC task-offloading code into a prototype for
adaptive offloading in a Network Intrusion Detection System (NIDS).

The goal is to decide, for each real flow inspection task, whether the edge
device should process it locally or offload it to a cloud IDS backend.

## Architecture

The project is designed for a two-VM demo:

- VM1: Edge IDS + RL agent
  - Replays extracted flow feature rows.
  - Observes simulated edge/network state.
  - Chooses local processing or cloud offloading.
  - Sends offloaded flow features to VM2 over TCP.
- VM2: Cloud IDS backend
  - Receives offloaded flow features.
  - Runs the trained cloud IDS model.
  - Returns the cloud IDS prediction, confidence, and class probabilities.

The edge and cloud IDS classifiers are trained on UNSW-NB15 feature rows. A
live deployment should feed equivalent extracted flow features into the same
schema. Raw packet capture is not consumed directly yet; packets must first be
converted into flow features by a future extractor.

## State Vector

The RL policy observes these simulated values:

- `edge_cpu`
- `edge_ram`
- `flow_queue`
- `processing_latency`
- `bandwidth_used`
- `rtt`
- `flow_size`

## Actions

- `0`: process locally at the edge
- `1`: offload to cloud IDS

When action `1` is selected in `edge.py`, the flow feature row is sent to the
cloud VM over a TCP socket and a result is received back.

## Reward

The reward encourages:

- successful flow inspection
- lower processing latency
- fewer flow drops
- lower edge overload
- reasonable bandwidth use
- CPU saving when the edge is busy

It penalizes unnecessary offloading when the edge is not busy.
Dropped flows receive a strong negative reward, and high queue pressure is
penalized before the queue reaches the drop limit.

## Files

- `systemModel.py`: real-flow offloading environment and reward logic
- `Training.py`: trains and saves the RL offloading policy
- `Simulation.py`: compares local-only, threshold, and RL adaptive offloading
- `edge.py`: run on VM1, loads trained edge IDS/RL policy and sends offloaded flows to cloud
- `cloud.py`: run on VM2, receives flow features and returns cloud IDS predictions
- `DQN.py`, `Double.py`, `Dueling.py`: original deep RL files kept for reference/future work

## Quick Local Replay Simulation

Train the RL policy:

```bash
python Training.py
```

Run comparison on real UNSW-NB15 feature rows:

```bash
python Simulation.py --flows 600
```

The comparison result is saved to:

```text
nids_offloading_results.csv
```

## Two-VM Demo

### 1. Copy the repo to both VMs

Both VMs should have Python 3 installed. This prototype only uses the Python
standard library.

### 2. Fresh VM setup: train files that are not stored in GitHub

Large trained IDS model files are not stored in GitHub. After pulling the repo
on a fresh VM, train the required local files before running the full IDS demo.

On the edge VM, install the IDS dependencies:

```bash
python3 -m pip install -r requirements-edge-ids.txt
```

Place the official processed UNSW-NB15 files here:

```text
ids_data/unsw_nb15/Training and Testing Sets/UNSW_NB15_training-set.csv
ids_data/unsw_nb15/Training and Testing Sets/UNSW_NB15_testing-set.csv
```

Train the edge IDS:

```bash
python3 edge_ids_train.py
```

This creates:

```text
models/edge_ids/best_edge_ids_model.joblib
models/edge_ids/label_encoder.joblib
models/edge_ids/feature_schema.json
```

If the VM is small, train with fewer rows:

```bash
python3 edge_ids_train.py --max-train-rows 50000
```

Train the RL/offloading policy:

```bash
python3 Training.py
```

This creates:

```text
offload_q_table.json
```

Retrain this policy after changing reward parameters such as drop penalties or
queue thresholds.

### 3. Start the cloud backend on VM2

On the cloud VM:

```bash
python3 cloud.py --host 0.0.0.0 --port 9000
```

Make sure VM1 can reach VM2 on port `9000`.

For a more visible demo with request logging:

```bash
python3 cloud.py --host 0.0.0.0 --port 9000 --verbose --log results/cloud_ids/cloud_requests_log.csv
```

### 4. Run the edge sender on VM1

Replace `<VM2_IP>` with the cloud VM IP address:

```bash
python3 edge.py --cloud-host <VM2_IP> --cloud-port 9000 --flows 50 --confidence-threshold 0.90
```

### 5. RL-first IDS flow behavior

After training the edge IDS, `edge.py` replays rows from the UNSW-NB15 testing
CSV as real flow feature inputs:

```bash
python3 edge.py --cloud-host <VM2_IP> --cloud-port 9000 --flows 50 --confidence-threshold 0.90
```

The order is:

1. RL/offloading policy decides first from edge resource/network state.
2. If RL chooses cloud, the flow is sent to VM2 immediately and edge IDS is skipped.
3. Cloud IDS returns its prediction, confidence, and class probabilities to the edge.
4. If RL chooses local, the lightweight edge IDS classifies the flow.
5. If the local IDS predicts an attack, the flow is escalated to cloud.
6. If the local IDS predicts `Normal` with confidence below the threshold, the flow is also escalated.
7. Only high-confidence local `Normal` results stay at the edge.
8. Edge logs both edge and cloud predictions, then records the final prediction source.

Alternative security-escalation policy:

The current demo escalates every local attack prediction to cloud. This is safe,
but it can make the cloud look like the main worker when the test traffic has
many attacks. A lighter alternative is to escalate only uncertain IDS results:

```text
Normal + confidence >= 0.90 -> keep local
Normal + confidence < 0.90  -> cloud confirmation
Attack + confidence >= 0.85 -> keep local alert
Attack + confidence < 0.85  -> cloud confirmation
```

This keeps the cloud closer to a backup/deep-confirmation role instead of
offloading every attack prediction. It can be implemented later with separate
normal and attack confidence thresholds.

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

- real flow feature replay from CSV
- simulated edge metrics
- trained RL offloading policy saved to disk
- real TCP send/receive for offloaded flows in the two-VM demo
- trained local IDS and cloud IDS processing
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

## Strong Cloud IDS Training

The cloud IDS is intended to be stronger than the edge IDS. The edge model is
kept lightweight, while the cloud model may use more CPU/RAM and slower
training.

Train the cloud neural-network IDS:

```bash
python3 -m pip install -r requirements-cloud-ids.txt
python3 cloud_ids_train.py
```

This trains a scikit-learn MLP classifier and saves the best cloud IDS artifact:

```text
models/cloud_ids/neural_net.joblib
models/cloud_ids/best_cloud_ids_model.joblib
models/cloud_ids/label_encoder.joblib
models/cloud_ids/feature_schema.json
models/cloud_ids/best_model_info.json
```

Results are saved here:

```text
results/cloud_ids/cloud_ids_model_comparison.csv
results/cloud_ids/classification_report_neural_net.csv
results/cloud_ids/confusion_matrix_neural_net.csv
```

Compare the cloud IDS against the current edge IDS using balanced accuracy /
macro recall, not only normal accuracy. The current edge Random Forest result is
around:

```text
accuracy: 0.7086
balanced accuracy / macro recall: 0.5668
macro F1: 0.4850
```

For a smaller test run:

```bash
python3 cloud_ids_train.py --max-train-rows 50000 --nn-max-iter 40
```

Tune the neural network size if the cloud VM has enough CPU/RAM:

```bash
python3 cloud_ids_train.py --nn-hidden-layers 512 256 128 --nn-max-iter 120 --nn-batch-size 1024
```

Run the cloud backend with the trained neural-network IDS:

```bash
python3 cloud.py --host 0.0.0.0 --port 9000 --verbose --log results/cloud_ids/cloud_requests_log.csv
```

To test the cloud IDS directly on a CSV:

```bash
python3 cloud_ids_predictor.py --input-csv "ids_data/unsw_nb15/Training and Testing Sets/UNSW_NB15_testing-set.csv" --output-csv results/cloud_ids/cloud_ids_predictions.csv
```

Future work:

- implement a packet-to-flow extractor, then connect live capture or pcap replay
  to that extractor
- reduce dropped flows further by tuning queue thresholds, measuring real queue
  length, and retraining RL with stronger drop-avoidance reward shaping
- optionally migrate the saved policy to DQN/Double DQN/Dueling DQN
- Currently, when the cloud backend (`cloud.py`) is started on VM2, it displays the model path and request log path. With `--verbose`, it also prints each received flow request and status.

For easier debugging and demonstration, should include:

Logging incoming requests (e.g., timestamp, source IP, flow ID).

Displaying processing status on screen (e.g., “Received flow #42, offloaded to cloud IDS”).

Returning confirmation messages to the edge client and logging them.

Optional verbose/debug mode to toggle detailed runtime information.

This will make it clearer that the system is actively handling traffic, not just passively listening.

- shape reward: tune parameter so RL learns smater policies 
