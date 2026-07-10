# Adaptive Offloading for NIDS

This repository contains a prototype **adaptive edge-cloud Network Intrusion
Detection System (NIDS)**. The system uses a reinforcement learning offloading
policy to decide whether each network flow should be:

- processed locally by the edge IDS,
- offloaded to the cloud IDS, or
- dropped when the simulated edge environment is overloaded.

The current implementation is **flow-based**. Raw packets are not processed
directly yet. A future packet-to-flow extractor must convert live packets or
pcap traffic into the same flow feature schema used by the IDS models.

## Project Overview

The project is designed for a two-VM demo:

- **Edge VM**
  - Replays flow feature rows from UNSW-NB15.
  - Loads the RL offloading policy.
  - Loads lightweight edge IDS models.
  - Decides whether to process locally or offload to cloud.
  - Sends offloaded flow features to the cloud VM over TCP.
  - Records the final IDS result.

- **Cloud VM**
  - Receives offloaded flow feature rows from the edge VM.
  - Loads the trained cloud IDS model.
  - Returns prediction, confidence, and class probabilities to the edge.

The edge remains the main controller. Even when a flow is classified by the
cloud, the final result is sent back and logged on the edge side.

## Repository Structure

```text
.
|-- Training.py                  # Train RL offloading policy
|-- Simulation.py                # Compare local-only, threshold, and RL modes
|-- systemModel.py               # Flow offloading environment and reward logic
|-- edge.py                      # Edge VM runtime: RL + edge IDS + cloud offload
|-- cloud.py                     # Cloud VM runtime: receive flow and classify
|-- edge_ids_train.py            # Train edge IDS models
|-- edge_ids_predictor.py        # Run edge IDS prediction on CSV input
|-- cloud_ids_train.py           # Train cloud IDS models
|-- cloud_ids_predictor.py       # Run cloud IDS prediction on CSV input
|-- DQN.py                       # Original deep RL file kept for future work
|-- Double.py                    # Original Double DQN file kept for future work
|-- Dueling.py                   # Original Dueling DQN file kept for future work
|-- ids_data/                    # IDS datasets
|-- data/                        # Original/offloading simulation artifacts
|-- models/                      # Saved trained models and metadata
|-- results/                     # Evaluation outputs and runtime logs
|-- requirements-edge-ids.txt    # Edge IDS dependencies
`-- requirements-cloud-ids.txt   # Cloud IDS dependencies
```

## Dataset

Use the official processed UNSW-NB15 training/testing split:

```text
ids_data/unsw_nb15/Training and Testing Sets/UNSW_NB15_training-set.csv
ids_data/unsw_nb15/Training and Testing Sets/UNSW_NB15_testing-set.csv
```

The processed split already includes the `attack_cat` column used for
multi-class labels.

The `data/` folder is kept for offloading simulation artifacts. IDS
datasets are kept under `ids_data/`.

## RL Offloading Policy

### State Vector

The RL policy currently observes simulated values:

- `edge_cpu`
- `edge_ram`
- `flow_queue`
- `processing_latency`
- `bandwidth_used`
- `rtt`
- `flow_size`

`flow_size` is useful for estimating processing/transmission cost, but it may
need to be removed later if the offloading decision must be made before reading
traffic information.

### Actions

- `0`: process locally at the edge
- `1`: offload to cloud IDS
- drop action inside the environment when the queue is overloaded

When offloading is selected in `edge.py`, the flow feature row is sent to the
cloud VM over TCP and the cloud IDS result is received back.

### Reward

The reward encourages:

- successful flow inspection,
- lower processing latency,
- fewer dropped flows,
- lower edge overload,
- reasonable bandwidth usage,
- CPU saving when the edge is busy.

It penalizes:

- unnecessary offloading when the edge is not busy,
- high queue pressure,
- dropped flows..

## Installation

Install Python 3.9 or newer on both VMs.

Clone the repository:

```bash
git clone https://github.com/nguyentuancuong20214950/offloading-for-nids.git
cd offloading-for-nids
```

Optional but recommended: create a virtual environment.

```bash
python3 -m venv .venv
source .venv/bin/activate
```

On Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Install dependencies according to the VM role:

```bash
python3 -m pip install -r requirements-edge-ids.txt
python3 -m pip install -r requirements-cloud-ids.txt
```

On a VM that will only run the edge, `requirements-edge-ids.txt` is enough.
On a VM that will train/run cloud IDS, install `requirements-cloud-ids.txt`.

## Training

Large trained model files are not stored in GitHub. On 2 VMs, train the
required model files before running the full demo.

### Train Edge IDS Models

```bash
python3 edge_ids_train.py
```

This trains:

```text
models/edge_ids/random_forest.joblib
models/edge_ids/extra_trees.joblib
models/edge_ids/logistic_regression.joblib
models/edge_ids/best_edge_ids_model.joblib
```

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

For a smaller machine, train with fewer rows:

```bash
python3 edge_ids_train.py --max-train-rows 50000
```

### Train Cloud IDS Models

The cloud IDS is intended to be stronger than the edge IDS. The edge model is
kept lightweight, while the cloud side may use more CPU/RAM and slower training.

Train all cloud models:

```bash
python3 cloud_ids_train.py
```

This trains and compares:

- CatBoost,
- LightGBM,
- Neural Network.

The best cloud model is selected by balanced accuracy and saved to:

```text
models/cloud_ids/best_cloud_ids_model.joblib
models/cloud_ids/label_encoder.joblib
models/cloud_ids/feature_schema.json
models/cloud_ids/best_model_info.json
```

Individual model artifacts may include:

```text
models/cloud_ids/catboost.joblib
models/cloud_ids/lightgbm.joblib
models/cloud_ids/neural_net.joblib
```

Results are saved here:

```text
results/cloud_ids/cloud_ids_model_comparison.csv
results/cloud_ids/classification_report_catboost.csv
results/cloud_ids/classification_report_lightgbm.csv
results/cloud_ids/classification_report_neural_net.csv
results/cloud_ids/confusion_matrix_catboost.png
results/cloud_ids/confusion_matrix_lightgbm.png
results/cloud_ids/confusion_matrix_neural_net.png
```

Train only one model, for example, CatBoost:

```bash
python3 cloud_ids_train.py --models catboost
```

For a smaller VM:

```bash
python3 cloud_ids_train.py --max-train-rows 50000 --nn-max-iter 40
```

Tune the neural network if the cloud VM has enough CPU/RAM:

```bash
python3 cloud_ids_train.py --nn-hidden-layers 512 256 128 --nn-max-iter 120 --nn-batch-size 1024
```

Note: some scikit-learn versions do not support `sample_weight` for
`MLPClassifier.fit()`. If neural network training fails with a `sample_weight`
error, train only CatBoost and LightGBM or update the training code to skip sample weights
for the neural network.

### Train RL Offloading Policy

```bash
python3 Training.py
```

This creates:

```text
offload_q_table.json
```

## Quick Local Replay Simulation

Run a local comparison on UNSW-NB15 flow feature rows:

```bash
python3 Simulation.py --flows 600
```

The comparison result is saved to:

```text
nids_offloading_results.csv
```

This simulation compares local-only, simple threshold, and RL adaptive offloading
behavior.

## Two-VM Demo

### VM Network Setup

Recommended setup:

- Adapter 1: NAT, for internet and package installation.
- Adapter 2: Host-only Adapter or Internal Network, for edge-cloud traffic.

Example static IPs:

```text
Cloud VM: 192.168.56.10
Edge VM:  192.168.56.11
```

Make sure the edge VM can reach the cloud VM on the selected port:

```bash
ping 192.168.56.10
```

### Start the Cloud Backend on VM2

```bash
python3 cloud.py --host 0.0.0.0 --port 9000
```

For a more visible demo with request logging:

```bash
python3 cloud.py --host 0.0.0.0 --port 9000 --verbose --log results/cloud_ids/cloud_requests_log.csv
```

Cloud-side request logs are saved to:

```text
results/cloud_ids/cloud_requests_log.csv
```

### Run the Edge on VM1

Replace `<CLOUD_VM_IP>` with the cloud VM IP address:

```bash
python3 edge.py --cloud-host <CLOUD_VM_IP> --cloud-port 9000 --flows 50 --confidence-threshold 0.90
```

Example:

```bash
python3 edge.py --cloud-host 192.168.56.10 --cloud-port 9000 --flows 50 --confidence-threshold 0.90
```

Edge-side demo logs are saved to:

```text
results/edge_ids/edge_ids_offloading_demo.csv
```

## Future Work

- Implement a packet-to-flow extractor and connect live capture or pcap replay.
- Replace simulated edge metrics with real device's measurements.
- Reconsider whether `flow_size` should remain in the RL state if offloading
  must be decided before reading traffic information.
- Reduce dropped flows further by tuning queue thresholds, measuring real queue
  length, and retraining RL with stronger drop-avoidance reward shaping.
- Improve verbose runtime logging with timestamp, source IP, flow ID, selected
  action, prediction, confidence, and final result.
- Add stronger communication handling for invalid messages, timeouts, and cloud
  unavailability.
- Add authentication/encryption for edge-cloud communication.
- Optionally migrate the saved policy to DQN, Double DQN, or Dueling DQN.
