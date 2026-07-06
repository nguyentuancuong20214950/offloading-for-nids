import argparse
import json
import os

import joblib
import numpy as np
import pandas as pd


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_DIR = os.path.join(BASE_DIR, "models", "cloud_ids")
DEFAULT_MODEL_PATH = os.path.join(DEFAULT_MODEL_DIR, "best_cloud_ids_model.joblib")
DEFAULT_LABEL_ENCODER_PATH = os.path.join(DEFAULT_MODEL_DIR, "label_encoder.joblib")
DEFAULT_SCHEMA_PATH = os.path.join(DEFAULT_MODEL_DIR, "feature_schema.json")


class CloudIDSPredictor:
    def __init__(
        self,
        model_path=DEFAULT_MODEL_PATH,
        label_encoder_path=DEFAULT_LABEL_ENCODER_PATH,
        schema_path=DEFAULT_SCHEMA_PATH,
    ):
        self.model_path = model_path
        self.label_encoder_path = label_encoder_path
        self.schema_path = schema_path
        self.model = joblib.load(model_path)
        self.label_encoder = joblib.load(label_encoder_path)
        with open(schema_path, "r", encoding="utf-8") as f:
            self.schema = json.load(f)
        self.feature_columns = self.schema["feature_columns"]

    def prepare_frame(self, flow):
        if isinstance(flow, dict):
            df = pd.DataFrame([flow])
        elif isinstance(flow, pd.Series):
            df = pd.DataFrame([flow.to_dict()])
        elif isinstance(flow, pd.DataFrame):
            df = flow.copy()
        else:
            raise TypeError("flow must be a dict, pandas Series, or pandas DataFrame")

        missing = [col for col in self.feature_columns if col not in df.columns]
        if missing:
            raise ValueError(
                "Runtime flow is missing required cloud IDS feature columns: {}. "
                "The live feature extractor must match the training schema.".format(missing)
            )

        return df[self.feature_columns]

    def predict(self, flow):
        df = self.prepare_frame(flow)
        pred_ids = self.model.predict(df)
        pred_ids = np.asarray(pred_ids).reshape(-1).astype(int)
        labels = self.label_encoder.inverse_transform(pred_ids)

        probabilities = None
        if hasattr(self.model, "predict_proba"):
            probabilities = self.model.predict_proba(df)

        results = []
        for index, label in enumerate(labels):
            item = {"attack_category": label}
            if probabilities is not None:
                class_probs = {
                    class_name: float(prob)
                    for class_name, prob in zip(self.label_encoder.classes_, probabilities[index])
                }
                item["confidence"] = float(max(class_probs.values()))
                item["class_probabilities"] = class_probs
            results.append(item)

        return results[0] if len(results) == 1 else results


def parse_args():
    parser = argparse.ArgumentParser(description="Run the trained cloud IDS model on extracted flow features.")
    parser.add_argument("--input-csv", required=True, help="CSV containing extracted flow features.")
    parser.add_argument("--output-csv", default=None, help="Optional path for prediction output CSV.")
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH, help="Path to a saved cloud IDS model.")
    parser.add_argument("--label-encoder", default=DEFAULT_LABEL_ENCODER_PATH)
    parser.add_argument("--schema", default=DEFAULT_SCHEMA_PATH)
    return parser.parse_args()


def main():
    args = parse_args()
    predictor = CloudIDSPredictor(args.model, args.label_encoder, args.schema)
    df = pd.read_csv(args.input_csv)
    predictions = predictor.predict(df)
    if isinstance(predictions, dict):
        predictions = [predictions]
    out_df = pd.concat([df.reset_index(drop=True), pd.DataFrame(predictions)], axis=1)

    if args.output_csv:
        out_df.to_csv(args.output_csv, index=False)
        print("Predictions saved to: {}".format(args.output_csv))
    else:
        print(out_df.to_string(index=False))


if __name__ == "__main__":
    main()
