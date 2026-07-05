import argparse
import json
import os
import time

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("MPLCONFIGDIR", os.path.join(BASE_DIR, ".matplotlib_cache"))
DEFAULT_TRAIN_PATH = os.path.join(
    BASE_DIR,
    "ids_data",
    "unsw_nb15",
    "Training and Testing Sets",
    "UNSW_NB15_training-set.csv",
)
DEFAULT_TEST_PATH = os.path.join(
    BASE_DIR,
    "ids_data",
    "unsw_nb15",
    "Training and Testing Sets",
    "UNSW_NB15_testing-set.csv",
)
DEFAULT_MODEL_DIR = os.path.join(BASE_DIR, "models", "edge_ids")
DEFAULT_RESULT_DIR = os.path.join(BASE_DIR, "results", "edge_ids")

TARGET_COLUMN = "attack_cat"
BINARY_LABEL_COLUMN = "label"
DROP_COLUMNS = {"id", TARGET_COLUMN, BINARY_LABEL_COLUMN}


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def make_one_hot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


def normalize_attack_name(value):
    if pd.isna(value):
        return "Normal"

    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "-", "normal"}:
        return "Normal"

    key = text.lower().replace("_", " ").replace("-", " ")
    mapping = {
        "analysis": "Analysis",
        "backdoor": "Backdoor",
        "backdoors": "Backdoor",
        "dos": "DoS",
        "exploits": "Exploits",
        "fuzzers": "Fuzzers",
        "generic": "Generic",
        "reconnaissance": "Reconnaissance",
        "shellcode": "Shellcode",
        "worms": "Worms",
    }
    return mapping.get(key, text)


def load_unsw_csv(path):
    if not os.path.exists(path):
        raise FileNotFoundError(
            "Dataset file not found: {}\n"
            "Expected the official processed UNSW-NB15 CSV files by default:\n"
            "  ids_data/unsw_nb15/Training and Testing Sets/UNSW_NB15_training-set.csv\n"
            "  ids_data/unsw_nb15/Training and Testing Sets/UNSW_NB15_testing-set.csv".format(path)
        )
    return pd.read_csv(path)


def prepare_labels(df):
    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            "Column '{}' was not found. Use the official processed UNSW-NB15 "
            "training/testing CSV files, not the raw UNSW-NB15_1.csv files.".format(TARGET_COLUMN)
        )

    labels = df[TARGET_COLUMN].apply(normalize_attack_name)

    if BINARY_LABEL_COLUMN in df.columns:
        normal_mask = pd.to_numeric(df[BINARY_LABEL_COLUMN], errors="coerce").fillna(1).astype(int) == 0
        labels.loc[normal_mask] = "Normal"

    return labels


def build_feature_frame(df, feature_columns=None):
    drop_cols = [col for col in DROP_COLUMNS if col in df.columns]
    X = df.drop(columns=drop_cols)
    X = X.replace([np.inf, -np.inf], np.nan)

    if feature_columns is not None:
        missing = [col for col in feature_columns if col not in X.columns]
        if missing:
            raise ValueError("Input data is missing required feature columns: {}".format(missing))
        X = X[feature_columns]

    return X


def maybe_sample(df, max_rows, label_column):
    if not max_rows or len(df) <= max_rows:
        return df

    pieces = []
    grouped = df.groupby(label_column, group_keys=False)
    for _, group in grouped:
        take = max(1, int(round(max_rows * len(group) / len(df))))
        pieces.append(group.sample(n=min(take, len(group)), random_state=42))

    sampled = pd.concat(pieces).sample(frac=1, random_state=42).reset_index(drop=True)
    if len(sampled) > max_rows:
        sampled = sampled.sample(n=max_rows, random_state=42).reset_index(drop=True)
    return sampled


def build_preprocessor(X):
    categorical_features = [
        col for col in X.columns
        if X[col].dtype == "object" or col.lower() in {"proto", "service", "state"}
    ]
    numeric_features = [col for col in X.columns if col not in categorical_features]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", make_one_hot_encoder()),
        ]
    )

    transformers = []
    if numeric_features:
        transformers.append(("num", numeric_pipeline, numeric_features))
    if categorical_features:
        transformers.append(("cat", categorical_pipeline, categorical_features))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    return preprocessor, numeric_features, categorical_features


def get_models():
    return {
        "random_forest": RandomForestClassifier(
            n_estimators=120,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=42,
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=160,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42,
        ),
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            solver="saga",
            n_jobs=-1,
            random_state=42,
        ),
    }


def save_confusion_matrix(cm, class_names, title, png_path, csv_path):
    pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(csv_path)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig_size = max(8, min(15, len(class_names) * 1.2))
        fig, ax = plt.subplots(figsize=(fig_size, fig_size))
        image = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        ax.figure.colorbar(image, ax=ax)
        ax.set(
            xticks=np.arange(len(class_names)),
            yticks=np.arange(len(class_names)),
            xticklabels=class_names,
            yticklabels=class_names,
            title=title,
            ylabel="True label",
            xlabel="Predicted label",
        )
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        threshold = cm.max() / 2.0 if cm.size else 0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(
                    j,
                    i,
                    format(cm[i, j], "d"),
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > threshold else "black",
                    fontsize=8,
                )

        fig.tight_layout()
        fig.savefig(png_path, dpi=200)
        plt.close(fig)
    except Exception as exc:
        print("Could not save confusion matrix PNG for {}: {}".format(title, exc))


def train_and_evaluate(args):
    ensure_dir(args.model_dir)
    ensure_dir(args.result_dir)

    print("Loading UNSW-NB15 data...")
    train_df = load_unsw_csv(args.train)
    test_df = load_unsw_csv(args.test)

    train_df = train_df.copy()
    test_df = test_df.copy()
    train_df["_target_name"] = prepare_labels(train_df)
    test_df["_target_name"] = prepare_labels(test_df)

    if args.max_train_rows:
        train_df = maybe_sample(train_df, args.max_train_rows, "_target_name")

    y_train_names = train_df["_target_name"]
    y_test_names = test_df["_target_name"]

    X_train = build_feature_frame(train_df.drop(columns=["_target_name"]))
    X_test = build_feature_frame(test_df.drop(columns=["_target_name"]), feature_columns=list(X_train.columns))

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train_names)
    y_test = label_encoder.transform(y_test_names)
    class_names = list(label_encoder.classes_)

    print("Classes: {}".format(", ".join(class_names)))
    print("Training rows: {}, test rows: {}, features: {}".format(len(X_train), len(X_test), X_train.shape[1]))

    preprocessor, numeric_features, categorical_features = build_preprocessor(X_train)
    model_defs = get_models()
    results = []

    joblib.dump(label_encoder, os.path.join(args.model_dir, "label_encoder.joblib"))

    schema = {
        "target_column": TARGET_COLUMN,
        "feature_columns": list(X_train.columns),
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "class_names": class_names,
        "train_path": os.path.abspath(args.train),
        "test_path": os.path.abspath(args.test),
    }
    with open(os.path.join(args.model_dir, "feature_schema.json"), "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2)

    for model_name, classifier in model_defs.items():
        print("\nTraining {}...".format(model_name))
        start = time.time()

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", classifier),
            ]
        )
        pipeline.fit(X_train, y_train)

        train_seconds = time.time() - start
        pred_start = time.time()
        y_pred = pipeline.predict(X_test)
        predict_seconds = time.time() - pred_start

        accuracy = accuracy_score(y_test, y_pred)
        macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
        weighted_f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        macro_precision = precision_score(y_test, y_pred, average="macro", zero_division=0)
        macro_recall = recall_score(y_test, y_pred, average="macro", zero_division=0)

        model_path = os.path.join(args.model_dir, "{}.joblib".format(model_name))
        joblib.dump(pipeline, model_path)

        report_dict = classification_report(
            y_test,
            y_pred,
            labels=list(range(len(class_names))),
            target_names=class_names,
            output_dict=True,
            zero_division=0,
        )
        report_text = classification_report(
            y_test,
            y_pred,
            labels=list(range(len(class_names))),
            target_names=class_names,
            zero_division=0,
        )

        report_csv_path = os.path.join(args.result_dir, "classification_report_{}.csv".format(model_name))
        report_txt_path = os.path.join(args.result_dir, "classification_report_{}.txt".format(model_name))
        pd.DataFrame(report_dict).transpose().to_csv(report_csv_path)
        with open(report_txt_path, "w", encoding="utf-8") as f:
            f.write(report_text)

        cm = confusion_matrix(y_test, y_pred, labels=list(range(len(class_names))))
        save_confusion_matrix(
            cm,
            class_names,
            "{} confusion matrix".format(model_name),
            os.path.join(args.result_dir, "confusion_matrix_{}.png".format(model_name)),
            os.path.join(args.result_dir, "confusion_matrix_{}.csv".format(model_name)),
        )

        results.append(
            {
                "model": model_name,
                "accuracy": accuracy,
                "macro_f1": macro_f1,
                "weighted_f1": weighted_f1,
                "macro_precision": macro_precision,
                "macro_recall": macro_recall,
                "train_seconds": train_seconds,
                "predict_seconds": predict_seconds,
                "model_path": model_path,
            }
        )

        print(
            "{} done: accuracy={:.4f}, macro_f1={:.4f}, weighted_f1={:.4f}, train_time={:.1f}s".format(
                model_name,
                accuracy,
                macro_f1,
                weighted_f1,
                train_seconds,
            )
        )

    results_df = pd.DataFrame(results).sort_values(
        by=["macro_f1", "weighted_f1", "accuracy"],
        ascending=[False, False, False],
    )
    results_path = os.path.join(args.result_dir, "edge_ids_model_comparison.csv")
    results_df.to_csv(results_path, index=False)

    best_row = results_df.iloc[0].to_dict()
    best_model_name = best_row["model"]
    best_model_src = best_row["model_path"]
    best_model_dst = os.path.join(args.model_dir, "best_edge_ids_model.joblib")
    joblib.dump(joblib.load(best_model_src), best_model_dst)

    best_info = {
        "best_model": best_model_name,
        "selection_metric": "macro_f1",
        "best_model_path": best_model_dst,
        "all_models": results,
    }
    with open(os.path.join(args.model_dir, "best_model_info.json"), "w", encoding="utf-8") as f:
        json.dump(best_info, f, indent=2)

    print("\nModel comparison saved to: {}".format(results_path))
    print("All models saved in: {}".format(args.model_dir))
    print("Best edge IDS model: {} -> {}".format(best_model_name, best_model_dst))


def parse_args():
    parser = argparse.ArgumentParser(description="Train lightweight multi-class edge IDS models on UNSW-NB15.")
    parser.add_argument("--train", default=DEFAULT_TRAIN_PATH, help="Path to UNSW_NB15_training-set.csv")
    parser.add_argument("--test", default=DEFAULT_TEST_PATH, help="Path to UNSW_NB15_testing-set.csv")
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR, help="Directory for saved IDS models")
    parser.add_argument("--result-dir", default=DEFAULT_RESULT_DIR, help="Directory for reports and confusion matrices")
    parser.add_argument(
        "--max-train-rows",
        type=int,
        default=None,
        help="Optional stratified training sample size for very small edge machines.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    train_and_evaluate(parse_args())
