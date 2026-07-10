import argparse
import importlib.util
import json
import os
import shutil
import time

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier

from edge_ids_train import (
    DEFAULT_TEST_PATH,
    DEFAULT_TRAIN_PATH,
    TARGET_COLUMN,
    build_feature_frame,
    prepare_labels,
)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_DIR = os.path.join(BASE_DIR, "models", "cloud_ids")
DEFAULT_RESULT_DIR = os.path.join(BASE_DIR, "results", "cloud_ids")


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def optional_import(module_name):
    if importlib.util.find_spec(module_name) is None:
        return None
    return __import__(module_name)


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
            (
                "ordinal",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
            ),
        ]
    )

    transformers = []
    if numeric_features:
        transformers.append(("num", numeric_pipeline, numeric_features))
    if categorical_features:
        transformers.append(("cat", categorical_pipeline, categorical_features))

    return ColumnTransformer(transformers=transformers, remainder="drop"), numeric_features, categorical_features


def maybe_sample(df, max_rows):
    if not max_rows or len(df) <= max_rows:
        return df
    return (
        df.groupby("_target_name", group_keys=False)
        .apply(lambda group: group.sample(
            n=max(1, min(len(group), int(round(max_rows * len(group) / len(df))))),
            random_state=42,
        ))
        .sample(frac=1, random_state=42)
        .head(max_rows)
        .reset_index(drop=True)
    )


def get_class_weights(y_train):
    values, counts = np.unique(y_train, return_counts=True)
    total = len(y_train)
    n_classes = len(values)
    return {int(value): float(total / (n_classes * count)) for value, count in zip(values, counts)}


def build_models(args, num_classes, class_weights):
    models = {}
    requested = {name.strip().lower() for name in args.models.split(",") if name.strip()}

    if "neural_net" in requested or "nn" in requested or "mlp" in requested:
        models["neural_net"] = MLPClassifier(
            hidden_layer_sizes=tuple(args.nn_hidden_layers),
            activation=args.nn_activation,
            solver="adam",
            alpha=args.nn_alpha,
            batch_size=args.nn_batch_size,
            learning_rate_init=args.nn_learning_rate,
            max_iter=args.nn_max_iter,
            early_stopping=True,
            validation_fraction=args.nn_validation_fraction,
            n_iter_no_change=args.nn_patience,
            random_state=42,
            verbose=args.nn_verbose,
        )

    lightgbm = optional_import("lightgbm")
    if "lightgbm" in requested and lightgbm is not None:
        models["lightgbm"] = lightgbm.LGBMClassifier(
            objective="multiclass",
            num_class=num_classes,
            n_estimators=args.n_estimators,
            learning_rate=args.learning_rate,
            num_leaves=args.num_leaves,
            max_depth=args.max_depth,
            class_weight=class_weights,
            n_jobs=args.n_jobs,
            random_state=42,
            verbose=-1,
        )
    elif "lightgbm" in requested:
        print("Skipping LightGBM: package is not installed.")

    catboost = optional_import("catboost")
    if "catboost" in requested and catboost is not None:
        models["catboost"] = catboost.CatBoostClassifier(
            loss_function="MultiClass",
            iterations=args.n_estimators,
            learning_rate=args.learning_rate,
            depth=args.catboost_depth,
            class_weights=[class_weights[i] for i in range(num_classes)],
            random_seed=42,
            verbose=False,
            thread_count=args.n_jobs,
            allow_writing_files=False,
        )
    elif "catboost" in requested:
        print("Skipping CatBoost: package is not installed.")

    return models


def evaluate_model(model, X_test, y_test, class_names):
    y_pred = model.predict(X_test)
    if isinstance(y_pred, pd.DataFrame):
        y_pred = y_pred.values
    y_pred = np.asarray(y_pred).reshape(-1).astype(int)

    report = classification_report(
        y_test,
        y_pred,
        labels=list(range(len(class_names))),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_test, y_pred, labels=list(range(len(class_names))))

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
        "macro_f1": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "macro_precision": precision_score(y_test, y_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_test, y_pred, average="macro", zero_division=0),
        "report": report,
        "confusion_matrix": cm,
    }


def train_cloud_ids(args):
    ensure_dir(args.model_dir)
    ensure_dir(args.result_dir)

    train_df = pd.read_csv(args.train)
    test_df = pd.read_csv(args.test)
    train_df["_target_name"] = prepare_labels(train_df)
    test_df["_target_name"] = prepare_labels(test_df)
    train_df = maybe_sample(train_df, args.max_train_rows)

    y_train_names = train_df["_target_name"]
    y_test_names = test_df["_target_name"]
    X_train = build_feature_frame(train_df.drop(columns=["_target_name"]))
    X_test = build_feature_frame(test_df.drop(columns=["_target_name"]), feature_columns=list(X_train.columns))

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train_names)
    y_test = label_encoder.transform(y_test_names)
    class_names = list(label_encoder.classes_)
    class_weights = get_class_weights(y_train)

    preprocessor, numeric_features, categorical_features = build_preprocessor(X_train)
    models = build_models(args, len(class_names), class_weights)
    if not models:
        raise RuntimeError(
            "No cloud IDS models are available. Install at least one package:\n"
            "  pip install -r requirements-cloud-ids.txt"
        )

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

    print("Cloud IDS training rows: {}, test rows: {}, features: {}".format(len(X_train), len(X_test), X_train.shape[1]))
    print("Classes: {}".format(", ".join(class_names)))

    results = []
    best_model_name = None
    best_score = -1.0
    best_model_path = None

    for name, classifier in models.items():
        print("\nTraining cloud {}...".format(name))
        start = time.time()
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", classifier),
            ]
        )
        fit_kwargs = {}
        if name == "neural_net" and args.nn_class_weight:
            sample_weight = np.array([class_weights[int(label)] for label in y_train], dtype=float)
            fit_kwargs["classifier__sample_weight"] = sample_weight
        pipeline.fit(X_train, y_train, **fit_kwargs)
        train_seconds = time.time() - start

        pred_start = time.time()
        metrics = evaluate_model(pipeline, X_test, y_test, class_names)
        predict_seconds = time.time() - pred_start

        model_path = os.path.join(args.model_dir, "{}.joblib".format(name))
        joblib.dump(pipeline, model_path)

        pd.DataFrame(metrics["report"]).transpose().to_csv(
            os.path.join(args.result_dir, "classification_report_{}.csv".format(name))
        )
        pd.DataFrame(
            metrics["confusion_matrix"],
            index=class_names,
            columns=class_names,
        ).to_csv(os.path.join(args.result_dir, "confusion_matrix_{}.csv".format(name)))

        row = {
            "model": name,
            "accuracy": metrics["accuracy"],
            "balanced_accuracy": metrics["balanced_accuracy"],
            "macro_f1": metrics["macro_f1"],
            "weighted_f1": metrics["weighted_f1"],
            "macro_precision": metrics["macro_precision"],
            "macro_recall": metrics["macro_recall"],
            "train_seconds": train_seconds,
            "predict_seconds": predict_seconds,
            "model_path": model_path,
        }
        results.append(row)
        print(
            "{model}: balanced_accuracy={balanced_accuracy:.4f}, macro_f1={macro_f1:.4f}, "
            "accuracy={accuracy:.4f}, train_time={train_seconds:.1f}s".format(**row)
        )

        if metrics["balanced_accuracy"] > best_score:
            best_score = metrics["balanced_accuracy"]
            best_model_name = name
            best_model_path = model_path

    results_df = pd.DataFrame(results).sort_values(
        by=["balanced_accuracy", "macro_f1", "accuracy"],
        ascending=[False, False, False],
    )
    results_path = os.path.join(args.result_dir, "cloud_ids_model_comparison.csv")
    results_df.to_csv(results_path, index=False)

    best_dst = os.path.join(args.model_dir, "best_cloud_ids_model.joblib")
    shutil.copy2(best_model_path, best_dst)
    best_info = {
        "best_model": best_model_name,
        "selection_metric": "balanced_accuracy",
        "best_model_path": best_dst,
        "all_models": results,
    }
    with open(os.path.join(args.model_dir, "best_model_info.json"), "w", encoding="utf-8") as f:
        json.dump(best_info, f, indent=2)

    print("\nCloud model comparison saved to: {}".format(results_path))
    print("Best cloud IDS model: {} -> {}".format(best_model_name, best_dst))
    print(results_df.to_string(index=False))


def parse_args():
    parser = argparse.ArgumentParser(description="Train cloud IDS models on UNSW-NB15.")
    parser.add_argument("--train", default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--test", default=DEFAULT_TEST_PATH)
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--result-dir", default=DEFAULT_RESULT_DIR)
    parser.add_argument("--max-train-rows", type=int, default=None)
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--num-leaves", type=int, default=96)
    parser.add_argument("--max-depth", type=int, default=-1)
    parser.add_argument("--catboost-depth", type=int, default=8)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument(
        "--nn-hidden-layers",
        type=int,
        nargs="+",
        default=[256, 128, 64],
        help="Hidden layer sizes for the cloud neural network.",
    )
    parser.add_argument("--nn-activation", default="relu", choices=["identity", "logistic", "tanh", "relu"])
    parser.add_argument("--nn-alpha", type=float, default=0.0001)
    parser.add_argument("--nn-batch-size", type=int, default=512)
    parser.add_argument("--nn-learning-rate", type=float, default=0.001)
    parser.add_argument("--nn-max-iter", type=int, default=80)
    parser.add_argument("--nn-validation-fraction", type=float, default=0.1)
    parser.add_argument("--nn-patience", type=int, default=8)
    parser.add_argument(
        "--no-nn-class-weight",
        dest="nn_class_weight",
        action="store_false",
        help="Disable balanced sample weights for neural network training.",
    )
    parser.set_defaults(nn_class_weight=True)
    parser.add_argument("--nn-verbose", action="store_true")
    parser.add_argument(
        "--models",
        default="neural_net,lightgbm,catboost",
        help="Comma-separated cloud models to train: neural_net,lightgbm,catboost",
    )
    return parser.parse_args()


if __name__ == "__main__":
    train_cloud_ids(parse_args())
