from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = PROJECT_ROOT / "models"

DATA_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

DATA_URL = "https://raw.githubusercontent.com/minakdr/Car-price-prediction-using-Stacking-ensemble-modeling-technique/main/car%20data.csv"


def load_data() -> pd.DataFrame:
    """Load car data from local file, online URL, or sample file."""
    full_data_path = DATA_DIR / "car_data.csv"
    sample_data_path = DATA_DIR / "car_data_sample.csv"

    if full_data_path.exists():
        return pd.read_csv(full_data_path)

    try:
        df = pd.read_csv(DATA_URL)
        df.to_csv(full_data_path, index=False)
        print(f"Downloaded full dataset to {full_data_path}")
        return df
    except Exception:
        print("Could not download full dataset. Using sample dataset.")
        return pd.read_csv(sample_data_path)


def prepare_data(df: pd.DataFrame):
    """Clean and prepare features for modelling."""
    df = df.copy()
    df.columns = df.columns.str.strip()

    if "Car_Name" in df.columns:
        df = df.drop(columns=["Car_Name"])

    df["Car_Age"] = 2025 - df["Year"]
    df = df.drop(columns=["Year"])

    X = df.drop(columns=["Selling_Price"])
    y = df["Selling_Price"]

    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()
    numeric_features = X.select_dtypes(exclude=["object"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_features),
        ]
    )

    return X, y, preprocessor


def evaluate_model(model_name: str, y_true, y_pred) -> dict:
    """Calculate common regression metrics."""
    return {
        "model": model_name,
        "r2_score": r2_score(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
    }


def main():
    df = load_data()
    print("Dataset shape:", df.shape)

    X, y, preprocessor = prepare_data(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42),
    }

    results = []
    predictions = pd.DataFrame({"actual": y_test.reset_index(drop=True)})

    fitted_models = {}

    for name, model in models.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model),
            ]
        )

        pipeline.fit(X_train, y_train)
        pred = pipeline.predict(X_test)

        results.append(evaluate_model(name, y_test, pred))
        predictions[name] = pred
        fitted_models[name] = pipeline

    metrics_df = pd.DataFrame(results).sort_values("r2_score", ascending=False)
    metrics_df.to_csv(OUTPUTS_DIR / "model_metrics.csv", index=False)
    predictions.to_csv(OUTPUTS_DIR / "predictions.csv", index=False)

    best_model_name = metrics_df.iloc[0]["model"]
    best_model = fitted_models[best_model_name]
    joblib.dump(best_model, MODELS_DIR / "car_sales_model.pkl")

    # Actual vs predicted plot for best model
    best_pred = predictions[best_model_name]
    plt.figure(figsize=(7, 5))
    plt.scatter(predictions["actual"], best_pred)
    plt.xlabel("Actual Selling Price")
    plt.ylabel("Predicted Selling Price")
    plt.title(f"Actual vs Predicted - {best_model_name}")
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / "actual_vs_predicted.png", dpi=150)
    plt.close()

    print("\nModel results:")
    print(metrics_df)
    print(f"\nBest model saved to: {MODELS_DIR / 'car_sales_model.pkl'}")
    print(f"Outputs saved to: {OUTPUTS_DIR}")


if __name__ == "__main__":
    main()
