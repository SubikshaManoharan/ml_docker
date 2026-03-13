import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pickle
from pathlib import Path
import mlflow
import mlflow.sklearn


def load_and_prepare(csv_path: str):
    df = pd.read_csv(csv_path)
    # Map CSV columns to the feature names expected by the app
    mapping = {
        'radius_mean': 'mean_radius',
        'texture_mean': 'mean_texture',
        'perimeter_mean': 'mean_perimeter',
        'area_mean': 'mean_area',
        'smoothness_mean': 'mean_smoothness',
        'compactness_mean': 'mean_compactness',
        'concavity_mean': 'mean_concavity',
        'concave points_mean': 'mean_concave_points',
        'symmetry_mean': 'mean_symmetry',
        'fractal_dimension_mean': 'mean_fractal_dimension',

        'radius_se': 'radius_error',
        'texture_se': 'texture_error',
        'perimeter_se': 'perimeter_error',
        'area_se': 'area_error',
        'smoothness_se': 'smoothness_error',
        'compactness_se': 'compactness_error',
        'concavity_se': 'concavity_error',
        'concave points_se': 'concave_points_error',
        'symmetry_se': 'symmetry_error',
        'fractal_dimension_se': 'fractal_dimension_error',

        'radius_worst': 'worst_radius',
        'texture_worst': 'worst_texture'
    }

    # Some CSV column names contain spaces like 'concave points_mean'
    # Ensure all required columns exist
    required_cols = list(mapping.keys())
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing expected columns in CSV: {missing}")

    X = df[required_cols].copy()
    # rename to the app's names (not strictly necessary for training ordering,
    # but keeps things clear)
    X = X.rename(columns=mapping)

    # Target: diagnosis (M/B) -> 1/0
    if 'diagnosis' not in df.columns:
        raise RuntimeError("CSV must contain a 'diagnosis' column with 'M'/'B' values")
    y = df['diagnosis'].map({'B': 0, 'M': 1}).astype(int)

    return X.values, y.values


def train_and_save(csv_path: str, out_path: str):
    X, y = load_and_prepare(csv_path)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # MLflow experiment tracking
    mlflow.set_experiment("breast-cancer-prediction")
    
    with mlflow.start_run(run_name="random_forest_model"):
        # Model hyperparameters
        n_estimators = 200
        random_state = 42
        test_size = 0.2
        
        # Log parameters
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("random_state", random_state)
        mlflow.log_param("test_size", test_size)
        mlflow.log_param("model_type", "RandomForestClassifier")
        mlflow.log_param("scaler", "StandardScaler")
        
        # Simple pipeline: scaler + random forest
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', RandomForestClassifier(n_estimators=n_estimators, random_state=random_state))
        ])

        pipe.fit(X_train, y_train)
        
        # Make predictions
        y_pred = pipe.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        
        print(f"Validation accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")

        # Save the trained pipeline with pickle
        out = Path(out_path)
        with out.open('wb') as f:
            pickle.dump(pipe, f)

        print(f"Saved model to: {out.resolve()}")
        
        # Log model to MLflow
        mlflow.sklearn.log_model(pipe, "model")
        
        # Log the pickle file as artifact
        mlflow.log_artifact(str(out))
        
        print(f"MLflow Run ID: {mlflow.active_run().info.run_id}")


if __name__ == '__main__':
    csv = Path(__file__).parent / 'breast_cancer.csv'
    out = Path(__file__).parent / 'brest_cancer.pkl'
    train_and_save(str(csv), str(out))
