
import pandas as pd
import numpy as np
import os
import logging
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    r2_score, mean_absolute_error, mean_squared_error
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from xgboost import XGBClassifier, XGBRegressor
import optuna
import pickle
import io
from typing import Dict, Tuple, Any, Optional, List

logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self, df: pd.DataFrame, target: str, test_size: float = 0.25, 
                 random_state: int = 42, output_dir: str = "model_outputs"):
        self.df = df.copy()  # Work with a copy to avoid modifying original data
        self.target = target
        self.test_size = test_size
        self.random_state = random_state
        self.output_dir = output_dir
        self.target_type = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}
        self.results = []
        self.best_model_name = None
        self.best_model = None
        os.makedirs(self.output_dir, exist_ok=True)

    def _detect_task_type(self):
        """Detect if the problem is classification or regression."""
        try:
            y = self.df[self.target]
            if y.nunique() < 15 and y.dtype != "float":
                self.target_type = "classification"
            else:
                self.target_type = "regression"
            logger.info(f"Detected task type: {self.target_type}")
        except Exception as e:
            logger.error(f"Error in _detect_task_type: {e}")
            raise

    def _preprocess_data(self):
        """Preprocess the data: handle missing values and encode categoricals."""
        try:
            # Drop rows with missing values in the target column
            initial_shape = self.df.shape
            self.df = self.df.dropna(subset=[self.target])
            logger.info(f"Dropped {initial_shape[0] - self.df.shape[0]} rows with missing target values")
            
            # For simplicity, we drop rows with any missing value. 
            # In a more advanced version, we could impute.
            self.df = self.df.dropna(axis=0, how='any')
            logger.info(f"Final dataset shape after dropping NA rows: {self.df.shape}")
            
            if self.df.empty:
                raise ValueError("No data remaining after preprocessing")
            
            # Separate features and target
            X = self.df.drop(columns=[self.target])
            y = self.df[self.target]
            
            # Encode categorical variables
            X = pd.get_dummies(X, drop_first=True)
            
            # Split the data
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=self.test_size, random_state=self.random_state
            )
            
            logger.info(f"Data preprocessed. Train shape: {self.X_train.shape}, Test shape: {self.X_test.shape}")
        except Exception as e:
            logger.error(f"Error in _preprocess_data: {e}")
            raise

    def _define_models(self):
        """Define the models to be trained."""
        try:
            if self.target_type == "classification":
                self.models = {
                    "LightGBM": LGBMClassifier(random_state=self.random_state),
                    "XGBoost": XGBClassifier(random_state=self.random_state, eval_metric="logloss"),
                    "RandomForest": RandomForestClassifier(random_state=self.random_state)
                }
            else:
                self.models = {
                    "LightGBM": LGBMRegressor(random_state=self.random_state),
                    "XGBoost": XGBRegressor(random_state=self.random_state),
                    "RandomForest": RandomForestRegressor(random_state=self.random_state)
                }
            logger.info(f"Defined {len(self.models)} models for {self.target_type}")
        except Exception as e:
            logger.error(f"Error in _define_models: {e}")
            raise

    def train_baseline_models(self) -> pd.DataFrame:
        """Train baseline models and return results."""
        try:
            self._detect_task_type()
            self._preprocess_data()
            self._define_models()
            
            self.results = []  # Reset results
            
            for name, model in self.models.items():
                try:
                    logger.info(f"Training {name}...")
                    model.fit(self.X_train, self.y_train)
                    y_pred = model.predict(self.X_test)
                    
                    if self.target_type == "classification":
                        metrics = {
                            "Model": name,
                            "Accuracy": accuracy_score(self.y_test, y_pred),
                            "F1": f1_score(self.y_test, y_pred, average="weighted")
                        }
                        try:
                            metrics["ROC-AUC"] = roc_auc_score(
                                self.y_test, 
                                model.predict_proba(self.X_test), 
                                multi_class="ovr"
                            )
                        except Exception as e:
                            logger.warning(f"Could not compute ROC-AUC for {name}: {e}")
                            metrics["ROC-AUC"] = None
                    else:
                        metrics = {
                            "Model": name,
                            "R²": r2_score(self.y_test, y_pred),
                            "RMSE": mean_squared_error(self.y_test, y_pred, squared=False),
                            "MAE": mean_absolute_error(self.y_test, y_pred)
                        }
                    
                    self.results.append(metrics)
                    logger.info(f"{name} training completed with metrics: {metrics}")
                except Exception as e:
                    logger.error(f"Error training {name}: {e}")
                    continue
            
            if not self.results:
                raise ValueError("No models were successfully trained")
            
            results_df = pd.DataFrame(self.results).set_index("Model")
            
            # Determine the best model
            if self.target_type == "classification":
                # Use F1 score as the primary metric for classification
                best_model_name = results_df["F1"].idxmax()
            else:
                # Use R² for regression
                best_model_name = results_df["R²"].idxmax()
            
            self.best_model_name = best_model_name
            self.best_model = self.models[best_model_name]
            
            logger.info(f"Best model: {best_model_name}")
            return results_df
        except Exception as e:
            logger.error(f"Error in train_baseline_models: {e}")
            raise

    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance from the best model."""
        try:
            if self.best_model is None:
                raise ValueError("No model trained. Call train_baseline_models first.")
            
            fi = pd.DataFrame({
                "Feature": self.X_train.columns,
                "Importance": self.best_model.feature_importances_
            }).sort_values("Importance", ascending=False)
            
            return fi
        except Exception as e:
            logger.error(f"Error in get_feature_importance: {e}")
            raise

    def hyperparameter_tuning(self, n_trials: int = 15) -> Dict[str, Any]:
        """Perform hyperparameter tuning on the best model using Optuna."""
        try:
            if self.best_model_name is None:
                raise ValueError("No best model found. Train baseline models first.")
            
            logger.info(f"Starting hyperparameter tuning for {self.best_model_name}...")
            
            def objective(trial):
                try:
                    if self.target_type == "classification":
                        params = {
                            "num_leaves": trial.suggest_int("num_leaves", 20, 200),
                            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
                            "n_estimators": trial.suggest_int("n_estimators", 50, 500)
                        }
                        model = LGBMClassifier(**params, random_state=self.random_state)
                    else:
                        params = {
                            "num_leaves": trial.suggest_int("num_leaves", 20, 200),
                            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
                            "n_estimators": trial.suggest_int("n_estimators", 50, 500)
                        }
                        model = LGBMRegressor(**params, random_state=self.random_state)

                    model.fit(self.X_train, self.y_train)
                    preds = model.predict(self.X_test)
                    
                    if self.target_type == "classification":
                        return accuracy_score(self.y_test, preds)
                    else:
                        return r2_score(self.y_test, preds)
                except Exception as e:
                    logger.error(f"Error in objective function: {e}")
                    return float('-inf')  # Return worst possible score

            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=n_trials)
            
            logger.info(f"Best parameters: {study.best_params}")
            return study.best_params
        except Exception as e:
            logger.error(f"Error in hyperparameter_tuning: {e}")
            raise

    def save_model(self) -> bytes:
        """Save the best model as a pickle file and return as bytes."""
        try:
            if self.best_model is None:
                raise ValueError("No model to save. Train a model first.")
            
            buffer = io.BytesIO()
            pickle.dump(self.best_model, buffer)
            buffer.seek(0)
            return buffer.getvalue()
        except Exception as e:
            logger.error(f"Error in save_model: {e}")
            raise