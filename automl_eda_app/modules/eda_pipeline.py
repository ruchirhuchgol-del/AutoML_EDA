import os
import logging
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from jinja2 import Template
from scipy import stats
from typing import List, Optional, Tuple, Dict, Any
from lightgbm import LGBMClassifier, LGBMRegressor
from warnings import filterwarnings

filterwarnings("ignore")

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class ModelAwareEDA:
    """Advanced EDA + model-driven feature importance."""

    def __init__(self, df: pd.DataFrame, target: Optional[str] = None,
                 output_dir: str = "eda_outputs"):
        self.df = df.copy()  # Work with a copy to avoid modifying original data
        self.target = target
        self.output_dir = output_dir
        self.numerical_cols: List[str] = []
        self.categorical_cols: List[str] = []
        self.target_type: Optional[str] = None
        os.makedirs(self.output_dir, exist_ok=True)

    def identify_columns(self):
        """Detect data types."""
        try:
            self.numerical_cols = self.df.select_dtypes(include=np.number).columns.tolist()
            self.categorical_cols = self.df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

            if self.target and self.target in self.numerical_cols:
                self.target_type = "numerical"
            elif self.target and self.target in self.categorical_cols:
                self.target_type = "categorical"

            logger.info(f"Numerical: {len(self.numerical_cols)} | Categorical: {len(self.categorical_cols)}")
            if self.target:
                logger.info(f"Target detected: '{self.target}' ({self.target_type})")
        except Exception as e:
            logger.error(f"Error in identify_columns: {e}")
            raise

    def assess_data_quality(self) -> pd.DataFrame:
        """Missing values + duplicates."""
        try:
            missing = self.df.isnull().sum()
            missing_df = missing[missing > 0].reset_index()
            missing_df.columns = ["Column", "Missing Count"]
            missing_df["Missing %"] = (missing_df["Missing Count"] / len(self.df)) * 100
            
            if not missing_df.empty:
                plt.figure(figsize=(10, 5))
                sns.barplot(x="Column", y="Missing Count", data=missing_df, palette="crest")
                plt.title("Missing Values per Column")
                plt.xticks(rotation=45, ha="right")
                plt.tight_layout()
                plt.savefig(os.path.join(self.output_dir, "missing_values.png"))
                plt.close()
            
            return missing_df
        except Exception as e:
            logger.error(f"Error in assess_data_quality: {e}")
            raise

    def univariate(self):
        """Feature distributions."""
        try:
            for col in self.numerical_cols:
                try:
                    plt.figure(figsize=(12, 5))
                    plt.subplot(1, 2, 1)
                    sns.histplot(self.df[col], kde=True, color="teal")
                    plt.title(f"Distribution of {col}")
                    plt.subplot(1, 2, 2)
                    sns.boxplot(x=self.df[col], color="coral")
                    plt.title(f"Boxplot of {col}")
                    plt.tight_layout()
                    plt.savefig(os.path.join(self.output_dir, f"univariate_{col}.png"))
                    plt.close()
                except Exception as e:
                    logger.warning(f"Could not generate plots for column {col}: {e}")
                    continue
        except Exception as e:
            logger.error(f"Error in univariate: {e}")
            raise

    def correlation_heatmap(self):
        """Numeric correlations."""
        try:
            if len(self.numerical_cols) > 1:
                corr = self.df[self.numerical_cols].corr()
                plt.figure(figsize=(10, 8))
                sns.heatmap(corr, annot=False, cmap="coolwarm")
                plt.title("Correlation Heatmap")
                plt.tight_layout()
                plt.savefig(os.path.join(self.output_dir, "correlation_heatmap.png"))
                plt.close()
            else:
                logger.info("Not enough numerical columns for correlation heatmap.")
        except Exception as e:
            logger.error(f"Error in correlation_heatmap: {e}")
            raise

    def feature_engineering_insights(self) -> pd.DataFrame:
        """Detect skew, variance, cardinality."""
        try:
            insights = []
            for col in self.numerical_cols:
                try:
                    skewness = self.df[col].skew()
                    var = self.df[col].var()
                    insights.append({
                        "Feature": col, 
                        "Type": "Numerical", 
                        "Skew": round(skewness, 2), 
                        "Variance": round(var, 2)
                    })
                except Exception as e:
                    logger.warning(f"Could not compute insights for numerical column {col}: {e}")
                    continue

            for col in self.categorical_cols:
                try:
                    uniq = self.df[col].nunique()
                    insights.append({
                        "Feature": col, 
                        "Type": "Categorical", 
                        "Unique Values": uniq
                    })
                except Exception as e:
                    logger.warning(f"Could not compute insights for categorical column {col}: {e}")
                    continue

            insights_df = pd.DataFrame(insights)
            insights_df.to_csv(os.path.join(self.output_dir, "feature_insights.csv"), index=False)
            return insights_df
        except Exception as e:
            logger.error(f"Error in feature_engineering_insights: {e}")
            raise

    def compute_feature_importance(self) -> Optional[pd.DataFrame]:
        """Model-based feature importance via LightGBM."""
        try:
            if not self.target or self.target not in self.df.columns:
                logger.info("No target variable provided — skipping model importance.")
                return None

            df = self.df.dropna(subset=[self.target])
            if df.empty:
                logger.warning("No rows remaining after dropping NA values in target column.")
                return None
                
            X = df.drop(columns=[self.target])
            y = df[self.target]

            # Encode categoricals
            X = pd.get_dummies(X, drop_first=True)
            if self.target_type == "categorical":
                y = y.astype("category").cat.codes

            logger.info("Training LightGBM for feature importance...")
            model = LGBMClassifier(random_state=42) if self.target_type == "categorical" else LGBMRegressor(random_state=42)
            model.fit(X, y)

            importance_df = pd.DataFrame({
                "Feature": X.columns,
                "Importance": model.feature_importances_
            }).sort_values("Importance", ascending=False)

            plt.figure(figsize=(10, 6))
            sns.barplot(x="Importance", y="Feature", data=importance_df.head(20), palette="crest")
            plt.title("Top 20 Feature Importances")
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, "feature_importance.png"))
            plt.close()

            importance_df.to_csv(os.path.join(self.output_dir, "feature_importances.csv"), index=False)
            return importance_df
        except Exception as e:
            logger.error(f"Error in compute_feature_importance: {e}")
            raise

    def run(self) -> Tuple[Dict[str, Any], pd.DataFrame, Optional[pd.DataFrame]]:
        """Run the full model-aware EDA pipeline."""
        try:
            self.identify_columns()
            missing_df = self.assess_data_quality()
            self.univariate()
            self.correlation_heatmap()
            feature_insights = self.feature_engineering_insights()
            importance_df = self.compute_feature_importance()

            summary = {
                "shape": self.df.shape,
                "target": self.target,
                "target_type": self.target_type,
            }

            logger.info("Model-aware EDA completed.")
            return summary, feature_insights, importance_df
        except Exception as e:
            logger.error(f"Error in EDA pipeline run: {e}")
            raise