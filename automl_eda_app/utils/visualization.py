import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from typing import Optional, List
import logging

logger = logging.getLogger(__name__)

class VisualizationHelper:
    """Utility class for creating common visualizations."""
    
    def plot_missing_values(self, missing_df: pd.DataFrame, 
                           output_dir: Optional[str] = None, 
                           save_plot: bool = False):
        """Plot missing values per column."""
        try:
            if missing_df.empty:
                logger.info("No missing values to plot")
                return None
            
            plt.figure(figsize=(10, 5))
            sns.barplot(x="Column", y="Missing Count", data=missing_df, palette="crest")
            plt.title("Missing Values per Column")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            
            if save_plot and output_dir:
                os.makedirs(output_dir, exist_ok=True)
                path = os.path.join(output_dir, "missing_values.png")
                plt.savefig(path)
                plt.close()
                logger.info(f"Missing values plot saved to {path}")
                return path
            else:
                return plt.gcf()
        except Exception as e:
            logger.error(f"Error in plot_missing_values: {e}")
            if not save_plot:
                plt.close()
            raise
    
    def plot_feature_importance(self, importance_df: pd.DataFrame, 
                              title: str = "Feature Importance", 
                              output_dir: Optional[str] = None, 
                              save_plot: bool = False,
                              top_n: int = 15):
        """Plot feature importance."""
        try:
            if importance_df.empty:
                logger.info("No feature importance data to plot")
                return None
                
            plt.figure(figsize=(10, 6))
            sns.barplot(
                y="Feature", 
                x="Importance", 
                data=importance_df.head(top_n), 
                palette="crest"
            )
            plt.title(title)
            plt.tight_layout()
            
            if save_plot and output_dir:
                os.makedirs(output_dir, exist_ok=True)
                path = os.path.join(output_dir, "feature_importance.png")
                plt.savefig(path)
                plt.close()
                logger.info(f"Feature importance plot saved to {path}")
                return path
            else:
                return plt.gcf()
        except Exception as e:
            logger.error(f"Error in plot_feature_importance: {e}")
            if not save_plot:
                plt.close()
            raise
    
    def plot_correlation_heatmap(self, df: pd.DataFrame, 
                                output_dir: Optional[str] = None, 
                                save_plot: bool = False):
        """Plot correlation heatmap for numerical columns."""
        try:
            numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
            
            if len(numerical_cols) > 1:
                plt.figure(figsize=(10, 8))
                corr = df[numerical_cols].corr()
                sns.heatmap(corr, annot=False, cmap="coolwarm")
                plt.title("Correlation Heatmap")
                plt.tight_layout()
                
                if save_plot and output_dir:
                    os.makedirs(output_dir, exist_ok=True)
                    path = os.path.join(output_dir, "correlation_heatmap.png")
                    plt.savefig(path)
                    plt.close()
                    logger.info(f"Correlation heatmap saved to {path}")
                    return path
                else:
                    return plt.gcf()
            else:
                logger.info("Not enough numerical columns for correlation heatmap")
                return None
        except Exception as e:
            logger.error(f"Error in plot_correlation_heatmap: {e}")
            if not save_plot:
                plt.close()
            raise
    
    def plot_distribution(self, df: pd.DataFrame, column: str, 
                         output_dir: Optional[str] = None, 
                         save_plot: bool = False):
        """Plot distribution and boxplot for a numerical column."""
        try:
            if column not in df.columns:
                raise ValueError(f"Column '{column}' not found in DataFrame")
            
            if column not in df.select_dtypes(include=np.number).columns:
                raise ValueError(f"Column '{column}' is not numerical")
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Distribution plot
            sns.histplot(df[column], kde=True, color="teal", ax=ax1)
            ax1.set_title(f"Distribution of {column}")
            
            # Boxplot
            sns.boxplot(x=df[column], color="coral", ax=ax2)
            ax2.set_title(f"Boxplot of {column}")
            
            plt.tight_layout()
            
            if save_plot and output_dir:
                os.makedirs(output_dir, exist_ok=True)
                path = os.path.join(output_dir, f"distribution_{column}.png")
                plt.savefig(path)
                plt.close()
                logger.info(f"Distribution plot for {column} saved to {path}")
                return path
            else:
                return fig
        except Exception as e:
            logger.error(f"Error in plot_distribution: {e}")
            if not save_plot:
                plt.close()
            raise