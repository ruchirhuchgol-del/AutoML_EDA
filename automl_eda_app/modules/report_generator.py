
import os
import base64
import pandas as pd
from jinja2 import Template
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class ReportGenerator:
    def __init__(self):
        pass
    
    def generate_eda_report(
        self,
        eda_summary: Dict[str, Any],
        missing_df: pd.DataFrame,
        feature_insights: pd.DataFrame,
        importance_df: Optional[pd.DataFrame],
        output_dir: str = "report_outputs"
    ) -> str:
        """Generate EDA HTML report."""
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            template = Template("""
            <html>
            <head>
                <title>Model-Aware EDA Report</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 40px; color:#222; }
                    h1,h2,h3 { color: #003366; }
                    table { border-collapse: collapse; width: 100%; margin:20px 0; }
                    th,td { border:1px solid #ddd; padding:8px; text-align: left; }
                    th { background:#f2f2f2; }
                    img { max-width:90%; margin:10px; border:1px solid #ccc; }
                    .section { margin-bottom: 30px; }
                    .metric { background-color: #f9f9f9; padding: 15px; border-radius: 5px; margin: 10px 0; }
                </style>
            </head>
            <body>
                <h1>Model-Aware EDA Report</h1>

                <div class="section">
                    <h2>Dataset Overview</h2>
                    <div class="metric">
                        <p><b>Shape:</b> {{ summary.shape[0] }} × {{ summary.shape[1] }}</p>
                        {% if summary.target %}
                        <p><b>Target:</b> {{ summary.target }} ({{ summary.target_type }})</p>
                        {% endif %}
                    </div>
                </div>

                <div class="section">
                    <h2>Missing Values</h2>
                    {% if missing_df.shape[0] > 0 %}
                        {{ missing_df.to_html(index=False, table_id="missing-table") }}
                        <img src="missing_values.png" alt="Missing Values Plot">
                    {% else %}
                        <p>No missing values detected.</p>
                    {% endif %}
                </div>

                <div class="section">
                    <h2>Feature Insights</h2>
                    {{ feature_insights.to_html(index=False, table_id="insights-table") }}
                </div>

                <div class="section">
                    <h2>Feature Importance (LightGBM)</h2>
                    {% if importance_df is not none %}
                        <img src="feature_importance.png" alt="Feature Importance Plot">
                        {{ importance_df.head(20).to_html(index=False, table_id="importance-table") }}
                    {% else %}
                        <p>No feature importance (no target provided).</p>
                    {% endif %}
                </div>

                <div class="section">
                    <h2>Correlation Heatmap</h2>
                    <img src="correlation_heatmap.png" alt="Correlation Heatmap">
                </div>
            </body>
            </html>
            """)

            html = template.render(
                summary=eda_summary,
                missing_df=missing_df,
                feature_insights=feature_insights,
                importance_df=importance_df
            )

            path = os.path.join(output_dir, "eda_report.html")
            with open(path, "w", encoding="utf-8") as f:
                f.write(html)
            
            logger.info(f"EDA report generated at {path}")
            return path
        except Exception as e:
            logger.error(f"Error generating EDA report: {e}")
            raise

    def generate_automl_report(
        self,
        eda_summary: Dict[str, Any],
        eda_missing_df: pd.DataFrame,
        eda_feature_insights: pd.DataFrame,
        eda_importance_df: Optional[pd.DataFrame],
        model_results_df: pd.DataFrame,
        model_feature_importance: pd.DataFrame,
        best_model_name: str,
        output_dir: str = "report_outputs"
    ) -> str:
        """Generate a comprehensive AutoML + EDA HTML report."""
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            # Convert dataframes to HTML for embedding
            eda_missing_html = eda_missing_df.to_html(index=False, table_id="eda-missing-table") if not eda_missing_df.empty else "<p>No missing values detected.</p>"
            eda_insights_html = eda_feature_insights.to_html(index=False, table_id="eda-insights-table")
            model_results_html = model_results_df.to_html(table_id="model-results-table")
            model_fi_html = model_feature_importance.head(20).to_html(index=False, table_id="model-fi-table")
            
            # Check if we have EDA feature importance
            eda_fi_html = ""
            if eda_importance_df is not None:
                eda_fi_html = eda_importance_df.head(20).to_html(index=False, table_id="eda-fi-table")
            
            template = Template("""
            <html>
            <head>
                <title>AutoML + EDA Report</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 40px; color: #333; }
                    h1, h2, h3 { color: #003366; }
                    table { border-collapse: collapse; width: 100%; margin: 20px 0; }
                    th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                    th { background-color: #f2f2f2; }
                    img { max-width: 90%; margin: 20px 0; border: 1px solid #ccc; }
                    .section { margin-bottom: 40px; }
                    .metric { background-color: #f9f9f9; padding: 15px; border-radius: 5px; margin: 10px 0; }
                    .best-model { background-color: #e8f5e9; padding: 15px; border-left: 5px solid #4caf50; margin: 10px 0; }
                </style>
            </head>
            <body>
                <h1>AutoML + EDA Report</h1>
                
                <div class="section">
                    <h2>Dataset Overview</h2>
                    <div class="metric">
                        <p><b>Shape:</b> {{ eda_summary.shape[0] }} rows × {{ eda_summary.shape[1] }} columns</p>
                        {% if eda_summary.target %}
                        <p><b>Target Variable:</b> {{ eda_summary.target }} ({{ eda_summary.target_type }})</p>
                        {% endif %}
                    </div>
                </div>
                
                <div class="section">
                    <h2>Exploratory Data Analysis</h2>
                    <h3>Missing Values</h3>
                    {{ eda_missing_html }}
                    
                    <h3>Feature Insights</h3>
                    {{ eda_insights_html }}
                    
                    {% if eda_fi_html %}
                    <h3>EDA Feature Importance</h3>
                    {{ eda_fi_html }}
                    {% endif %}
                </div>
                
                <div class="section">
                    <h2>Model Training Results</h2>
                    <div class="best-model">
                        <h3>🏆 Best Model: {{ best_model_name }}</h3>
                    </div>
                    <h3>Baseline Model Comparison</h3>
                    {{ model_results_html }}
                    
                    <h3>Feature Importance ({{ best_model_name }})</h3>
                    {{ model_fi_html }}
                </div>
            </body>
            </html>
            """)
            
            html = template.render(
                eda_summary=eda_summary,
                eda_missing_html=eda_missing_html,
                eda_insights_html=eda_insights_html,
                eda_fi_html=eda_fi_html,
                model_results_html=model_results_html,
                model_fi_html=model_fi_html,
                best_model_name=best_model_name
            )
            
            # Save the report
            report_path = os.path.join(output_dir, "automl_report.html")
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(html)
            
            logger.info(f"AutoML report generated at {report_path}")
            return report_path
        except Exception as e:
            logger.error(f"Error generating AutoML report: {e}")
            raise