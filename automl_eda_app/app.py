import streamlit as st
import pandas as pd
import numpy as np
import os
import tempfile
import shutil
import base64
from io import BytesIO
import time
import matplotlib.pyplot as plt
import seaborn as sns
import logging

# Import our modules
from modules.eda_pipeline import ModelAwareEDA
from modules.model_training import ModelTrainer
from modules.report_generator import ReportGenerator
from utils.data_processor import DataProcessor
from utils.visualization import VisualizationHelper

# Set page config
st.set_page_config(
    page_title="AutoML EDA Suite",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .section-header {
        font-size: 1.8rem;
        color: #43A047;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f9f9f9;
        border-radius: 5px;
        padding: 15px;
        margin-bottom: 15px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .stTabs [data-baseweb="tab-list"] {
        background-color: #f0f2f6;
        border-radius: 5px;
        padding: 5px;
        margin-bottom: 20px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 5px;
        padding: 10px 20px;
        font-weight: 600;
    }
    .tab-content {
        margin-top: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state variables
def init_session_state():
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'target_col' not in st.session_state:
        st.session_state.target_col = None
    if 'temp_dir' not in st.session_state:
        st.session_state.temp_dir = tempfile.mkdtemp()
    if 'eda_done' not in st.session_state:
        st.session_state.eda_done = False
    if 'model_training_done' not in st.session_state:
        st.session_state.model_training_done = False
    if 'tuning_done' not in st.session_state:
        st.session_state.tuning_done = False
    if 'report_done' not in st.session_state:
        st.session_state.report_done = False
    if 'cleanup_registered' not in st.session_state:
        st.session_state.cleanup_registered = False

init_session_state()

# Register cleanup function
def cleanup_temp_dir():
    try:
        if hasattr(st.session_state, 'temp_dir') and os.path.exists(st.session_state.temp_dir):
            shutil.rmtree(st.session_state.temp_dir)
            logger.info(f"Cleaned up temporary directory: {st.session_state.temp_dir}")
    except Exception as e:
        logger.error(f"Error cleaning up temporary directory: {e}")

# Ensure cleanup is registered only once
if not st.session_state.cleanup_registered:
    st.session_state.cleanup_registered = True
    # Register cleanup function to run when the session ends
    st.session_state._cleanup_func = cleanup_temp_dir

# Sidebar
st.sidebar.title("🔧 Navigation")
st.sidebar.markdown("### Upload your dataset to get started")

# Main app header
st.markdown('<h1 class="main-header">🤖 Smart AutoML + EDA Suite</h1>', unsafe_allow_html=True)
st.markdown("""
An advanced AI-powered framework that performs:
- 📊 Automated Exploratory Data Analysis (EDA)  
- 🧹 Data cleaning recommendations  
- ⚙️ AutoML with LightGBM, XGBoost, and RandomForest  
- 🧪 Hyperparameter tuning with Optuna  
- 📈 Model comparison and export  
""")
st.markdown("---")

# File upload
uploaded_file = st.sidebar.file_uploader("📂 Upload your dataset (.csv or .xlsx)", type=["csv", "xlsx"])

if uploaded_file is not None:
    try:
        # Load data using our data processor
        data_processor = DataProcessor()
        st.session_state.df = data_processor.load_data(uploaded_file)
        st.session_state.data_loaded = True
        
        st.success(f"✅ Dataset loaded successfully! Shape: {st.session_state.df.shape}")
        
        # Display data preview
        with st.expander("📋 Data Preview"):
            st.dataframe(st.session_state.df.head())
        
        # Get basic data info
        data_types, missing_values, total_missing = data_processor.get_data_info(st.session_state.df)
        
        # Display basic info
        col1, col2, col3 = st.columns(3)
        col1.metric("Rows", st.session_state.df.shape[0])
        col2.metric("Columns", st.session_state.df.shape[1])
        col3.metric("Missing Values", total_missing)
        
        # Select target variable
        st.session_state.target_col = st.selectbox(
            "🎯 Select Target Variable", 
            ["None"] + st.session_state.df.columns.tolist()
        )
        
        if st.session_state.target_col == "None":
            st.warning("⚠️ Please select a target variable to continue.")
            st.stop()
        
        # Create tabs for different sections
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 EDA", 
            "🤖 Model Training", 
            "🔧 Hyperparameter Tuning", 
            "📄 Report", 
            "💾 Export"
        ])
        
        # Tab 1: EDA
        with tab1:
            st.markdown('<div class="tab-content">', unsafe_allow_html=True)
            st.markdown('<h2 class="section-header">Exploratory Data Analysis</h2>', unsafe_allow_html=True)
            
            if not st.session_state.eda_done:
                if st.button("Run EDA Analysis", key="run_eda"):
                    with st.spinner("Running EDA..."):
                        # Initialize EDA
                        eda = ModelAwareEDA(
                            st.session_state.df, 
                            target=st.session_state.target_col, 
                            output_dir=st.session_state.temp_dir
                        )
                        
                        # Run EDA
                        eda_summary, eda_feature_insights, eda_importance_df = eda.run()
                        
                        # Store results in session state
                        st.session_state.eda_summary = eda_summary
                        st.session_state.eda_feature_insights = eda_feature_insights
                        st.session_state.eda_importance_df = eda_importance_df
                        st.session_state.eda_done = True
                        st.session_state.data_types = data_types
                        st.session_state.missing_values = missing_values
                        st.session_state.total_missing = total_missing
                        st.session_state.numerical_cols = st.session_state.df.select_dtypes(include=np.number).columns.tolist()
                    
                    st.success("EDA Analysis Completed!")
            
            if st.session_state.eda_done:
                # Display EDA results
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.subheader("Data Types")
                    st.write(st.session_state.data_types)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col2:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.subheader("Missing Values")
                    if st.session_state.total_missing > 0:
                        missing_df = pd.DataFrame({
                            "Column": st.session_state.missing_values[st.session_state.missing_values > 0].index,
                            "Missing Count": st.session_state.missing_values[st.session_state.missing_values > 0].values,
                            "Missing %": (st.session_state.missing_values[st.session_state.missing_values > 0] / len(st.session_state.df) * 100).round(2)
                        })
                        st.dataframe(missing_df)
                    else:
                        st.success("No missing values detected!")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Feature insights
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.subheader("Feature Insights")
                st.dataframe(st.session_state.eda_feature_insights)
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Feature importance
                if st.session_state.eda_importance_df is not None:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.subheader("Feature Importance (EDA)")
                    viz_helper = VisualizationHelper()
                    st.pyplot(viz_helper.plot_feature_importance(st.session_state.eda_importance_df, "EDA Feature Importance"))
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Univariate analysis
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.subheader("Univariate Analysis")
                if st.session_state.numerical_cols:
                    selected_num_col = st.selectbox("Select a numerical column for distribution", st.session_state.numerical_cols)
                    if selected_num_col:
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(f"Distribution of {selected_num_col}")
                            fig, ax = plt.subplots()
                            sns.histplot(st.session_state.df[selected_num_col], kde=True, ax=ax)
                            st.pyplot(fig)
                        with col2:
                            st.write(f"Boxplot of {selected_num_col}")
                            fig, ax = plt.subplots()
                            sns.boxplot(x=st.session_state.df[selected_num_col], ax=ax)
                            st.pyplot(fig)
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Correlation heatmap
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.subheader("Correlation Heatmap")
                if len(st.session_state.numerical_cols) > 1:
                    fig, ax = plt.subplots(figsize=(10, 8))
                    corr = st.session_state.df[st.session_state.numerical_cols].corr()
                    sns.heatmap(corr, annot=False, cmap="coolwarm", ax=ax)
                    st.pyplot(fig)
                else:
                    st.info("Not enough numerical columns for correlation heatmap.")
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.info("Click 'Run EDA Analysis' to start exploratory data analysis.")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Tab 2: Model Training
        with tab2:
            st.markdown('<div class="tab-content">', unsafe_allow_html=True)
            st.markdown('<h2 class="section-header">Model Training</h2>', unsafe_allow_html=True)
            
            if not st.session_state.model_training_done:
                if st.button("Train Models", key="train_models"):
                    with st.spinner("Training baseline models..."):
                        # Initialize model trainer
                        trainer = ModelTrainer(
                            st.session_state.df, 
                            target=st.session_state.target_col,
                            output_dir=st.session_state.temp_dir
                        )
                        
                        # Train baseline models
                        model_results_df = trainer.train_baseline_models()
                        
                        # Store results in session state
                        st.session_state.trainer = trainer
                        st.session_state.model_results_df = model_results_df
                        st.session_state.model_training_done = True
                    
                    st.success("Model Training Completed!")
            
            if st.session_state.model_training_done:
                # Display model results
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.subheader("Baseline Model Comparison")
                st.dataframe(st.session_state.model_results_df.style.format("{:.4f}").highlight_max(color='lightgreen', axis=0))
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Display best model
                st.success(f"🏆 Best Model (Baseline): **{st.session_state.trainer.best_model_name}**")
                
                # Display feature importance
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.subheader("Feature Importance")
                model_fi_df = st.session_state.trainer.get_feature_importance()
                viz_helper = VisualizationHelper()
                st.pyplot(viz_helper.plot_feature_importance(model_fi_df, f"Feature Importance ({st.session_state.trainer.best_model_name})"))
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.info("Click 'Train Models' to start model training.")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Tab 3: Hyperparameter Tuning
        with tab3:
            st.markdown('<div class="tab-content">', unsafe_allow_html=True)
            st.markdown('<h2 class="section-header">Hyperparameter Tuning</h2>', unsafe_allow_html=True)
            
            if not st.session_state.model_training_done:
                st.warning("Please complete Model Training first before tuning hyperparameters.")
            else:
                if not st.session_state.tuning_done:
                    if st.button("Run Hyperparameter Tuning", key="run_tuning"):
                        with st.spinner("Running hyperparameter optimization... please wait"):
                            best_params = st.session_state.trainer.hyperparameter_tuning(n_trials=15)
                            st.session_state.best_params = best_params
                            st.session_state.tuning_done = True
                        st.success("✅ Tuning completed!")
                
                if st.session_state.tuning_done:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.subheader("Best Parameters")
                    st.json(st.session_state.best_params)
                    st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.info("Click 'Run Hyperparameter Tuning' to optimize model parameters.")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Tab 4: Report
        with tab4:
            st.markdown('<div class="tab-content">', unsafe_allow_html=True)
            st.markdown('<h2 class="section-header">Generate Report</h2>', unsafe_allow_html=True)
            
            if not st.session_state.eda_done or not st.session_state.model_training_done:
                st.warning("Please complete both EDA and Model Training before generating a report.")
            else:
                if not st.session_state.report_done:
                    if st.button("Generate Comprehensive Report", key="generate_report"):
                        with st.spinner("Generating report..."):
                            # Get missing values dataframe for the report
                            if st.session_state.total_missing > 0:
                                missing_df = pd.DataFrame({
                                    "Column": st.session_state.missing_values[st.session_state.missing_values > 0].index,
                                    "Missing Count": st.session_state.missing_values[st.session_state.missing_values > 0].values,
                                    "Missing %": (st.session_state.missing_values[st.session_state.missing_values > 0] / len(st.session_state.df) * 100).round(2)
                                })
                            else:
                                missing_df = pd.DataFrame(columns=["Column", "Missing Count", "Missing %"])
                            
                            # Get model feature importance
                            model_fi_df = st.session_state.trainer.get_feature_importance()
                            
                            # Generate the report
                            report_generator = ReportGenerator()
                            report_path = report_generator.generate_automl_report(
                                eda_summary=st.session_state.eda_summary,
                                eda_missing_df=missing_df,
                                eda_feature_insights=st.session_state.eda_feature_insights,
                                eda_importance_df=st.session_state.eda_importance_df,
                                model_results_df=st.session_state.model_results_df,
                                model_feature_importance=model_fi_df,
                                best_model_name=st.session_state.trainer.best_model_name,
                                output_dir=st.session_state.temp_dir
                            )
                            
                            # Read the report and provide download button
                            with open(report_path, "r", encoding="utf-8") as f:
                                report_html = f.read()
                            
                            st.session_state.report_html = report_html
                            st.session_state.report_done = True
                        
                        st.success("Report generated successfully!")
                
                if st.session_state.report_done:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.subheader("Download Report")
                    b64 = base64.b64encode(st.session_state.report_html.encode()).decode()
                    href = f'<a href="data:text/html;base64,{b64}" download="automl_report.html">📥 Download Report</a>'
                    st.markdown(href, unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.info("Click 'Generate Comprehensive Report' to create your report.")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Tab 5: Export
        with tab5:
            st.markdown('<div class="tab-content">', unsafe_allow_html=True)
            st.markdown('<h2 class="section-header">Export Results</h2>', unsafe_allow_html=True)
            
            if not st.session_state.model_training_done:
                st.warning("Please complete Model Training before exporting results.")
            else:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.subheader("Export Model")
                model_bytes = st.session_state.trainer.save_model()
                st.download_button(
                    label="Download Best Model (Pickle)",
                    data=model_bytes,
                    file_name=f"{st.session_state.trainer.best_model_name}_model.pkl",
                    mime="application/octet-stream"
                )
                st.markdown('</div>', unsafe_allow_html=True)
                
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.subheader("Export Data")
                # Export cleaned data
                clean_data = st.session_state.df.dropna()
                clean_csv = clean_data.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Cleaned Data (CSV)",
                    data=clean_csv,
                    file_name="cleaned_data.csv",
                    mime="text/csv"
                )
                st.markdown('</div>', unsafe_allow_html=True)
                
                if st.session_state.eda_done:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.subheader("Export EDA Results")
                    # Export feature insights
                    insights_csv = st.session_state.eda_feature_insights.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download Feature Insights (CSV)",
                        data=insights_csv,
                        file_name="feature_insights.csv",
                        mime="text/csv"
                    )
                    
                    # Export feature importance if available
                    if st.session_state.eda_importance_df is not None:
                        fi_csv = st.session_state.eda_importance_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="Download EDA Feature Importance (CSV)",
                            data=fi_csv,
                            file_name="eda_feature_importance.csv",
                            mime="text/csv"
                        )
                    st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        st.error(f"An error occurred: {e}")
    
    finally:
        # Clean up temporary directory when app closes
        pass  # Streamlit handles session cleanup automatically

else:
    st.info("👆 Upload a dataset to start automated EDA + modeling.")