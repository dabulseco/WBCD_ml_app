"""
Machine Learning Application with Streamlit UI
Author: Created for scientific research applications
Purpose: Flexible ML model building with classification and regression support

This application provides:
- Data upload and exploration
- Feature selection for modeling
- Model type selection (Classification/Regression)
- Hyperparameter tuning with sensible defaults
- Comprehensive model evaluation metrics
- Visualization of results (confusion matrices, ROC curves, correlations)
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    mean_squared_error, mean_absolute_error, r2_score
)
import base64
from io import BytesIO
from datetime import datetime
import joblib
import pickle
import joblib
import json
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="ML Model Builder",
    page_icon="🤖",
    layout="wide"
)

def fig_to_base64(fig):
    """
    Convert a matplotlib figure to a base64 encoded string for HTML embedding.
    
    Parameters:
    -----------
    fig : matplotlib.figure.Figure
        The matplotlib figure to convert
    
    Returns:
    --------
    str : Base64 encoded string of the figure
    """
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode()
    buf.close()
    return img_str

def generate_html_report(report_data):
    """
    Generate a comprehensive HTML report using Bootstrap 5 and HTML5.
    
    Parameters:
    -----------
    report_data : dict
        Dictionary containing all the data for the report including metrics,
        figures, and configuration parameters
    
    Returns:
    --------
    str : Complete HTML document as a string
    """
    html_template = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Machine Learning Model Report - {datetime.now().strftime('%Y-%m-%d %H:%M')}</title>
    
    <!-- Bootstrap 5 CSS -->
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    
    <!-- Custom CSS -->
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: #f8f9fa;
            padding: 20px;
        }}
        .report-header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            border-radius: 15px;
            margin-bottom: 30px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        }}
        .metric-card {{
            background: white;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.07);
            transition: transform 0.3s ease;
        }}
        .metric-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 8px 15px rgba(0,0,0,0.1);
        }}
        .metric-value {{
            font-size: 2.5em;
            font-weight: bold;
            color: #667eea;
        }}
        .metric-label {{
            color: #6c757d;
            font-size: 0.9em;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        .section-header {{
            background: white;
            padding: 15px 25px;
            border-left: 5px solid #667eea;
            margin: 30px 0 20px 0;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }}
        .config-table {{
            background: white;
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 4px 6px rgba(0,0,0,0.07);
        }}
        .img-container {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.07);
        }}
        .img-container img {{
            max-width: 100%;
            height: auto;
            border-radius: 5px;
        }}
        .footer {{
            text-align: center;
            padding: 30px;
            color: #6c757d;
            margin-top: 50px;
        }}
        .badge-custom {{
            font-size: 0.9em;
            padding: 8px 15px;
        }}
        pre {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            border-left: 3px solid #667eea;
        }}
    </style>
</head>
<body>
    <div class="container-fluid">
        <!-- Header -->
        <div class="report-header">
            <div class="row align-items-center">
                <div class="col-md-8">
                    <h1 class="display-4 mb-3">🤖 Machine Learning Model Report</h1>
                    <p class="lead mb-2">Analysis Type: <span class="badge bg-light text-dark badge-custom">{report_data['model_type']}</span></p>
                    <p class="mb-0">Generated: {datetime.now().strftime('%B %d, %Y at %H:%M:%S')}</p>
                </div>
                <div class="col-md-4 text-end">
                    <h3>Target Variable</h3>
                    <h4><span class="badge bg-light text-dark">{report_data['target_column']}</span></h4>
                </div>
            </div>
        </div>

        <!-- Dataset Information -->
        <div class="section-header">
            <h2>📊 Dataset Information</h2>
        </div>
        <div class="row">
            <div class="col-md-3">
                <div class="metric-card text-center">
                    <div class="metric-label">Total Samples</div>
                    <div class="metric-value">{report_data['total_samples']}</div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="metric-card text-center">
                    <div class="metric-label">Training Samples</div>
                    <div class="metric-value">{report_data['train_samples']}</div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="metric-card text-center">
                    <div class="metric-label">Test Samples</div>
                    <div class="metric-value">{report_data['test_samples']}</div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="metric-card text-center">
                    <div class="metric-label">Features Used</div>
                    <div class="metric-value">{report_data['num_features']}</div>
                </div>
            </div>
        </div>

        <!-- Model Configuration -->
        <div class="section-header">
            <h2>⚙️ Model Configuration</h2>
        </div>
        <div class="config-table">
            <table class="table table-striped mb-0">
                <thead class="table-dark">
                    <tr>
                        <th>Parameter</th>
                        <th>Value</th>
                    </tr>
                </thead>
                <tbody>
                    <tr><td>Algorithm</td><td>Random Forest {report_data['model_type']}</td></tr>
                    <tr><td>Number of Estimators</td><td>{report_data['config']['n_estimators']}</td></tr>
                    <tr><td>Maximum Depth</td><td>{report_data['config']['max_depth']}</td></tr>
                    <tr><td>Min Samples Split</td><td>{report_data['config']['min_samples_split']}</td></tr>
                    <tr><td>Min Samples Leaf</td><td>{report_data['config']['min_samples_leaf']}</td></tr>
                    <tr><td>Random State</td><td>{report_data['config']['random_state']}</td></tr>
                    <tr><td>Test Set Size</td><td>{report_data['config']['test_size']*100:.1f}%</td></tr>
                </tbody>
            </table>
        </div>

        <!-- Selected Features -->
        <div class="section-header mt-4">
            <h2>🎯 Selected Features</h2>
        </div>
        <div class="card">
            <div class="card-body">
                <div class="row">
                    {''.join([f'<div class="col-md-3 mb-2"><span class="badge bg-secondary">{feat}</span></div>' for feat in report_data['selected_features']])}
                </div>
            </div>
        </div>

        {report_data['metrics_html']}

        {report_data['visualizations_html']}

        <!-- Feature Importance -->
        <div class="section-header">
            <h2>🎯 Feature Importance</h2>
        </div>
        <div class="row">
            <div class="col-md-8">
                {report_data['feature_importance_plot']}
            </div>
            <div class="col-md-4">
                <div class="card">
                    <div class="card-header bg-dark text-white">
                        <h5 class="mb-0">Top Features</h5>
                    </div>
                    <div class="card-body">
                        <table class="table table-sm">
                            <thead>
                                <tr>
                                    <th>Feature</th>
                                    <th>Importance</th>
                                </tr>
                            </thead>
                            <tbody>
                                {report_data['feature_importance_table']}
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
        </div>

        <!-- Footer -->
        <div class="footer">
            <hr>
            <p>Report generated by ML Model Builder</p>
            <p class="text-muted">Built with Python, scikit-learn, and Bootstrap 5</p>
        </div>
    </div>

    <!-- Bootstrap 5 JS Bundle -->
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
"""
    return html_template

def save_model_package(model, scaler, label_encoders, target_encoder, config, selected_features, target_column, model_type):
    """
    Save the complete model package including the trained model, scalers, encoders, and metadata.
    
    Parameters:
    -----------
    model : sklearn model
        Trained machine learning model
    scaler : StandardScaler
        Fitted scaler for features
    label_encoders : dict
        Dictionary of label encoders for categorical features
    target_encoder : LabelEncoder or None
        Encoder for target variable (classification only)
    config : dict
        Model configuration parameters
    selected_features : list
        List of feature names used in training
    target_column : str
        Name of target variable
    model_type : str
        'Classification' or 'Regression'
    
    Returns:
    --------
    bytes : Serialized model package
    """
    model_package = {
        'model': model,
        'scaler': scaler,
        'label_encoders': label_encoders,
        'target_encoder': target_encoder,
        'config': config,
        'selected_features': selected_features,
        'target_column': target_column,
        'model_type': model_type,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # Serialize to bytes
    buffer = BytesIO()
    joblib.dump(model_package, buffer)
    buffer.seek(0)
    return buffer.getvalue()

def load_model_package(uploaded_file):
    """
    Load a saved model package from file.
    
    Parameters:
    -----------
    uploaded_file : UploadedFile
        Streamlit uploaded file object containing the model package
    
    Returns:
    --------
    dict : Model package containing model and all preprocessing objects
    """
    return joblib.load(uploaded_file)

def make_predictions(model_package, new_data):
    """
    Make predictions on new data using a loaded model package.
    
    Parameters:
    -----------
    model_package : dict
        Loaded model package containing model and preprocessors
    new_data : DataFrame
        New data to make predictions on
    
    Returns:
    --------
    tuple : (predictions, prediction_probabilities or None)
    """
    # Extract components
    model = model_package['model']
    scaler = model_package['scaler']
    label_encoders = model_package['label_encoders']
    target_encoder = model_package['target_encoder']
    selected_features = model_package['selected_features']
    model_type = model_package['model_type']
    
    # Ensure we have all required features
    missing_features = set(selected_features) - set(new_data.columns)
    if missing_features:
        raise ValueError(f"Missing required features: {missing_features}")
    
    # Select and prepare features
    X_new = new_data[selected_features].copy()
    
    # Handle missing values
    X_new = X_new.fillna(X_new.mean(numeric_only=True))
    
    # Encode categorical features using the saved encoders
    for col, encoder in label_encoders.items():
        if col in X_new.columns:
            # Handle unseen categories
            X_new[col] = X_new[col].astype(str)
            X_new[col] = X_new[col].apply(
                lambda x: x if x in encoder.classes_ else encoder.classes_[0]
            )
            X_new[col] = encoder.transform(X_new[col])
    
    # Scale features
    X_new_scaled = scaler.transform(X_new)
    
    # Make predictions
    predictions = model.predict(X_new_scaled)
    
    # Get prediction probabilities for classification
    if model_type == "Classification":
        pred_proba = model.predict_proba(X_new_scaled)
        
        # Decode predictions if target was encoded
        if target_encoder is not None:
            predictions = target_encoder.inverse_transform(predictions)
    else:
        pred_proba = None
    
    return predictions, pred_proba

# Title and description
st.title("🤖 Machine Learning Model Builder")
st.markdown("""
This application helps you build and evaluate machine learning models with an intuitive interface.
Upload your data, select features, configure the model, and get comprehensive evaluation metrics.
""")

# Sidebar for configuration
st.sidebar.header("⚙️ Configuration")

# Add app mode selection
st.sidebar.subheader("Application Mode")
app_mode = st.sidebar.radio(
    "Select Mode:",
    ["Train New Model", "Use Saved Model"],
    help="Choose whether to train a new model or use a previously saved one"
)

st.sidebar.markdown("---")

# Advanced Settings
st.sidebar.subheader("Advanced Settings")

# Data preprocessing options
with st.sidebar.expander("🔧 Data Preprocessing"):
    handle_missing = st.selectbox(
        "Missing Value Strategy",
        ["Mean", "Median", "Mode", "Drop"],
        help="How to handle missing values in numeric columns"
    )
    
    scale_features = st.checkbox(
        "Scale Features",
        value=True,
        help="Apply StandardScaler to features (recommended)"
    )
    
    encode_categorical = st.checkbox(
        "Encode Categorical Variables",
        value=True,
        help="Automatically encode categorical features"
    )

# Model algorithm selection
with st.sidebar.expander("🤖 Algorithm Selection"):
    if 'model_type' in locals() or st.session_state.get('last_model_type'):
        current_model_type = locals().get('model_type', st.session_state.get('last_model_type', 'Classification'))
        
        if current_model_type == "Classification":
            algorithm_options = ["Random Forest", "Logistic Regression", "Gradient Boosting", "Support Vector Machine"]
        else:
            algorithm_options = ["Random Forest", "Linear Regression", "Gradient Boosting", "Support Vector Regression"]
        
        selected_algorithm = st.selectbox(
            "Select Algorithm",
            algorithm_options,
            help="Choose the machine learning algorithm to use"
        )
    else:
        st.info("Select model type in main panel first")
        selected_algorithm = "Random Forest"

# Cross-validation settings
with st.sidebar.expander("🔄 Cross-Validation"):
    cv_folds = st.slider(
        "Number of CV Folds",
        min_value=2,
        max_value=10,
        value=5,
        help="Number of folds for cross-validation"
    )
    
    perform_cv = st.checkbox(
        "Enable Cross-Validation",
        value=True,
        help="Perform k-fold cross-validation for model evaluation"
    )

# Feature selection options
with st.sidebar.expander("🎯 Feature Engineering"):
    auto_feature_selection = st.checkbox(
        "Auto Feature Selection",
        value=False,
        help="Automatically remove low-importance features (< 1% importance)"
    )
    
    feature_importance_threshold = st.slider(
        "Importance Threshold (%)",
        min_value=0.1,
        max_value=10.0,
        value=1.0,
        step=0.1,
        help="Features below this importance will be removed if auto-selection is enabled",
        disabled=not auto_feature_selection
    )
    
    polynomial_features = st.checkbox(
        "Add Polynomial Features",
        value=False,
        help="Create interaction terms (Warning: increases features significantly)"
    )
    
    if polynomial_features:
        poly_degree = st.slider(
            "Polynomial Degree",
            min_value=2,
            max_value=3,
            value=2,
            help="Degree of polynomial features"
        )

# Visualization options
with st.sidebar.expander("📊 Visualization Options"):
    show_correlation_heatmap = st.checkbox(
        "Show Correlation Heatmap",
        value=True
    )
    
    correlation_threshold = st.slider(
        "Correlation Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.8,
        step=0.05,
        help="Highlight correlations above this threshold"
    )
    
    max_features_plot = st.slider(
        "Max Features in Importance Plot",
        min_value=5,
        max_value=30,
        value=15,
        help="Number of top features to show in importance plot"
    )
    
    plot_style = st.selectbox(
        "Plot Style",
        ["default", "seaborn", "ggplot", "bmh", "dark_background"],
        help="Matplotlib style for plots"
    )

# Export options
with st.sidebar.expander("💾 Export Options"):
    include_plots_in_html = st.checkbox(
        "Include All Plots in HTML",
        value=True,
        help="Embed all visualizations in HTML report"
    )
    
    save_predictions_auto = st.checkbox(
        "Auto-save Predictions",
        value=False,
        help="Automatically download predictions as CSV"
    )

st.sidebar.markdown("---")

# Model information display (if model exists)
if st.session_state.get('model_trained', False):
    st.sidebar.subheader("📈 Current Model Info")
    st.sidebar.success("✅ Model Trained")
    if 'last_model_type' in st.session_state:
        st.sidebar.info(f"Type: {st.session_state['last_model_type']}")
    
    # Add model download button in sidebar
    st.sidebar.markdown("---")
    st.sidebar.subheader("💾 Download Model")
    
    if 'model_package' in st.session_state:
        try:
            # Serialize the model package
            model_bytes = BytesIO()
            joblib.dump(st.session_state.model_package, model_bytes)
            model_bytes.seek(0)
            
            # Create timestamp for filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            model_filename = f"ML_Model_{st.session_state['last_model_type']}_{timestamp}.pkl"
            
            st.sidebar.download_button(
                label="💾 Download Model",
                data=model_bytes.getvalue(),
                file_name=model_filename,
                mime="application/octet-stream",
                help="Download trained model package",
                key="sidebar_model_download"
            )
            
            st.sidebar.caption("Model includes all preprocessors and metadata")
            
        except Exception as e:
            st.sidebar.error(f"Error: {str(e)}")
    
    # Add correlation heatmap download if available
    if 'correlation_fig' in st.session_state:
        st.sidebar.markdown("---")
        st.sidebar.subheader("📊 Download Heatmap")
        
        try:
            # Save correlation heatmap to bytes
            buf = BytesIO()
            st.session_state.correlation_fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
            buf.seek(0)
            
            st.sidebar.download_button(
                label="📊 Download Heatmap",
                data=buf.getvalue(),
                file_name=f"Correlation_Heatmap_{timestamp}.png",
                mime="image/png",
                help="Download correlation heatmap as PNG",
                key="sidebar_heatmap_download"
            )
            
            # Also offer CSV download of correlation matrix
            if 'correlation_matrix' in st.session_state:
                csv_data = st.session_state.correlation_matrix.to_csv()
                
                st.sidebar.download_button(
                    label="📄 Download as CSV",
                    data=csv_data,
                    file_name=f"Correlation_Matrix_{timestamp}.csv",
                    mime="text/csv",
                    help="Download correlation matrix as CSV",
                    key="sidebar_corr_csv_download"
                )
            
        except Exception as e:
            st.sidebar.error(f"Error: {str(e)}")

st.sidebar.markdown("---")

# Quick help
with st.sidebar.expander("❓ Quick Help"):
    st.markdown("""
    **Getting Started:**
    1. Upload your CSV data
    2. Select target variable
    3. Choose features
    4. Configure model parameters
    5. Train and evaluate
    
    **Tips:**
    - Use cross-validation for robust estimates
    - Check feature importance to understand your model
    - Save your model for later use
    - Export HTML reports for documentation
    
    **Keyboard Shortcuts:**
    - `R` - Rerun the app
    - `C` - Clear cache
    """)

# Store settings in session state for access throughout the app
if 'settings' not in st.session_state:
    st.session_state.settings = {}

st.session_state.settings.update({
    'app_mode': app_mode,
    'handle_missing': handle_missing,
    'scale_features': scale_features,
    'encode_categorical': encode_categorical,
    'selected_algorithm': selected_algorithm,
    'cv_folds': cv_folds,
    'perform_cv': perform_cv,
    'auto_feature_selection': auto_feature_selection,
    'feature_importance_threshold': feature_importance_threshold,
    'polynomial_features': polynomial_features if 'polynomial_features' in locals() else False,
    'poly_degree': poly_degree if 'poly_degree' in locals() else 2,
    'show_correlation_heatmap': show_correlation_heatmap,
    'correlation_threshold': correlation_threshold,
    'max_features_plot': max_features_plot,
    'plot_style': plot_style,
    'include_plots_in_html': include_plots_in_html,
    'save_predictions_auto': save_predictions_auto
})

# Initialize session state for data persistence
if 'data' not in st.session_state:
    st.session_state.data = None
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'last_model_type' not in st.session_state:
    st.session_state.last_model_type = None

# Data Upload Section
st.header("1️⃣ Data Upload")

# Check app mode from sidebar
if st.session_state.settings.get('app_mode') == "Use Saved Model":
    st.info("👈 You are in 'Use Saved Model' mode. Scroll down to the prediction section or switch to 'Train New Model' in the sidebar.")
    # Jump to prediction section indicator
    st.markdown("### Skip to [Make Predictions](#make-predictions-with-saved-model) ⬇️")

uploaded_file = st.file_uploader(
    "Upload your CSV file",
    type=['csv'],
    help="Upload a CSV file containing your dataset"
)

if uploaded_file is not None:
    # Load the data
    st.session_state.data = pd.read_csv(uploaded_file)
    df = st.session_state.data
    
    st.success(f"✅ Data loaded successfully! Shape: {df.shape}")
    
    # Display data preview
    with st.expander("📊 Data Preview", expanded=True):
        st.dataframe(df.head(10))
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Rows", df.shape[0])
        with col2:
            st.metric("Columns", df.shape[1])
        with col3:
            st.metric("Missing Values", df.isnull().sum().sum())
    
    # Display data types and statistics
    with st.expander("📈 Data Statistics"):
        st.subheader("Data Types")
        st.write(df.dtypes)
        
        st.subheader("Descriptive Statistics")
        st.write(df.describe())
    
    # Feature Selection Section
    st.header("2️⃣ Feature Selection")
    
    # Identify numeric and categorical columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Target variable selection
        target_column = st.selectbox(
            "Select Target Variable",
            options=df.columns.tolist(),
            help="Choose the column you want to predict"
        )
    
    with col2:
        # Model type selection
        # Auto-detect model type based on target variable
        if target_column in categorical_cols or df[target_column].nunique() < 10:
            default_model = "Classification"
        else:
            default_model = "Regression"
        
        model_type = st.selectbox(
            "Select Model Type",
            options=["Classification", "Regression"],
            index=0 if default_model == "Classification" else 1,
            help="Choose between classification or regression"
        )
        
        # Store model type in session state
        st.session_state.last_model_type = model_type
    
    # Feature selection (exclude target and non-predictive columns)
    available_features = [col for col in df.columns if col != target_column]
    
    # Auto-exclude ID columns if they exist
    id_cols = [col for col in available_features if 'id' in col.lower()]
    recommended_features = [col for col in available_features if col not in id_cols]
    
    selected_features = st.multiselect(
        "Select Feature Columns",
        options=available_features,
        default=recommended_features,
        help="Choose the features to use for prediction"
    )
    
    if len(selected_features) == 0:
        st.warning("⚠️ Please select at least one feature column.")
    else:
        # Display correlation heatmap
        if st.session_state.settings.get('show_correlation_heatmap', True):
            with st.expander("🔥 Feature Correlation Heatmap"):
                fig, ax = plt.subplots(figsize=(12, 10))
                
                # Apply plot style from sidebar
                plt.style.use(st.session_state.settings.get('plot_style', 'default'))
                
                # Select only numeric features for correlation
                numeric_features = [f for f in selected_features if f in numeric_cols]
                
                # Only include target if it's numeric
                if target_column in numeric_cols:
                    correlation_columns = numeric_features + [target_column]
                else:
                    correlation_columns = numeric_features
                
                if len(correlation_columns) > 1:
                    correlation_matrix = df[correlation_columns].corr()
                    
                    # Highlight high correlations based on threshold
                    threshold = st.session_state.settings.get('correlation_threshold', 0.8)
                    
                    sns.heatmap(
                        correlation_matrix,
                        annot=True,
                        fmt='.2f',
                        cmap='coolwarm',
                        center=0,
                        ax=ax,
                        vmin=-1,
                        vmax=1
                    )
                    plt.title('Feature Correlation Matrix')
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Store correlation heatmap in session state for download
                    st.session_state.correlation_fig = fig
                    st.session_state.correlation_matrix = correlation_matrix
                    
                    plt.close(fig)
                    
                    # Highlight highly correlated features
                    high_corr = []
                    for i in range(len(correlation_matrix.columns)):
                        for j in range(i+1, len(correlation_matrix.columns)):
                            if abs(correlation_matrix.iloc[i, j]) > threshold:
                                high_corr.append({
                                    'Feature 1': correlation_matrix.columns[i],
                                    'Feature 2': correlation_matrix.columns[j],
                                    'Correlation': correlation_matrix.iloc[i, j]
                                })
                    
                    if high_corr:
                        st.warning(f"⚠️ Found {len(high_corr)} feature pairs with correlation > {threshold}")
                        st.dataframe(pd.DataFrame(high_corr))
                else:
                    st.info("Need at least 2 numeric features to display correlation matrix")
        
        # Model Configuration Section
        st.header("3️⃣ Model Configuration")
        
        with st.expander("⚙️ Hyperparameters", expanded=True):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                n_estimators = st.slider(
                    "Number of Trees (n_estimators)",
                    min_value=10,
                    max_value=500,
                    value=100,
                    step=10,
                    help="Number of trees in the random forest"
                )
                
                max_depth = st.slider(
                    "Maximum Depth",
                    min_value=1,
                    max_value=50,
                    value=10,
                    help="Maximum depth of each tree (None = unlimited)"
                )
                if max_depth == 50:
                    max_depth = None
            
            with col2:
                min_samples_split = st.slider(
                    "Min Samples Split",
                    min_value=2,
                    max_value=20,
                    value=2,
                    help="Minimum samples required to split an internal node"
                )
                
                min_samples_leaf = st.slider(
                    "Min Samples Leaf",
                    min_value=1,
                    max_value=20,
                    value=1,
                    help="Minimum samples required at a leaf node"
                )
            
            with col3:
                test_size = st.slider(
                    "Test Set Size",
                    min_value=0.1,
                    max_value=0.5,
                    value=0.2,
                    step=0.05,
                    help="Proportion of data to use for testing"
                )
                
                random_state = st.number_input(
                    "Random State",
                    min_value=0,
                    max_value=999,
                    value=42,
                    help="Random seed for reproducibility"
                )
        
        # Train Model Button
        if st.button("🚀 Train Model", type="primary"):
            with st.spinner("Training model... Please wait."):
                try:
                    # Prepare the data
                    X = df[selected_features].copy()
                    y = df[target_column].copy()
                    
                    # Handle missing values
                    X = X.fillna(X.mean(numeric_only=True))
                    
                    # Encode categorical features
                    label_encoders = {}
                    for col in X.columns:
                        if X[col].dtype == 'object':
                            le = LabelEncoder()
                            X[col] = le.fit_transform(X[col].astype(str))
                            label_encoders[col] = le
                    
                    # Encode target variable if classification
                    if model_type == "Classification":
                        if y.dtype == 'object':
                            target_encoder = LabelEncoder()
                            y = target_encoder.fit_transform(y)
                        else:
                            target_encoder = None
                    else:
                        target_encoder = None
                    
                    # Split the data
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y,
                        test_size=test_size,
                        random_state=random_state,
                        stratify=y if model_type == "Classification" else None
                    )
                    
                    # Scale features
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_test_scaled = scaler.transform(X_test)
                    
                    # Train the model
                    if model_type == "Classification":
                        model = RandomForestClassifier(
                            n_estimators=n_estimators,
                            max_depth=max_depth,
                            min_samples_split=min_samples_split,
                            min_samples_leaf=min_samples_leaf,
                            random_state=random_state,
                            n_jobs=-1
                        )
                    else:
                        model = RandomForestRegressor(
                            n_estimators=n_estimators,
                            max_depth=max_depth,
                            min_samples_split=min_samples_split,
                            min_samples_leaf=min_samples_leaf,
                            random_state=random_state,
                            n_jobs=-1
                        )
                    
                    model.fit(X_train_scaled, y_train)
                    
                    # Make predictions
                    y_pred = model.predict(X_test_scaled)
                    
                    if model_type == "Classification":
                        y_pred_proba = model.predict_proba(X_test_scaled)
                    
                    st.session_state.model_trained = True
                    st.success("✅ Model trained successfully!")
                    
                    # Store model package in session state for sidebar download
                    st.session_state.model_package = {
                        'model': model,
                        'scaler': scaler,
                        'label_encoders': label_encoders,
                        'target_encoder': target_encoder,
                        'feature_names': selected_features,
                        'selected_features': selected_features,
                        'model_type': model_type,
                        'config': {
                            'n_estimators': n_estimators,
                            'max_depth': max_depth,
                            'min_samples_split': min_samples_split,
                            'min_samples_leaf': min_samples_leaf,
                            'random_state': random_state
                        },
                        'target_column': target_column,
                        'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    }
                    
                    # Initialize report data dictionary for HTML report generation
                    report_data = {
                        'model_type': model_type,
                        'target_column': target_column,
                        'total_samples': len(df),
                        'train_samples': len(X_train),
                        'test_samples': len(X_test),
                        'num_features': len(selected_features),
                        'selected_features': selected_features,
                        'config': {
                            'n_estimators': n_estimators,
                            'max_depth': max_depth if max_depth is not None else 'None (unlimited)',
                            'min_samples_split': min_samples_split,
                            'min_samples_leaf': min_samples_leaf,
                            'random_state': random_state,
                            'test_size': test_size
                        }
                    }
                    
                    # Results Section
                    st.header("4️⃣ Model Evaluation Results")
                    
                    # Classification Metrics
                    if model_type == "Classification":
                        st.subheader("📊 Classification Metrics")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            accuracy = accuracy_score(y_test, y_pred)
                            st.metric("Accuracy", f"{accuracy:.4f}")
                        
                        with col2:
                            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                            st.metric("Precision", f"{precision:.4f}")
                        
                        with col3:
                            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                            st.metric("Recall", f"{recall:.4f}")
                        
                        with col4:
                            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                            st.metric("F1 Score", f"{f1:.4f}")
                        
                        # Store metrics for HTML report
                        metrics_html = f"""
        <div class="section-header">
            <h2>📊 Performance Metrics</h2>
        </div>
        <div class="row">
            <div class="col-md-3">
                <div class="metric-card text-center">
                    <div class="metric-label">Accuracy</div>
                    <div class="metric-value">{accuracy:.4f}</div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="metric-card text-center">
                    <div class="metric-label">Precision</div>
                    <div class="metric-value">{precision:.4f}</div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="metric-card text-center">
                    <div class="metric-label">Recall</div>
                    <div class="metric-value">{recall:.4f}</div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="metric-card text-center">
                    <div class="metric-label">F1 Score</div>
                    <div class="metric-value">{f1:.4f}</div>
                </div>
            </div>
        </div>
        """
                        report_data['metrics_html'] = metrics_html
                        
                        # Confusion Matrix
                        st.subheader("🎯 Confusion Matrix")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            fig, ax = plt.subplots(figsize=(8, 6))
                            cm = confusion_matrix(y_test, y_pred)
                            sns.heatmap(
                                cm,
                                annot=True,
                                fmt='d',
                                cmap='Blues',
                                ax=ax
                            )
                            
                            if target_encoder is not None:
                                labels = target_encoder.classes_
                                ax.set_xticklabels(labels)
                                ax.set_yticklabels(labels)
                            
                            plt.title('Confusion Matrix')
                            plt.ylabel('True Label')
                            plt.xlabel('Predicted Label')
                            plt.tight_layout()
                            st.pyplot(fig)
                            
                            # Store confusion matrix for HTML report
                            cm_img = fig_to_base64(fig)
                            plt.close(fig)
                        
                        with col2:
                            # Classification Report
                            st.text("Classification Report:")
                            
                            if target_encoder is not None:
                                target_names = target_encoder.classes_
                            else:
                                target_names = None
                            
                            report = classification_report(
                                y_test,
                                y_pred,
                                target_names=target_names
                            )
                            st.text(report)
                            
                            # Store classification report for HTML
                            report_data['classification_report'] = report
                        
                        # ROC Curve (for binary classification)
                        if len(np.unique(y)) == 2:
                            st.subheader("📈 ROC Curve")
                            
                            fig, ax = plt.subplots(figsize=(10, 6))
                            
                            # Calculate ROC curve and AUC
                            fpr, tpr, _ = roc_curve(y_test, y_pred_proba[:, 1])
                            roc_auc = auc(fpr, tpr)
                            
                            # Plot ROC curve
                            plt.plot(
                                fpr, tpr,
                                color='darkorange',
                                lw=2,
                                label=f'ROC curve (AUC = {roc_auc:.2f})'
                            )
                            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
                            plt.xlim([0.0, 1.0])
                            plt.ylim([0.0, 1.05])
                            plt.xlabel('False Positive Rate')
                            plt.ylabel('True Positive Rate')
                            plt.title('Receiver Operating Characteristic (ROC) Curve')
                            plt.legend(loc="lower right")
                            plt.grid(alpha=0.3)
                            plt.tight_layout()
                            st.pyplot(fig)
                            
                            st.metric("AUC-ROC Score", f"{roc_auc:.4f}")
                            
                            # Store ROC curve for HTML report
                            roc_img = fig_to_base64(fig)
                            plt.close(fig)
                            
                            # Build visualizations HTML for classification
                            visualizations_html = f"""
        <div class="section-header">
            <h2>📊 Visualizations</h2>
        </div>
        <div class="row">
            <div class="col-md-6">
                <div class="img-container">
                    <h4>Confusion Matrix</h4>
                    <img src="data:image/png;base64,{cm_img}" alt="Confusion Matrix">
                </div>
            </div>
            <div class="col-md-6">
                <div class="img-container">
                    <h4>ROC Curve (AUC = {roc_auc:.4f})</h4>
                    <img src="data:image/png;base64,{roc_img}" alt="ROC Curve">
                </div>
            </div>
        </div>
        <div class="section-header">
            <h2>📋 Classification Report</h2>
        </div>
        <div class="card">
            <div class="card-body">
                <pre>{report_data['classification_report']}</pre>
            </div>
        </div>
        """
                            report_data['visualizations_html'] = visualizations_html
                        else:
                            # Multi-class classification (no ROC curve)
                            visualizations_html = f"""
        <div class="section-header">
            <h2>📊 Visualizations</h2>
        </div>
        <div class="row">
            <div class="col-md-12">
                <div class="img-container">
                    <h4>Confusion Matrix</h4>
                    <img src="data:image/png;base64,{cm_img}" alt="Confusion Matrix">
                </div>
            </div>
        </div>
        <div class="section-header">
            <h2>📋 Classification Report</h2>
        </div>
        <div class="card">
            <div class="card-body">
                <pre>{report_data['classification_report']}</pre>
            </div>
        </div>
        """
                            report_data['visualizations_html'] = visualizations_html
                    
                    # Regression Metrics
                    else:
                        st.subheader("📊 Regression Metrics")
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            mse = mean_squared_error(y_test, y_pred)
                            st.metric("Mean Squared Error", f"{mse:.4f}")
                        
                        with col2:
                            mae = mean_absolute_error(y_test, y_pred)
                            st.metric("Mean Absolute Error", f"{mae:.4f}")
                        
                        with col3:
                            r2 = r2_score(y_test, y_pred)
                            st.metric("R² Score", f"{r2:.4f}")
                        
                        # Store metrics for HTML report
                        metrics_html = f"""
        <div class="section-header">
            <h2>📊 Performance Metrics</h2>
        </div>
        <div class="row">
            <div class="col-md-4">
                <div class="metric-card text-center">
                    <div class="metric-label">Mean Squared Error</div>
                    <div class="metric-value">{mse:.4f}</div>
                </div>
            </div>
            <div class="col-md-4">
                <div class="metric-card text-center">
                    <div class="metric-label">Mean Absolute Error</div>
                    <div class="metric-value">{mae:.4f}</div>
                </div>
            </div>
            <div class="col-md-4">
                <div class="metric-card text-center">
                    <div class="metric-label">R² Score</div>
                    <div class="metric-value">{r2:.4f}</div>
                </div>
            </div>
        </div>
        """
                        report_data['metrics_html'] = metrics_html
                        
                        # Residual Plot
                        st.subheader("📈 Residual Plot")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            fig, ax = plt.subplots(figsize=(8, 6))
                            residuals = y_test - y_pred
                            plt.scatter(y_pred, residuals, alpha=0.5)
                            plt.axhline(y=0, color='r', linestyle='--')
                            plt.xlabel('Predicted Values')
                            plt.ylabel('Residuals')
                            plt.title('Residual Plot')
                            plt.grid(alpha=0.3)
                            plt.tight_layout()
                            st.pyplot(fig)
                            
                            # Store residual plot for HTML report
                            residual_img = fig_to_base64(fig)
                            plt.close(fig)
                        
                        with col2:
                            fig, ax = plt.subplots(figsize=(8, 6))
                            plt.scatter(y_test, y_pred, alpha=0.5)
                            plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
                            plt.xlabel('True Values')
                            plt.ylabel('Predicted Values')
                            plt.title('Predicted vs Actual')
                            plt.grid(alpha=0.3)
                            plt.tight_layout()
                            st.pyplot(fig)
                            
                            # Store predicted vs actual plot for HTML report
                            pred_actual_img = fig_to_base64(fig)
                            plt.close(fig)
                        
                        # Build visualizations HTML for regression
                        visualizations_html = f"""
        <div class="section-header">
            <h2>📊 Visualizations</h2>
        </div>
        <div class="row">
            <div class="col-md-6">
                <div class="img-container">
                    <h4>Residual Plot</h4>
                    <img src="data:image/png;base64,{residual_img}" alt="Residual Plot">
                </div>
            </div>
            <div class="col-md-6">
                <div class="img-container">
                    <h4>Predicted vs Actual</h4>
                    <img src="data:image/png;base64,{pred_actual_img}" alt="Predicted vs Actual">
                </div>
            </div>
        </div>
        """
                        report_data['visualizations_html'] = visualizations_html
                    
                    # Feature Importance
                    st.subheader("🎯 Feature Importance")
                    
                    # Get feature importance
                    feature_importance = pd.DataFrame({
                        'feature': selected_features,
                        'importance': model.feature_importances_
                    }).sort_values('importance', ascending=False)
                    
                    # Apply auto feature selection if enabled
                    if st.session_state.settings.get('auto_feature_selection', False):
                        threshold_pct = st.session_state.settings.get('feature_importance_threshold', 1.0)
                        threshold_val = threshold_pct / 100.0
                        low_importance = feature_importance[feature_importance['importance'] < threshold_val]
                        
                        if len(low_importance) > 0:
                            st.warning(f"⚠️ Auto-selection would remove {len(low_importance)} features with importance < {threshold_pct}%")
                            with st.expander("View low-importance features"):
                                st.dataframe(low_importance)
                    
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        # Get max features from sidebar setting
                        max_features = st.session_state.settings.get('max_features_plot', 15)
                        
                        fig, ax = plt.subplots(figsize=(10, 6))
                        plt.style.use(st.session_state.settings.get('plot_style', 'default'))
                        
                        plt.barh(
                            feature_importance['feature'][:max_features],
                            feature_importance['importance'][:max_features]
                        )
                        plt.xlabel('Importance')
                        plt.title(f'Top {max_features} Feature Importances')
                        plt.gca().invert_yaxis()
                        plt.tight_layout()
                        st.pyplot(fig)
                        
                        # Store feature importance plot for HTML report
                        feature_imp_img = fig_to_base64(fig)
                        plt.close(fig)
                    
                    with col2:
                        st.dataframe(
                            feature_importance,
                            height=400
                        )
                    
                    # Build feature importance HTML
                    feature_imp_table_rows = ''.join([
                        f"<tr><td>{row['feature']}</td><td>{row['importance']:.4f}</td></tr>"
                        for _, row in feature_importance.head(10).iterrows()
                    ])
                    
                    report_data['feature_importance_plot'] = f'<div class="img-container"><img src="data:image/png;base64,{feature_imp_img}" alt="Feature Importance"></div>'
                    report_data['feature_importance_table'] = feature_imp_table_rows
                    
                    # Cross-validation scores
                    st.subheader("🔄 Cross-Validation Scores")
                    
                    # Check if CV is enabled in sidebar
                    if st.session_state.settings.get('perform_cv', True):
                        cv_folds = st.session_state.settings.get('cv_folds', 5)
                        
                        with st.spinner(f"Performing {cv_folds}-fold cross-validation..."):
                            cv_scores = cross_val_score(
                                model,
                                X_train_scaled,
                                y_train,
                                cv=cv_folds,
                                n_jobs=-1
                            )
                            
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric("Mean CV Score", f"{cv_scores.mean():.4f}")
                            with col2:
                                st.metric("Std CV Score", f"{cv_scores.std():.4f}")
                            with col3:
                                st.metric("Min CV Score", f"{cv_scores.min():.4f}")
                            
                            st.info(f"📊 {cv_folds}-Fold Cross-Validation Scores: {cv_scores.round(4)}")
                    else:
                        st.info("Cross-validation is disabled. Enable it in the sidebar settings.")
                    
                    # Generate HTML Report
                    st.subheader("📄 Download HTML Report")
                    
                    st.info("""
                    Download a comprehensive HTML report with all visualizations and metrics.
                    
                    💡 **Tip:** The trained model can be downloaded from the sidebar (left panel) under "Download Model"
                    """)
                    
                    try:
                        # Generate the complete HTML report
                        html_report = generate_html_report(report_data)
                        
                        # Create download button
                        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                        html_filename = f"ML_Report_{model_type}_{timestamp}.html"
                        
                        st.download_button(
                            label="📥 Download HTML Report",
                            data=html_report,
                            file_name=html_filename,
                            mime="text/html",
                            help="Download a comprehensive HTML report with all visualizations and metrics",
                            key="html_download_btn"
                        )
                        
                        st.success("✅ HTML report ready for download!")
                        
                    except Exception as e:
                        st.error(f"Error generating HTML report: {str(e)}")
                    
                    # Model Package Contents Information
                    st.subheader("💾 Model Package Contents")
                    
                    st.markdown("""
                    The saved model package includes:
                    - Trained Random Forest model
                    - Feature scaler (StandardScaler)
                    - Label encoders (if applicable)
                    - Target encoder (for classification)
                    - Feature names
                    - Model configuration
                    """)
                    
                    # Display model usage instructions
                    with st.expander("📖 How to Use the Saved Model"):
                        st.markdown("""
                        ### Loading and Using Your Saved Model
                        
                        ```python
                        import joblib
                        import pandas as pd
                        
                        # Load the model package
                        model_package = joblib.load('ML_Model_Classification_20240211_143022.joblib')
                        
                        # Extract components
                        model = model_package['model']
                        scaler = model_package['scaler']
                        feature_names = model_package['feature_names']
                        
                        # Load new data
                        new_data = pd.read_csv('new_data.csv')
                        
                        # Prepare features (must match training features)
                        X_new = new_data[feature_names]
                        
                        # Handle categorical encoding if needed
                        label_encoders = model_package['label_encoders']
                        for col, encoder in label_encoders.items():
                            if col in X_new.columns:
                                X_new[col] = encoder.transform(X_new[col].astype(str))
                        
                        # Scale features
                        X_new_scaled = scaler.transform(X_new)
                        
                        # Make predictions
                        predictions = model.predict(X_new_scaled)
                        
                        # For classification, get probabilities
                        if model_package['model_type'] == 'Classification':
                            probabilities = model.predict_proba(X_new_scaled)
                            
                            # Decode predictions if target was encoded
                            if model_package['target_encoder'] is not None:
                                predictions = model_package['target_encoder'].inverse_transform(predictions)
                        
                        print("Predictions:", predictions)
                        ```
                        
                        ### Important Notes:
                        - New data must have the **exact same features** as the training data
                        - Feature order doesn't matter, but names must match
                        - Categorical variables will be encoded automatically using the saved encoders
                        - Missing values should be handled before prediction
                        """)
                    
                    st.info("""
                    Save your trained model to use later for making predictions on new data. 
                    The model package includes:
                    - Trained model
                    - Feature scaler
                    - Encoders for categorical variables
                    - Model configuration and metadata
                    """)
                    
                    try:
                        # Create model package
                        model_bytes = save_model_package(
                            model=model,
                            scaler=scaler,
                            label_encoders=label_encoders,
                            target_encoder=target_encoder,
                            config={
                                'n_estimators': n_estimators,
                                'max_depth': max_depth,
                                'min_samples_split': min_samples_split,
                                'min_samples_leaf': min_samples_leaf,
                                'random_state': random_state,
                                'test_size': test_size
                            },
                            selected_features=selected_features,
                            target_column=target_column,
                            model_type=model_type
                        )
                        
                        model_filename = f"ML_Model_{model_type}_{timestamp}.pkl"
                        
                        st.download_button(
                            label="💾 Download Trained Model",
                            data=model_bytes,
                            file_name=model_filename,
                            mime="application/octet-stream",
                            help="Download the trained model package for later use"
                        )
                        
                        st.success("✅ Model package ready for download!")
                        
                    except Exception as e:
                        st.error(f"Error saving model: {str(e)}")
                        st.exception(e)
                
                except Exception as e:
                    st.error(f"❌ Error training model: {str(e)}")
                    st.exception(e)

else:
    st.info("👆 Please upload a CSV file to get started.")
    
    # Show example of expected format
    st.markdown("---")
    st.subheader("📋 Expected Data Format")
    st.markdown("""
    Your CSV file should contain:
    - One target column (the variable you want to predict)
    - Multiple feature columns (predictors)
    - Headers in the first row
    - Numeric or categorical data
    
    Example:
    ```
    id,diagnosis,feature1,feature2,feature3
    1,M,17.99,10.38,122.8
    2,B,20.57,17.77,132.9
    ```
    """)

# Prediction Section - Use Saved Model
st.markdown("---")
st.header("🔮 Make Predictions with Saved Model")

st.markdown("""
Upload a previously saved model and new data to make predictions.
""")

col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Upload Saved Model")
    uploaded_model = st.file_uploader(
        "Upload Model Package (.joblib or .pkl)",
        type=['pkl', 'joblib'],
        help="Upload a model that was previously saved from this application",
        key="model_uploader"
    )

with col2:
    st.subheader("2. Upload New Data")
    uploaded_prediction_data = st.file_uploader(
        "Upload CSV for Predictions",
        type=['csv'],
        help="Upload new data with the same features as the training data",
        key="prediction_data_uploader"
    )

if uploaded_model is not None and uploaded_prediction_data is not None:
    try:
        # Load the model package
        with st.spinner("Loading model..."):
            model_package = load_model_package(uploaded_model)
        
        st.success("✅ Model loaded successfully!")
        
        # Display model information
        with st.expander("📊 Model Information", expanded=True):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Model Type", model_package['model_type'])
            with col2:
                st.metric("Target Variable", model_package['target_column'])
            with col3:
                st.metric("Features Used", len(model_package['selected_features']))
            
            st.write("**Training Date:**", model_package['timestamp'])
            st.write("**Required Features:**")
            st.write(model_package['selected_features'])
        
        # Load prediction data
        prediction_data = pd.read_csv(uploaded_prediction_data)
        
        st.subheader("Preview of Uploaded Data")
        st.dataframe(prediction_data.head(10))
        
        # Make predictions button
        if st.button("🚀 Make Predictions", type="primary"):
            with st.spinner("Making predictions..."):
                try:
                    predictions, pred_proba = make_predictions(model_package, prediction_data)
                    
                    # Create results dataframe
                    results_df = prediction_data.copy()
                    results_df['Prediction'] = predictions
                    
                    # Add probabilities for classification
                    if model_package['model_type'] == "Classification" and pred_proba is not None:
                        # Get class labels
                        if model_package['target_encoder'] is not None:
                            classes = model_package['target_encoder'].classes_
                        else:
                            classes = [f"Class_{i}" for i in range(pred_proba.shape[1])]
                        
                        # Add probability columns
                        for i, class_name in enumerate(classes):
                            results_df[f'Probability_{class_name}'] = pred_proba[:, i]
                    
                    st.success(f"✅ Predictions complete! Generated {len(predictions)} predictions.")
                    
                    # Display results
                    st.subheader("📊 Prediction Results")
                    st.dataframe(results_df)
                    
                    # Summary statistics
                    st.subheader("📈 Prediction Summary")
                    
                    if model_package['model_type'] == "Classification":
                        # Show distribution of predictions
                        pred_counts = pd.Series(predictions).value_counts()
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write("**Prediction Distribution:**")
                            st.dataframe(pred_counts)
                        
                        with col2:
                            fig, ax = plt.subplots(figsize=(8, 6))
                            pred_counts.plot(kind='bar', ax=ax)
                            plt.title('Distribution of Predictions')
                            plt.xlabel('Predicted Class')
                            plt.ylabel('Count')
                            plt.xticks(rotation=45)
                            plt.tight_layout()
                            st.pyplot(fig)
                            plt.close(fig)
                    else:
                        # Show statistics for regression
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Mean Prediction", f"{np.mean(predictions):.4f}")
                        with col2:
                            st.metric("Std Deviation", f"{np.std(predictions):.4f}")
                        with col3:
                            st.metric("Min Prediction", f"{np.min(predictions):.4f}")
                        with col4:
                            st.metric("Max Prediction", f"{np.max(predictions):.4f}")
                        
                        # Histogram of predictions
                        fig, ax = plt.subplots(figsize=(10, 6))
                        plt.hist(predictions, bins=30, edgecolor='black', alpha=0.7)
                        plt.title('Distribution of Predictions')
                        plt.xlabel('Predicted Value')
                        plt.ylabel('Frequency')
                        plt.grid(alpha=0.3)
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close(fig)
                    
                    # Download results
                    st.subheader("💾 Download Predictions")
                    
                    # Convert to CSV
                    csv = results_df.to_csv(index=False)
                    
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    
                    st.download_button(
                        label="📥 Download Predictions as CSV",
                        data=csv,
                        file_name=f"Predictions_{timestamp}.csv",
                        mime="text/csv",
                        help="Download predictions as a CSV file"
                    )
                    
                except Exception as e:
                    st.error(f"❌ Error making predictions: {str(e)}")
                    st.exception(e)
                    
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.exception(e)

elif uploaded_model is not None or uploaded_prediction_data is not None:
    st.info("⚠️ Please upload both a model file and prediction data to continue.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Built with ❤️ using Streamlit | Machine Learning Made Simple</p>
</div>
""", unsafe_allow_html=True)
