# Machine Learning Model Builder - Streamlit Application

## Overview
This is a comprehensive machine learning application built with Streamlit that allows you to:
- Upload CSV datasets
- Select features for modeling
- Choose between Classification and Regression
- Configure model hyperparameters with sensible defaults
- Evaluate models with various metrics including confusion matrices, ROC curves, and correlations
- **Export complete analysis as professional HTML reports using Bootstrap 5**
- **Save trained models for later use**
- **Make predictions on new data using saved models**

## Features

### Data Handling
- **CSV Upload**: Easy file upload interface
- **Data Preview**: View your data structure and statistics
- **Missing Value Detection**: Automatic identification of data quality issues
- **Feature Correlation**: Visual correlation heatmap

### Model Configuration
- **Model Type Selection**: Choose between Classification or Regression
- **Algorithm**: Random Forest (both classifier and regressor)
- **Hyperparameter Tuning**: Adjust all major parameters with sliders
  - Number of trees (n_estimators)
  - Maximum depth
  - Minimum samples split
  - Minimum samples leaf
  - Test set size
  - Random state for reproducibility

### Sidebar Configuration Panel (NEW!)
The left sidebar now provides comprehensive configuration options:

**Application Mode:**
- Train New Model
- Use Saved Model (quick switch)

**Data Preprocessing:**
- Missing value strategy (Mean/Median/Mode/Drop)
- Feature scaling toggle
- Categorical encoding options

**Algorithm Selection:**
- Choose from multiple algorithms:
  - Classification: Random Forest, Logistic Regression, Gradient Boosting, SVM
  - Regression: Random Forest, Linear Regression, Gradient Boosting, SVR

**Cross-Validation:**
- Configurable number of folds (2-10)
- Enable/disable CV
- Performance estimates with confidence

**Feature Engineering:**
- Auto feature selection by importance threshold
- Polynomial feature generation
- Interaction term creation

**Visualization Options:**
- Toggle correlation heatmap
- Correlation threshold for highlighting
- Max features in importance plots (5-30)
- Plot style selection (5 themes)

**Export Options:**
- Control HTML report contents
- Auto-save predictions toggle

### Evaluation Metrics

#### For Classification:
- Accuracy, Precision, Recall, F1 Score
- Confusion Matrix with heatmap visualization
- Classification Report
- ROC Curve with AUC score (for binary classification)
- Feature Importance rankings
- Cross-validation scores

#### For Regression:
- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- R² Score
- Residual plots
- Predicted vs Actual scatter plots
- Feature Importance rankings
- Cross-validation scores

### HTML Report Generation
- **Professional HTML5 reports** with Bootstrap 5 styling
- **Embedded visualizations** - all charts and plots included as base64 images
- **Responsive design** - looks great on desktop, tablet, and mobile
- **Complete analysis** - includes all metrics, configurations, and visualizations
- **Standalone file** - no external dependencies, can be shared easily
- **Beautiful layout** - modern, clean design with gradient headers and cards

### Model Persistence (NEW!)
- **Save trained models** - download complete model packages (.joblib format)
- **Load saved models** - reuse models on new data without retraining
- **Complete packages** - includes model, scalers, encoders, and metadata
- **Three usage options**:
  1. Streamlit app interface (GUI mode)
  2. Command-line script (batch processing)
  3. Python code integration (programmatic use)
- **Prediction outputs** - CSV files with predictions and probabilities
- **Model metadata** - track training date, features, and configuration

### Model Persistence & Prediction (NEW!)
- **Save trained models** - Download complete model packages (.pkl files)
- **Model package includes**:
  - Trained model
  - Feature scaler
  - Label encoders for categorical variables
  - Target encoder (for classification)
  - Model configuration and metadata
  - Timestamp of training
- **Load and predict** - Upload saved models to make predictions on new data
- **Batch predictions** - Process entire CSV files at once
- **Prediction export** - Download results with probabilities (classification) or values (regression)
- **Model validation** - Automatic checking of feature compatibility

## Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)

### Setup Instructions

1. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   
   # On Windows:
   venv\Scripts\activate
   
   # On Mac/Linux:
   source venv/bin/activate
   ```

2. **Install required packages**:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Application

1. **Start the Streamlit app**:
   ```bash
   streamlit run ml_app.py
   ```

2. **Access the application**:
   - The app will automatically open in your default web browser
   - If not, navigate to: `http://localhost:8501`

## Usage Guide

### Step 1: Upload Data
- Click the "Browse files" button
- Select your CSV file
- The data will be loaded and previewed automatically

### Step 2: Select Features
- Choose your **target variable** (the column you want to predict)
- Select **feature columns** (predictor variables)
- The app auto-detects model type based on your target variable
- View the correlation heatmap to understand feature relationships

### Step 3: Configure Model
- Adjust hyperparameters using the sliders
- Default values are set to commonly used, effective settings
- All parameters have helpful tooltips explaining their purpose

### Step 4: Train and Evaluate
- Click the "🚀 Train Model" button
- Wait for training to complete
- Review comprehensive evaluation metrics
- Examine visualizations (confusion matrix, ROC curve, etc.)
- Check feature importance to understand model decisions

### Step 5: Download HTML Report (NEW!)
- After training completes, scroll to the "📄 Download HTML Report" section
- Click the "📥 Download HTML Report" button
- Save the HTML file to your computer
- Open it in any web browser to view the complete analysis
- Share the report with colleagues or include it in presentations

### Step 6: Download Model & Correlation Heatmap from Sidebar (NEW!)
After training is complete, the **left sidebar** will show new download options:

**Download Model:**
- Look for "💾 Download Model" in the sidebar
- Click to download the .pkl file containing your trained model
- This file persists and can be downloaded even after downloading the HTML report
- The model package includes:
  - Trained Random Forest model
  - Feature scaler and encoders
  - Model configuration
  - Feature names and metadata

**Download Correlation Heatmap:**
- Click "📊 Download Heatmap" for high-resolution PNG image (300 DPI)
- Click "📄 Download as CSV" for the correlation matrix data
- Perfect for publications and presentations

## Using a Saved Model

### Option 1: Using the Streamlit App

1. **Select "Use Saved Model" mode** in the sidebar (or stay in "Train New Model" mode)
2. **Scroll to** "🔮 Make Predictions with Saved Model" section at the bottom
3. **Upload your saved model** (.pkl file from sidebar download)
4. **Upload new data** (CSV file with same features)
5. **Click "Make Predictions"**
6. **View and download results** with predictions and probabilities

### Option 2: Using the Command Line Script

We provide a standalone Python script for batch predictions:

```bash
python use_saved_model.py --model my_model.joblib --data new_data.csv --output predictions.csv
```

**Arguments:**
- `--model`: Path to your saved model file
- `--data`: Path to CSV file with new data
- `--output`: Path to save predictions (default: predictions.csv)

**Example:**
```bash
python use_saved_model.py \
    --model ML_Model_Classification_20240211_143022.joblib \
    --data patient_data_new.csv \
    --output patient_predictions.csv
```

### Option 3: Using Python Code

```python
import joblib
import pandas as pd

# Load the model
model_package = joblib.load('my_model.joblib')

# Load new data
new_data = pd.read_csv('new_data.csv')

# Extract components
model = model_package['model']
scaler = model_package['scaler']
feature_names = model_package['selected_features']

# Prepare features
X_new = new_data[feature_names].copy()

# Handle missing values
X_new = X_new.fillna(X_new.mean(numeric_only=True))

# Encode categorical features if needed
for col, encoder in model_package['label_encoders'].items():
    if col in X_new.columns:
        X_new[col] = X_new[col].astype(str)
        X_new[col] = encoder.transform(X_new[col])

# Scale features
X_new_scaled = scaler.transform(X_new)

# Make predictions
predictions = model.predict(X_new_scaled)

# For classification, get probabilities
if model_package['model_type'] == 'Classification':
    probabilities = model.predict_proba(X_new_scaled)
    
    # Decode if target was encoded
    if model_package['target_encoder'] is not None:
        predictions = model_package['target_encoder'].inverse_transform(predictions)

print("Predictions:", predictions)
```

### Step 6: Save Your Model (NEW!)
- In the "💾 Save Trained Model" section, click "💾 Download Trained Model"
- Save the .pkl file to your computer
- This file contains everything needed to make predictions later

### Step 7: Make Predictions with Saved Model (NEW!)
- Scroll to the "🔮 Make Predictions with Saved Model" section
- Upload your saved model (.pkl file)
- Upload new data as CSV (must have the same features as training data)
- Click "🚀 Make Predictions"
- View prediction results and download as CSV

## Example Dataset Format

Your CSV file should be structured like this:

```csv
id,diagnosis,feature1,feature2,feature3,feature4
1,M,17.99,10.38,122.8,1001
2,B,20.57,17.77,132.9,1326
3,M,19.69,21.25,130.0,1203
```

**Requirements**:
- First row must contain column headers
- One column should be your target variable
- Other columns are features (numeric or categorical)
- Data can contain missing values (will be handled automatically)

## Technical Details

### Algorithm: Random Forest
Random Forest is chosen as the default algorithm because:
- **Robust**: Handles both numeric and categorical data
- **No scaling required**: Works well without feature normalization
- **Feature importance**: Provides insights into which features matter most
- **Overfitting resistance**: Ensemble method reduces overfitting
- **Versatile**: Works for both classification and regression

### Data Preprocessing
The application automatically:
1. Handles missing values (fills with mean for numeric columns)
2. Encodes categorical variables using LabelEncoder
3. Scales features using StandardScaler
4. Splits data into training and test sets

### Model Validation
- Train/test split with configurable ratio
- 5-fold cross-validation for robust performance estimation
- Multiple evaluation metrics for comprehensive assessment

### HTML Report Features
The HTML report includes:
- **Bootstrap 5 framework** for responsive, professional design
- **Embedded visualizations** using base64-encoded images (no external files needed)
- **Complete metrics** - all performance indicators in one place
- **Model configuration** - full documentation of parameters used
- **Feature analysis** - importance rankings and correlations
- **Self-contained** - works offline, no internet connection needed
- **Print-friendly** - formatted for professional documentation

### Model Persistence
The application uses joblib for efficient model serialization:
- **Complete package** - saves model, scalers, encoders, and metadata
- **Portable** - .pkl files can be shared and used on different systems
- **Version tracking** - includes timestamp of when model was trained
- **Safe loading** - validates features match between training and prediction
- **Handles edge cases** - manages unseen categorical values gracefully
- **Efficient** - compressed binary format for small file sizes

### Making Predictions
When using a saved model:
1. **Model validation** - Checks that new data has required features
2. **Preprocessing** - Applies same transformations as training (encoding, scaling)
3. **Prediction** - Generates predictions using the trained model
4. **Post-processing** - Decodes categorical predictions if needed
5. **Probability estimates** - Returns class probabilities for classification
6. **Export** - Saves results to CSV with predictions and probabilities

## Troubleshooting

### Common Issues

1. **Module not found errors**:
   ```bash
   pip install -r requirements.txt --upgrade
   ```

2. **Port already in use**:
   ```bash
   streamlit run ml_app.py --server.port 8502
   ```

3. **Memory errors with large datasets**:
   - Reduce the number of features
   - Use a smaller test set size
   - Reduce n_estimators in model configuration

## Best Practices

### For Classification Problems:
- Ensure your target variable has clear categories
- Use binary classification (2 classes) for ROC curve analysis
- Check for class imbalance in your data
- Aim for at least 100 samples per class

### For Regression Problems:
- Ensure target variable is continuous
- Check residual plots for patterns (should be random)
- R² score closer to 1 indicates better fit
- Consider feature scaling if features have different ranges

### Feature Selection:
- Start with all features, then remove low-importance ones
- Check correlation heatmap to identify redundant features
- Domain knowledge should guide feature selection
- Remove ID columns and non-informative features

### Model Persistence:
- **Save your models** after training for reuse
- **Document model versions** using the timestamp in filenames
- **Validate predictions** by checking they make sense for your domain
- **Keep training data** to retrain if model performance degrades
- **Test on sample data** before applying to large datasets
- **Version control** - save different model versions as you improve them

### Making Predictions:
- **Ensure feature compatibility** - new data must have same features as training
- **Check data quality** - similar distributions and ranges as training data
- **Handle missing values** - model will fill with mean, but check if appropriate
- **Review predictions** - always verify results make sense
- **Monitor performance** - if predictions seem off, retrain with new data

## Performance Tips

1. **Start with default parameters** - they work well for most cases
2. **Increase n_estimators** for better performance (slower training)
3. **Limit max_depth** to prevent overfitting on small datasets
4. **Use cross-validation scores** to assess model stability
5. **Check feature importance** to understand your model

## Limitations

- Currently supports only Random Forest algorithm
- CSV files only (no Excel, JSON, etc.)
- Memory constraints may limit very large datasets
- Binary and multi-class classification supported
- Single target variable only

## Future Enhancements (Possible Extensions)

- Additional algorithms (XGBoost, SVM, Neural Networks)
- Automated hyperparameter tuning (Grid Search, Random Search)
- Support for more file formats
- Model persistence (save/load trained models)
- Batch prediction capability
- Advanced feature engineering tools

## Support

For issues or questions:
1. Check the troubleshooting section
2. Ensure all dependencies are correctly installed
3. Verify your data format matches the expected structure
4. Check Streamlit documentation: https://docs.streamlit.io

## License

This application is provided as-is for research and educational purposes.

---

**Happy Modeling! 🚀**
