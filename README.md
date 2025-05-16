# Fuel consumption prediction


Overview of the Fuel Consumption Prediction System, an end-to-end machine learning application designed to predict vehicle fuel efficiency (MPG - Miles Per Gallon) based on various vehicle specifications. The system includes data processing, exploratory data analysis, feature engineering, model training, evaluation, and deployment capabilities.

Table of Contents
System Architecture
Data Pipeline
Model Development
Feature Engineering
Model Selection and Evaluation
Model Deployment
Web Application
Performance Monitoring
Technical Specifications
Dependencies
System Architecture
The system follows a modular architecture with separate components for:

Data ingestion and preprocessing
Feature engineering and transformation
Model training and evaluation
Model serving and prediction
Web application interface
Deployment pipeline and monitoring
Key Components
Data Processing Pipeline: Handles data cleaning, imputation, and transformation
Feature Engineering Module: Creates derived features and transforms categorical variables
Model Training System: Trains and evaluates multiple regression models
Model Serving API: Provides prediction functionality through a streamlined interface
Web Application: User-friendly interface for making predictions
Monitoring System: Tracks model performance and detects drift
Data Pipeline
Data Source
The system uses the auto-mpg dataset containing vehicle specifications from the 1970s and 1980s, with the following columns:

MPG (target variable)
Cylinders
Displacement
Horsepower
Weight
Acceleration
Model Year
Origin
Data Preprocessing
The pipeline includes:

Missing Value Treatment:
Identification of missing values (marked as "?")
Imputation of missing Horsepower values using median
Validation checks to ensure data integrity
Categorical Variable Processing:
Origin column transformation (1,2,3 → India, USA, Germany)
One-hot encoding implementation via scikit-learn's OneHotEncoder
Numerical Feature Scaling:
Standard scaling applied to all numerical features
Custom attribute creation via the CustomAttrAdder class
Data Splitting Strategy
The system implements:

Train-test split with 80-20 ratio
Stratified sampling based on the Cylinders column to ensure representative distribution
Custom StratifiedShuffleSplit implementation for cross-validation
Model Development
Model Pipeline
The system uses scikit-learn's Pipeline API to create a robust model development flow:

python
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('attrs_adder', CustomAttrAdder()),
    ('std_scaler', StandardScaler()),
])

full_pipeline = ColumnTransformer([
    ("num", num_pipeline, list(num_attrs)),
    ("cat", OneHotEncoder(sparse_output=False), cat_attrs),
])
Custom Transformers
The system includes a custom transformer for feature engineering:

python
class CustomAttrAdder(BaseEstimator, TransformerMixin):
    def __init__(self, acc_on_power=True):
        self.acc_on_power = acc_on_power
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        acc_ix, hpower_ix, cyl_ix = 4, 2, 0
        acc_on_cyl = X[:, acc_ix] / X[:, cyl_ix]
        if self.acc_on_power:
            acc_on_power = X[:, acc_ix] / X[:, hpower_ix]
            return np.c_[X, acc_on_power, acc_on_cyl]
        return np.c_[X, acc_on_cyl]
Feature Engineering
Derived Features
The system creates the following engineered features:

Displacement-to-power ratio (displacement_on_power)
Weight-to-cylinder ratio (weight_on_cylinder)
Acceleration-to-power ratio (acceleration_on_power)
Acceleration-to-cylinder ratio (acceleration_on_cyl)
Automated Feature Engineering
The system includes capabilities for automated feature engineering using featuretools:

Entity set creation based on the dataset
Deep feature synthesis with configurable depth
Feature selection based on correlation with target variable
Feature Selection
Feature importance is calculated for the Random Forest model and used for feature selection:

python
feature_importances = grid_search.best_estimator_.feature_importances_
sorted(zip(attrs, feature_importances), reverse=True)
Model Selection and Evaluation
Models Evaluated
Linear Regression
Decision Tree Regressor
Random Forest Regressor
Support Vector Regressor
Evaluation Metrics
RMSE (Root Mean Squared Error): Primary evaluation metric
Cross-validated RMSE: 10-fold cross-validation for robust performance estimation
Hyperparameter Optimization
Two methods for hyperparameter tuning:

Grid Search:
python
param_grid = [
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
]
grid_search = GridSearchCV(
    RandomForestRegressor(random_state=42),
    param_grid,
    cv=5,
    scoring='neg_mean_squared_error',
    return_train_score=True
)
Bayesian Optimization (optional):
Utilizes scikit-optimize's BayesSearchCV
Defines a continuous search space for parameters
Performs efficient optimization with fewer iterations than grid search
Model Deployment
Model Serialization
The final model is serialized using Python's pickle library:

python
with open("model.bin", 'wb') as f_out:
    pickle.dump(final_model, f_out)
Deployment Pipeline
The system includes a comprehensive deployment pipeline with:

Version control for models
Data validation before deployment
Model evaluation on test data
Metadata tracking for deployments
Performance logging
python
class ModelDeploymentPipeline:
    def __init__(self):
        self.version = self._get_next_version()
        
    def run_pipeline(self, data, test_data):
        """Run the full pipeline"""
        print(f"Starting deployment pipeline for model v{self.version}")
        self.validate_data(data)
        model = self.train_model(data)
        rmse = self.evaluate_model(model, test_data)
        self.save_model(model, rmse)
        print(f"✓ Deployment pipeline completed for model v{self.version}")
        return model
Web Application
Streamlit Interface
The system provides a web-based user interface using Streamlit:

Input controls for all vehicle specifications
Real-time prediction when input changes
Visual display of prediction results
Information about model performance
Prediction Function
The core prediction functionality is exposed through a clean API:

python
def predict_mpg(config, model, pipeline=None):
    if type(config) == dict:
        df = pd.DataFrame(config)
    else:
        df = config
    
    preproc_df = preprocess_origin_cols(df)
    
    # Apply transformation using saved pipeline
    if pipeline:
        prepared_df = pipeline.transform(preproc_df)
    else:
        # For backward compatibility
        prepared_df = pipeline_transformer(preproc_df)
        
    y_pred = model.predict(prepared_df)
    return y_pred
Performance Monitoring
Drift Detection
The system includes a basic drift detection mechanism:

python
def monitor_model_performance(model, test_data, test_labels, threshold=0.05):
    """
    Monitor model drift by comparing current performance against baseline
    """
    # Get current performance
    predictions = model.predict(test_data)
    current_rmse = np.sqrt(mean_squared_error(test_labels, predictions))
    
    # Compare with stored baseline
    try:
        with open('model_baseline.json', 'r') as f:
            baseline = json.load(f)
            baseline_rmse = baseline['rmse']
            
        # Check if performance degraded beyond threshold
        if (current_rmse - baseline_rmse) / baseline_rmse > threshold:
            print(f"Model performance degraded: {baseline_rmse:.2f} -> {current_rmse:.2f}")
            return True
    except FileNotFoundError:
        # Create baseline if it doesn't exist
        with open('model_baseline.json', 'w') as f:
            json.dump({'rmse': current_rmse, 'timestamp': datetime.now().isoformat()}, f)
    
    return False
Model Explainability
The system includes capabilities for model explainability using SHAP values:

Feature importance visualization
Impact of features on specific predictions
Force plots for detailed prediction explanation
Technical Specifications
Performance Metrics
Training Time: Varies by model (Linear Regression < 1s, Random Forest ~5s)
Prediction Time: < 50ms per prediction
Model Size: ~100KB (serialized Random Forest model)
Final Model Performance: RMSE of ~3.0 MPG on test data
Resource Requirements
Memory: 250MB+ recommended
Disk: 10MB for code and models
CPU: Single core sufficient for prediction, multi-core beneficial for training
Dependencies
The system requires the following Python packages:

numpy: Numerical computing
pandas: Data manipulation
matplotlib/seaborn: Data visualization
scikit-learn: Core machine learning functionality
pickle: Model serialization
streamlit: Web application framework
Optional dependencies:

featuretools: Automated feature engineering
scikit-optimize: Bayesian optimization
shap: Model explainability
Version Compatibility
The system has been tested with:

Python 3.7+
scikit-learn 1.0+
pandas 1.3+
streamlit 1.10+
Integration Points
The system provides the following integration capabilities:

Model as Service: Through the prediction function
Web UI: Via Streamlit
Batch Processing: Through direct script execution
Monitoring API: For tracking model performance

