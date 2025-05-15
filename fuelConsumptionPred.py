import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split, cross_val_score, GridSearchCV
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
import pickle
import warnings
warnings.filterwarnings('ignore')

# reading the .data file using pandas
cols = ['MPG','Cylinders','Displacement','Horsepower','Weight','Acceleration', 'Model Year', 'Origin']

df = pd.read_csv('./auto-mpg.data', names=cols, na_values = "?", comment = '\t', sep= " ", skipinitialspace=True)
data = df.copy()

data.sample(20)

# The data contains MPG variable which is continuous data and tells us about the efficiency of fuel consumption of a vehicle in 70s and 80s.
# Our aim here is to predict the MPG value for a vehicle given we have other attributes of that vehicle.

# EDA
##checking the data info
data.info()

##checking for all the null values
data.isnull().sum()

##summary statistics of quantitative variables
data.describe()

sns.boxplot(x=data['Horsepower'])
plt.show()    

##imputing the values with median
median = data['Horsepower'].median()
data['Horsepower'] = data['Horsepower'].fillna(median)
print(data.info())

##category distribution
print(data["Cylinders"].value_counts() / len(data))
print(data['Origin'].value_counts())

#pairplots to get an intuition of potential correlations
sns.pairplot(data[["MPG", "Cylinders", "Displacement", "Weight", "Horsepower"]], diag_kind="kde")
plt.show()

# set aside the test data
train_set, test_set = train_test_split(data, test_size=0.2, random_state=42)

print(test_set.shape)

train_set['Cylinders'].value_counts() / len(train_set)
test_set["Cylinders"].value_counts() / len(test_set)

# Stratified Sampling
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_index, test_index in split.split(data, data["Cylinders"]):
    strat_train_set = data.loc[train_index]
    strat_test_set = data.loc[test_index]
print(strat_test_set.shape)

#segregate the feature and target variable
data = strat_train_set.drop("MPG", axis=1)
data_labels = strat_train_set["MPG"].copy()
data

#preprocess the Origin column in data
def preprocess_origin_cols(df):
    df = df.copy()
    df["Origin"] = df["Origin"].map({1: "India", 2: "USA", 3: "Germany"})
    return df

#checking for cylinder category distribution in training set
strat_train_set['Cylinders'].value_counts() / len(strat_train_set)

#converting integer classes to countries in Origin column
train_set['Origin'] = train_set['Origin'].map({1: 'India', 2: 'USA', 3 : 'Germany'})
train_set.sample(10)

#one hot encoding
train_set = pd.get_dummies(train_set, prefix='', prefix_sep='')
train_set.head()

data = strat_train_set.copy()

#Checking correlation matrix w.r.t. MPG
corr_matrix = data.corr()
corr_matrix['MPG'].sort_values(ascending=False)

# testing new variables by checking their correlation w.r.t. MPG
data['displacement_on_power'] = data['Displacement'] / data['Horsepower']
data['weight_on_cylinder'] = data['Weight'] / data['Cylinders']
data['acceleration_on_power'] = data['Acceleration'] / data['Horsepower']
data['acceleration_on_cyl'] = data['Acceleration'] / data['Cylinders']

corr_matrix = data.corr()
corr_matrix['MPG'].sort_values(ascending=False)

# === Automated Feature Engineering === 
def setup_automated_feature_engineering():
    try:
        import featuretools as ft
        
        def automate_feature_engineering(data):
            """
            Use featuretools to automatically generate features
            """
            # Create entity set
            es = ft.EntitySet(id="vehicles")
            
            # Add entity
            es.add_dataframe(
                dataframe_name="data",
                dataframe=data.reset_index(),
                index="index",
                make_index=True
            )
            
            # Run deep feature synthesis
            print("Generating automated features...")
            feature_matrix, feature_defs = ft.dfs(
                entityset=es,
                target_dataframe_name="data",
                max_depth=2,
                features_only=False,
                verbose=1
            )
            
            # Show top new features by correlation with MPG
            feature_matrix['MPG'] = data_labels.values
            corr_with_mpg = feature_matrix.corr()['MPG'].sort_values(ascending=False)
            print("\nTop 10 auto-generated features by correlation with MPG:")
            print(corr_with_mpg.head(10))
            
            return feature_matrix, feature_defs
        
        # Uncomment to run automated feature engineering
        # auto_features, feature_defs = automate_feature_engineering(data)
        
        return automate_feature_engineering
        
    except ImportError:
        print("Featuretools not found. Install with: pip install featuretools")
        return None

auto_feature_engineering = setup_automated_feature_engineering()

#handling missing values
imputer = SimpleImputer(strategy="median")
imputer.fit(data)

print(imputer.statistics_)
print(data.median().values)

X = imputer.transform(data)

data_tr = pd.DataFrame(X, columns=data.columns, index=data.index)

#Segregating Target and Feature variables
data = strat_train_set.drop("MPG", axis=1)
data_labels = strat_train_set["MPG"].copy()
print(data)

#Preprocessing the Origin Column
data_tr = preprocess_origin_cols(data)
data_tr.head()

data_tr.info()

#isolating the origin column
data_cat = data_tr[["Origin"]]
data_cat.head()

#onehotencoding the categorical values
cat_encoder = OneHotEncoder()
data_cat_1hot = cat_encoder.fit_transform(data_cat)
data_cat_1hot   # returns a sparse matrix

data_cat_1hot.toarray()[:5]

cat_encoder.categories_

#segregating the numerical columns
num_data = data.iloc[:, :-1]
num_data.info()

#handling missing values
imputer = SimpleImputer(strategy="median")
imputer.fit(num_data)
print(imputer.statistics_)
print(data.median().values)

#imputing the missing values by transforming the dataframe
X = imputer.transform(num_data)
X

#converting the 2D array back into a dataframe
data_tr = pd.DataFrame(X, columns=num_data.columns,
                          index=num_data.index)
print(data_tr.info())
num_data.head()

#creating custom attribute adder class
##creating custom transformer to add new attributes to the numerical data
acc_ix, hpower_ix, cyl_ix = 4, 2, 0

class CustomAttrAdder(BaseEstimator, TransformerMixin):
    def __init__(self, acc_on_power=True): # no *args or **kargs
        self.acc_on_power = acc_on_power
    def fit(self, X, y=None):
        return self  # nothing else to do
    def transform(self, X):
        acc_on_cyl = X[:, acc_ix] / X[:, cyl_ix]
        if self.acc_on_power:
            acc_on_power = X[:, acc_ix] / X[:, hpower_ix]
            return np.c_[X, acc_on_power, acc_on_cyl]
        
        return np.c_[X, acc_on_cyl]
    
##Using Pipeline class
##Using StandardScaler to scale all the numerical attributes
def num_pipeline_transformer(data):
    '''
    Function to process numerical transformations
    Argument:
        data: original dataframe 
    Returns:
        num_attrs: numerical dataframe
        num_pipeline: numerical pipeline object
        
    '''
    numerics = ['float64', 'int64']

    num_attrs = data.select_dtypes(include=numerics)

    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('attrs_adder', CustomAttrAdder()),
        ('std_scaler', StandardScaler()),
        ])
    return num_attrs, num_pipeline

def pipeline_transformer(data):
    '''
    Complete transformation pipeline for both
    nuerical and categorical data.
    
    Argument:
        data: original dataframe 
    Returns:
        prepared_data: transformed data, ready to use
    '''
    cat_attrs = ["Origin"]

    # Get numerical attributes (all columns except Origin)
    numerics = ['float64', 'int64']
    num_attrs = list(data.select_dtypes(include=numerics).columns)
    num_attrs = [col for col in num_attrs if col != "Origin"]  

    # Create preprocessing pipelines
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('attrs_adder', CustomAttrAdder()),
        ('std_scaler', StandardScaler()),
    ])
    full_pipeline = ColumnTransformer([
        ("num", num_pipeline, list(num_attrs)),
        ("cat", OneHotEncoder(sparse_output = False), cat_attrs),
        ])
    prepared_data = full_pipeline.fit_transform(data)
    return prepared_data

#from raw data to processed data in 2 steps
preprocessed_df = preprocess_origin_cols(data)
prepared_data = pipeline_transformer(preprocessed_df)
print(prepared_data)
print(prepared_data[0])

lin_reg = LinearRegression()
lin_reg.fit(prepared_data, data_labels)
     
##testing the predictions with the 
sample_data = data.iloc[:5]
sample_labels = data_labels.iloc[:5]

sample_data_prepared = pipeline_transformer(sample_data)

print("Prediction of samples: ", lin_reg.predict(sample_data_prepared))     
print("Actual Labels of samples: ", list(sample_labels))

#Mean Squared Error
mpg_predictions = lin_reg.predict(prepared_data)
lin_mse = mean_squared_error(data_labels, mpg_predictions)
lin_rmse = np.sqrt(lin_mse)
lin_rmse

# Decision Tree
tree_reg = DecisionTreeRegressor()
tree_reg.fit(prepared_data, data_labels)

mpg_predictions = tree_reg.predict(prepared_data)
tree_mse = mean_squared_error(data_labels, mpg_predictions)
tree_rmse = np.sqrt(tree_mse)
tree_rmse

#Model Evaluation using Cross Validation
scores = cross_val_score(tree_reg, 
                         prepared_data, 
                         data_labels, 
                         scoring="neg_mean_squared_error", 
                         cv = 10)
tree_reg_rmse_scores = np.sqrt(-scores)

print(tree_reg_rmse_scores)
print(tree_reg_rmse_scores.mean())

scores = cross_val_score(lin_reg, prepared_data, data_labels, scoring="neg_mean_squared_error", cv = 10)
lin_reg_rmse_scores = np.sqrt(-scores)
print(lin_reg_rmse_scores)
print(lin_reg_rmse_scores.mean())
print(lin_reg_rmse_scores.std())

# Random Forest model
forest_reg = RandomForestRegressor()
forest_reg.fit(prepared_data, data_labels)
forest_reg_cv_scores = cross_val_score(forest_reg,
                                         prepared_data,
                                         data_labels,
                                         scoring='neg_mean_squared_error',
                                         cv = 10)

forest_reg_rmse_scores = np.sqrt(-forest_reg_cv_scores)
print(forest_reg_rmse_scores.mean())

#Support Vector Machine Regressor
svm_reg = SVR(kernel='linear')
svm_reg.fit(prepared_data, data_labels)
svm_cv_scores = cross_val_score(svm_reg, prepared_data, data_labels,
                                scoring='neg_mean_squared_error',
                                cv = 10)
svm_rmse_scores = np.sqrt(-svm_cv_scores)
print(svm_rmse_scores.mean())

# === Advanced Hyperparameter Optimization ===
def optimize_with_bayesian():
    try:
        from skopt import BayesSearchCV
        from skopt.space import Real, Integer, Categorical
        
        def optimize_model_bayesian():
            """
            Optimize model using Bayesian optimization
            """
            # Define search space
            search_space = {
                'n_estimators': Integer(10,  100),
                'max_depth': Integer(10, 20),
                'min_samples_split': Integer(2,  10),
                'min_samples_leaf': Integer(1,  4),
                'max_features': Real(0.1, 0.9)
            }
            
            # Create model
            forest_reg = RandomForestRegressor(random_state=42)
            
            # Create BayesSearchCV
            bayes_search = BayesSearchCV(
                forest_reg,
                search_space,
                n_iter=50,
                cv=5,
                scoring='neg_mean_squared_error',
                verbose=1,
                n_jobs=-1
            )
            
            print("Running Bayesian optimization. This may take some time...")
            # Fit model
            bayes_search.fit(prepared_data, data_labels)
            
            print(f"Best parameters: {bayes_search.best_params_}")
            print(f"Best score: {np.sqrt(-bayes_search.best_score_)}")
            
            return bayes_search.best_estimator_
        
        # Uncomment to run Bayesian optimization
        better_model = optimize_model_bayesian()
        return better_model
        
        return optimize_model_bayesian
    
    except ImportError:
        print("Scikit-optimize not found. Install with: pip install scikit-optimize")
        return None

bayesian_optimizer = optimize_with_bayesian()

# Define parameter grid for RandomForest optimization
param_grid = [
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
]

# Create the GridSearchCV object
grid_search = GridSearchCV(
    RandomForestRegressor(random_state=42),
    param_grid,
    cv=5,
    scoring='neg_mean_squared_error',
    return_train_score=True
)

# Fit the grid search (assuming prepared_data and data_labels are already defined)
grid_search.fit(prepared_data, data_labels)

##printing all the parameters along with their scores
for mean_score, params in zip(grid_search.cv_results_['mean_test_score'], grid_search.cv_results_["params"]):
    print(np.sqrt(-mean_score), params)

# feature importances 
feature_importances = grid_search.best_estimator_.feature_importances_
print(feature_importances)

extra_attrs = ["acc_on_power", "acc_on_cyl"]
numerics = ['float64', 'int64']
num_attrs = list(data.select_dtypes(include=numerics))

attrs = num_attrs + extra_attrs
sorted(zip(attrs, feature_importances), reverse=True)

# === Model Explainability ===
def add_explainability_features():
    try:
        import shap
        
        def explain_prediction(model, input_data):
            """
            Generate SHAP values to explain model predictions
            """
            # Create explainer
            explainer = shap.TreeExplainer(model)
            
            # Calculate SHAP values
            shap_values = explainer.shap_values(input_data)
            
            # Plot feature importance
            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_values, input_data, plot_type="bar")
            
            # Plot impact on prediction
            plt.figure(figsize=(10, 6))
            shap.force_plot(explainer.expected_value, shap_values[0,:], input_data.iloc[0,:])
            
            return shap_values
        
        # Example explanation for a sample
        sample_to_explain = X_test_prepared[:5]
        sample_explanation = explain_prediction(final_model, pd.DataFrame(sample_to_explain))
        
        return explain_prediction
    
    except ImportError:
        print("SHAP library not found. Install with: pip install shap")
        return None



#Evaluating the entire system on Test Data
final_model = grid_search.best_estimator_

X_test = strat_test_set.drop("MPG", axis=1)
y_test = strat_test_set["MPG"].copy()

X_test_preprocessed = preprocess_origin_cols(X_test)
X_test_prepared = pipeline_transformer(X_test_preprocessed)

explain_func = add_explainability_features()

final_predictions = final_model.predict(X_test_prepared)
final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)
print(final_rmse)
print(final_predictions[:5])

# === Model Monitoring and Updating ===
import json
from datetime import datetime

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

# Optionally run monitoring on the test data
needs_retraining = monitor_model_performance(final_model, X_test_prepared, y_test)
if needs_retraining:
    print("Model retraining recommended")

#Creating a function to cover this entire flow
def predict_mpg(config, model, pipeline = None):
    
    if type(config) == dict:
        df = pd.DataFrame(config)
    else:
        df = config
    
    preproc_df = preprocess_origin_cols(df)
    prepared_df = pipeline_transformer(preproc_df)
    # Apply transformation using saved pipeline
    if pipeline:
        prepared_df = pipeline.transform(preproc_df)
    else:
        # For backward compatibility
        prepared_df = pipeline_transformer(preproc_df)
        
    y_pred = model.predict(prepared_df)
    return y_pred

# Checking it on a random sample
vehicle_config = {
    'Cylinders': [4, 6, 8],
    'Displacement': [155.0, 160.0, 165.5],
    'Horsepower': [93.0, 130.0, 98.0],
    'Weight': [2500.0, 3150.0, 2600.0],
    'Acceleration': [15.0, 14.0, 16.0],
    'Model Year': [81, 80, 78],
    'Origin': [3, 2, 1]
}

predict_mpg(vehicle_config, final_model)

# === Web Application Interface ===
if __name__ == "__main__":
    import streamlit as st
    
    def create_web_app():
        st.title('Vehicle MPG Predictor')
    
        # Load the model at the beginning of the function
        try:
            with open('model.bin', 'rb') as f_in:
                model = pickle.load(f_in)
        except FileNotFoundError:
            st.error("Model file not found. Please make sure 'model.bin' exists in the current directory.")
            
            # Input widgets for vehicle attributes
            cylinders = st.selectbox('Cylinders', options=[3, 4, 5, 6, 8])
            displacement = st.slider('Displacement (cu. inches)', 50.0, 500.0, 150.0)
            horsepower = st.slider('Horsepower', 40.0, 250.0, 100.0)
            weight = st.slider('Weight (lbs)', 1500.0, 5000.0, 3000.0)
            acceleration = st.slider('Acceleration (sec to 60mph)', 8.0, 25.0, 15.0)
            model_year = st.selectbox('Model Year', options=list(range(70, 85)))
            origin = st.selectbox('Origin', options=[1, 2, 3], 
                                format_func=lambda x: {1: "India", 2: "USA", 3: "Germany"}[x])
            
            if st.button('Predict MPG'):
                # Create config from inputs
                config = {
                    'Cylinders': [cylinders],
                    'Displacement': [displacement],
                    'Horsepower': [horsepower],
                    'Weight': [weight],
                    'Acceleration': [acceleration],
                    'Model Year': [model_year],
                    'Origin': [origin]
                }
                
                # Load model and predict
                prediction = predict_mpg(config, model)
                st.success(f'The estimated MPG is: {prediction[0]:.2f}')
        
        # Comment out this line to disable web app when running as module
        create_web_app()

# === Deployment Pipeline ===
def create_deployment_pipeline():
    """
    Create a simple deployment pipeline
    """
    import json
    from datetime import datetime
    
    class ModelDeploymentPipeline:
        def __init__(self):
            self.version = self._get_next_version()
            
        def _get_next_version(self):
            """Get next model version"""
            try:
                with open('model_metadata.json', 'r') as f:
                    metadata = json.load(f)
                    return metadata.get('latest_version', 0) + 1
            except FileNotFoundError:
                return 1
                
        def validate_data(self, data):
            """Validate input data"""
            assert 'MPG' in data.columns, "MPG column missing"
            assert len(data) > 100, "Insufficient data"
            assert not data['Horsepower'].isnull().any(), "Missing Horsepower values"
            print("✓ Data validation passed")
            return True
        
        def train_model(self, data):
            """Train the model"""
            # Split data
            X = data.drop('MPG', axis=1)
            y = data['MPG']
            
            # Create pipeline
            preprocessed_X = preprocess_origin_cols(X)
            prepared_X = pipeline_transformer(preprocessed_X)
            
            # Get best parameters from previous runs or use default
            try:
                with open('model_metadata.json', 'r') as f:
                    metadata = json.load(f)
                    best_params = metadata.get('best_params', {})
            except FileNotFoundError:
                best_params = {'n_estimators': 30, 'max_features': 6}
            
            # Train model
            print(f"Training model with parameters: {best_params}")
            model = RandomForestRegressor(**best_params)
            model.fit(prepared_X, y)
            print("✓ Model training completed")
            return model
        
        def evaluate_model(self, model, test_data):
            """Evaluate model performance"""
            X_test = test_data.drop('MPG', axis=1)
            y_test = test_data['MPG']
            
            X_test_preprocessed = preprocess_origin_cols(X_test)
            X_test_prepared = pipeline_transformer(X_test_preprocessed)
            
            predictions = model.predict(X_test_prepared)
            rmse = np.sqrt(mean_squared_error(y_test, predictions))
            print(f"✓ Model evaluation - RMSE: {rmse:.4f}")
            return rmse
        
        def save_model(self, model, rmse):
            """Save model and metadata"""
            # Save model
            model_filename = f"model_v{self.version}.bin"
            with open(model_filename, 'wb') as f_out:
                pickle.dump(model, f_out)
            
            # Save metadata
            try:
                with open('model_metadata.json', 'r') as f:
                    metadata = json.load(f)
            except FileNotFoundError:
                metadata = {'versions': []}
            
            # Update metadata
            metadata['latest_version'] = self.version
            metadata['latest_model'] = model_filename
            metadata['best_params'] = model.get_params()
            
            # Add version info
            version_info = {
                'version': self.version,
                'filename': model_filename,
                'timestamp': datetime.now().isoformat(),
                'rmse': rmse,
                'parameters': model.get_params()
            }
            
            metadata['versions'].append(version_info)
            
            with open('model_metadata.json', 'w') as f_out:
                json.dump(metadata, f_out, indent=2)
                
            print(f"✓ Model v{self.version} saved successfully")
            
        def run_pipeline(self, data, test_data):
            """Run the full pipeline"""
            print(f"Starting deployment pipeline for model v{self.version}")
            self.validate_data(data)
            model = self.train_model(data)
            rmse = self.evaluate_model(model, test_data)
            self.save_model(model, rmse)
            print(f"✓ Deployment pipeline completed for model v{self.version}")
            return model
    
    # Create pipeline instance
    pipeline = ModelDeploymentPipeline()
    
    # Uncomment to run the pipeline
    # deployed_model = pipeline.run_pipeline(strat_train_set, strat_test_set)
    
    return pipeline

deployment_pipeline = create_deployment_pipeline()

#Save the Model
##saving the model
with open("model.bin", 'wb') as f_out:
    pickle.dump(final_model, f_out)
    f_out.close() 

##loading the model from the saved file
with open('model.bin', 'rb') as f_in:
    model = pickle.load(f_in)

predict_mpg(vehicle_config, model)