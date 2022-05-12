# Packages
import os
import pickle
import shutil

import mlflow
import pandas as pd
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from mlflow.pyfunc import PythonModel


# Model Class
class WrapperLRModel(PythonModel):
    
    def __init__(self, sklearn_model_features, cat_features, model_artifact_name):
        """
        cat_features: Mapping from categorical feature names to all
        possible values
        """
        self.feature_names = sklearn_model_features
        self.cat_features = cat_features
        self.model_artifact_name = model_artifact_name
    
    def load_context(self, context):
        with open(context.artifacts[self.model_artifact_name], 'rb') as m:
            self.lr_model = pickle.load(m)
            
    def _encode(self, row, colname):
        value = row[colname]
        row[value] = 1
        return row
        
    def predict(self, context, model_input):
        model_features = model_input
        for col, unique_values in self.cat_features.items():
            for uv in unique_values:
                model_features[uv] = 0
            model_features = model_features.apply(lambda row: self._encode(row, col), axis=1)
        model_features = model_features.loc[:, self.feature_names]
        return self.lr_model.predict(model_features.to_numpy())

# Prepare the data
def prepare_data(dataframe):
    df = dataframe
    cat_features_values = {}
    # Encode all the categorical features
    for col in list(dataframe.columns):
        if col in cat_features:
            cat_features_values[col] = list(dataframe[col].unique())
            # One-hot encoding
            dummies = pd.get_dummies(df[col])
            # Drop the original column
            df = pd.concat([df.drop([col], axis=1), dummies], axis=1)

    # Fill missing values with 0
    df = df.fillna(0)

    return df, cat_features_values

# Train and Evaluate
def train_and_evaluate(dataframe, cat_features_values):
    features = dataframe.drop(["SalePrice"], axis=1)
    target = dataframe.loc[:, "SalePrice"]
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features.to_numpy(), target.to_numpy(), test_size=0.2, random_state=12)

    # Plot
    plot = dataframe.plot.scatter(x=0, y='SalePrice')
    fig = plot.get_figure();
    fig.savefig('tmp/plot.png');

    # Save the dataset
    dataframe.to_csv('tmp/dataset.csv', index=False)

    # Train the model
    model = linear_model.LinearRegression()
    model.fit(X_train, y_train)

    #####
    # Save the model into mlflow
    serialized_model = pickle.dumps(model)
    with open('tmp/model.pkl', 'wb') as f:
        f.write(serialized_model)
        
    # Model Artifacts
    model_artifact_name = 'original_sklearn_model'
    model_artifacts = {
        model_artifact_name: 'tmp/model.pkl'
    }
    
    # Log Model
    mlflow.pyfunc.log_model(
        'custom_model',
        python_model=
        WrapperLRModel(
            sklearn_model_features=list(features.columns),
            cat_features=cat_features_values,
            model_artifact_name = model_artifact_name,
                                   ),
        artifacts=model_artifacts,
        registered_model_name="House prices"
    )

    ## Add the log artifacts to store it in the ui
    # mlflow.log_artifacts('tmp/model.pkl', artifact_path='models')
    mlflow.log_artifacts('tmp') # It will pass all the artifacts on that directory
    #####

    # Evaluate the model
    y_pred = model.predict(X_test)
    err = mean_squared_error(y_test, y_pred)
    mlflow.log_metric("MSE", err)
    
# Main Logic

# Define localhost
mlflow.set_tracking_uri("http://localhost:8000/")
mlflow.set_experiment("Models exercise")

# Create a Temporary directory to store the models
# It will be cleaned after the cycle runs
if not os.path.exists('tmp'):
    os.makedirs('tmp')
    
# Load the complete dataset, select features + target variable
data = pd.read_csv("ames_housing.csv")
feature_columns = ["Lot Area", "Gr Liv Area", "Garage Area", "Bldg Type"]
selected = data.loc[:, feature_columns + ["SalePrice"]]

# Features that need encoding (categorical ones)
cat_features = ["Bldg Type"]

# Add None so that we get one training with all the columns
columns_to_drop = feature_columns + [None]

for to_drop in columns_to_drop:
    if to_drop:
        dropped = selected.drop([to_drop], axis=1)
    else:
        dropped = selected

    with mlflow.start_run():
        mlflow.log_param("dropped_column", to_drop)
        prepared, cat_features_values = prepare_data(dropped)
        train_and_evaluate(prepared, cat_features_values)


# Clean directory
shutil.rmtree('tmp')







