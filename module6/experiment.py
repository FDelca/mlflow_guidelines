import os
import pickle
import shutil

import mlflow
import pandas as pd
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("Dropping columns")

os.makedirs("tmp")

# Load the complete dataset, select features + target variable
data = pd.read_csv("../ames_housing.csv")
feature_columns = ["Lot Area", "Gr Liv Area", "Garage Area", "Bldg Type"]
selected = data.loc[:, feature_columns + ["SalePrice"]]

# Features that need encoding (categorical ones)
cat_features = ["Bldg Type"]


def prepare_data(dataframe):
    df = dataframe
    # Encode all the categorical features
    for col in list(dataframe.columns):
        if col in cat_features:
            # One-hot encoding
            dummies = pd.get_dummies(df[col])
            # Drop the original column
            df = pd.concat([df.drop([col], axis=1), dummies], axis=1)

    # Fill missing values with 0
    df = df.fillna(0)

    return df


def train_and_evaluate(dataframe):
    # Separate features from the target variable and convert to NumPy
    features = dataframe.drop(["SalePrice"], axis=1).to_numpy()
    target = dataframe.loc[:, "SalePrice"].to_numpy()
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=12)

    # Plot
    plot = dataframe.plot.scatter(x=0, y="SalePrice")
    fig = plot.get_figure()
    fig.savefig("tmp/plot.png")

    # Save the dataset
    dataframe.to_csv("tmp/dataset.csv", index=False)

    # Train the model
    model = linear_model.LinearRegression()
    model.fit(X_train, y_train)

    # Save the model
    serialized_model = pickle.dumps(model)
    with open("tmp/model.pkl", "wb") as f:
        f.write(serialized_model)

    # Evaluate the model
    y_pred = model.predict(X_test)
    err = mean_squared_error(y_test, y_pred)
    mlflow.log_metric("MSE", err)

    # Log the artifacts
    mlflow.log_artifacts("tmp")


# Add None so that we get one training with all the columns
columns_to_drop = feature_columns + [None]

for to_drop in columns_to_drop:

    if to_drop:
        dropped = selected.drop([to_drop], axis=1)
    else:
        dropped = selected

    with mlflow.start_run():
        mlflow.log_param("dropped_column", to_drop)
        prepared = prepare_data(dropped)
        train_and_evaluate(prepared)


shutil.rmtree("tmp")
