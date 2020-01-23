import os
import pandas as pd
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor


def _merge_external_data(X):
    X.loc[:, 'DateOfDeparture'] = pd.to_datetime(X.loc[:, 'DateOfDeparture'])
    filepath = os.path.join(
        os.path.dirname(__file__), 'external_data.csv'
    )
    data_weather = pd.read_csv(filepath, parse_dates=["Date"])
    X_weather = data_weather[['Date', 'AirPort', 'Events']]
    X_weather = X_weather.rename(
        columns={'Date': 'DateOfDeparture', 'AirPort': 'Arrival'}
    )
    X_merged = pd.merge(
        X, X_weather, how='left', on=['DateOfDeparture', 'Arrival'], sort=False
    )
    return X_merged


def _encode_dates(X):
    X.loc[:, 'year'] = X['DateOfDeparture'].dt.year
    X.loc[:, 'month'] = X['DateOfDeparture'].dt.month
    X.loc[:, 'day'] = X['DateOfDeparture'].dt.day
    X.loc[:, 'weekday'] = X['DateOfDeparture'].dt.weekday
    X.loc[:, 'week'] = X['DateOfDeparture'].dt.week
    X.loc[:, 'n_days'] = X['DateOfDeparture'].apply(
        lambda date: (date - pd.to_datetime("1970-01-01")).days
    )
    return X


def get_estimator():
    merge_transformer = FunctionTransformer(_merge_external_data)
    date_transformer = FunctionTransformer(_encode_dates)

    categorical_cols = ['Arrival', 'Departure', 'Events']
    drop_col = ['DateOfDeparture']
    preprocessor = make_column_transformer(
        (make_pipeline(
            SimpleImputer(strategy="constant", fill_value="missing"),
            OrdinalEncoder()
        ), categorical_cols),
        ('drop', drop_col),
        remainder='passthrough'
    )
    pipeline = make_pipeline(
        merge_transformer, date_transformer, preprocessor,
        RandomForestRegressor(n_estimators=10, max_depth=10, max_features=10)
    )
    return pipeline
