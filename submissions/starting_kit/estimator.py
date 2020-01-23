import os
import pandas as pd
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor


def _merge_external_data(X):
    filepath = os.path.join(
        os.path.dirname(__file__), 'external_data.csv'
    )
    data_weather = pd.read_csv(filepath)
    X_weather = data_weather[['Date', 'AirPort', 'Max TemperatureC']]
    X_weather = X_weather.rename(
        columns={'Date': 'DateOfDeparture', 'AirPort': 'Arrival'}
    )
    X_merged = pd.merge(
        X, X_weather, how='left', on=['DateOfDeparture', 'Arrival'], sort=False
    )
    return X_merged


def _encode_dates(X):
    # to avoid SettingwithCopyWarning
    X_encoded = X.copy()
    X_encoded['DateOfDeparture'] = pd.to_datetime(X['DateOfDeparture'])
    X_encoded['year'] = X_encoded['DateOfDeparture'].dt.year
    X_encoded['month'] = X_encoded['DateOfDeparture'].dt.month
    X_encoded['day'] = X_encoded['DateOfDeparture'].dt.day
    X_encoded['weekday'] = X_encoded['DateOfDeparture'].dt.weekday
    X_encoded['week'] = X_encoded['DateOfDeparture'].dt.week
    X_encoded['n_days'] = X_encoded['DateOfDeparture'].apply(
        lambda date: (date - pd.to_datetime("1970-01-01")).days
    )
    return X_encoded


def get_estimator():
    merge_transformer = FunctionTransformer(_merge_external_data)
    date_transformer = FunctionTransformer(_encode_dates)

    categorical_cols = ['Arrival', 'Departure']
    drop_col = ['DateOfDeparture']
    preprocessor = make_column_transformer(
        (OrdinalEncoder(), categorical_cols),
        ('drop', drop_col),
        remainder='passthrough'
    )
    pipeline = make_pipeline(
        merge_transformer, date_transformer, preprocessor,
        RandomForestRegressor(n_estimators=10, max_depth=10, max_features=10)
    )
    return pipeline
