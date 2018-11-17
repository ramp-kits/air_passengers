import pandas as pd
import os


def _encode(X_df):
    X_encoded = X_df
    path = os.path.dirname(__file__)
    data_weather = pd.read_csv(os.path.join(path, 'external_data.csv'))
    X_weather = data_weather[['Date', 'AirPort', 'Events']]
    X_weather = X_weather.rename(
        columns={'Date': 'DateOfDeparture', 'AirPort': 'Arrival'})
    X_encoded = pd.merge(
        X_encoded, X_weather, how='left',
        left_on=['DateOfDeparture', 'Arrival'],
        right_on=['DateOfDeparture', 'Arrival'],
        sort=False)

    X_encoded = X_encoded.join(pd.get_dummies(
        X_encoded['Departure'], prefix='d'))
    X_encoded = X_encoded.join(
        pd.get_dummies(X_encoded['Arrival'], prefix='a'))
    X_encoded = X_encoded.join(
        pd.get_dummies(X_encoded['Events']))
    X_encoded = X_encoded.drop('Departure', axis=1)
    X_encoded = X_encoded.drop('Arrival', axis=1)
    X_encoded = X_encoded.drop('Events', axis=1)
    X_encoded = X_encoded.drop('DateOfDeparture', axis=1)
    return X_encoded


class FeatureExtractor(object):
    def __init__(self):
        pass

    def fit(self, X_df, y_array):
        X_encoded = _encode(X_df)
        self.columns = X_encoded.columns
        return self

    def transform(self, X_df):
        X_encoded = _encode(X_df)
        X_empty = pd.DataFrame(columns=self.columns)
        X_encoded = pd.concat([X_empty, X_encoded], axis=0, sort=False)
        X_encoded = X_encoded.fillna(0)

        # Reorder/Pick columns from train
        X_encoded = X_encoded[list(self.columns)]
        # Check that columns of test set are the same than train set
        assert list(X_encoded.columns) == list(self.columns)
        X_array = X_encoded.values
        return X_array
