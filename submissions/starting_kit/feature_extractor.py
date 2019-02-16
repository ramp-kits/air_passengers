import os
import warnings

import numpy as np
import pandas as pd

from sklearn.base import TransformerMixin
from sklearn.exceptions import DataConversionWarning
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder


class FeatureExtractor(TransformerMixin):
    def __init__(self):
        numerical_columns = ['WeeksToDeparture', 'std_wtd', 'Max TemperatureC']
        categorical_columns = ['Departure', 'Arrival']
        self.preprocessor = make_column_transformer(
            (StandardScaler(), numerical_columns),
            (OneHotEncoder(), categorical_columns)
        )

    def _merge_external_data(self, X_df):
        """Merge the orignal data with the external data."""
        submission_path = os.path.dirname(__file__)
        df_weather = pd.read_csv(
            os.path.join(submission_path, 'external_data.csv'),
            usecols=['Date', 'AirPort', 'Max TemperatureC'],
        )
        df_weather = df_weather.rename(
            columns={'Date': 'DateOfDeparture', 'AirPort': 'Arrival'}
        )
        df_weather['DateOfDeparture'] = pd.to_datetime(
            df_weather['DateOfDeparture']
        )
        return X_df.merge(
            df_weather, how='left',
            left_on=['DateOfDeparture', 'Arrival'],
            right_on=['DateOfDeparture', 'Arrival'],
            sort=False
        )

    def fit(self, X_df, y_array):
        X_df = self._merge_external_data(X_df)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DataConversionWarning)
            self.preprocessor.fit(X_df)
        return self

    def transform(self, X_df):
        X_df = self._merge_external_data(X_df)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DataConversionWarning)
            return self.preprocessor.transform(X_df)
