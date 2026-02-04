"""
Diagnoser module for analyzing time series data characteristics.

This module provides tools for diagnosing time series data, including seasonality
measurement using STL decomposition.
"""
import numpy as np
from statsmodels.tsa.seasonal import STL
from nazuna.data_manager import TimeSeriesDataManager


class Diagnoser:
    """
    Diagnose time series data characteristics such as seasonality.

    This class analyzes time series data from a TimeSeriesDataManager and computes
    various statistical properties including seasonality strength based on STL
    decomposition.
    """

    def __init__(self, dm, data_range):
        """
        Args:
            dm: TimeSeriesDataManager instance containing the time series data.
            data_range: Data range for diagnostics as (start, end) ratios.
        """
        self.dm = dm
        self.df = dm.extract_data(data_range)
        self.df = self.df.drop(columns=['timestamp', 'timestep'])

    def measure_seasonality(self, period: int | None = None) -> dict:
        """
        Measure seasonality strength for each channel using STL decomposition.

        The seasonality strength is computed as:
            max(0, 1 - var(R) / var(X - T))
        where R is the residual and T is the trend component from STL decomposition.

        Args:
            period: Seasonal period for STL decomposition. If None, defaults to 7.

        Returns:
            Dictionary containing:
                - 'seasonality_per_channel': list of seasonality strength for each channel
                - 'seasonality_mean': mean seasonality strength across all channels
                - 'period': the period used for decomposition
        """
        if period is None:
            period = 7

        cols_org = {}
        seasonality_per_channel = {}

        for i_col, col in enumerate(self.df.columns):
            series = self.df[col].values

            if len(series) < 2 * period:
                seasonality_per_channel[col] = float('nan')
                continue

            stl = STL(series, period=period, robust=True)
            result = stl.fit()

            residual = result.resid
            trend = result.trend
            detrended = series - trend

            var_residual = np.var(residual)
            var_detrended = np.var(detrended)

            if var_detrended == 0:
                seasonality_strength = 0.0
            else:
                seasonality_strength = max(0.0, 1.0 - var_residual / var_detrended)

            cols_org[col] = self.dm.cols_org[i_col]
            seasonality_per_channel[col] = float(seasonality_strength)

        valid_values = [v for v in seasonality_per_channel.values() if not np.isnan(v)]
        seasonality_mean = float(np.mean(valid_values)) if valid_values else np.nan

        return {
            'cols_org': cols_org,
            'seasonality_per_channel': seasonality_per_channel,
            'seasonality_mean': seasonality_mean,
            'period': period,
        }

    def sample(self, n_channels: int = 4, n_steps: int = 96) -> dict:
        """
        Extract a small sample of data for visualization.

        Args:
            n_channels: Maximum number of channels to extract.
            n_steps: Number of time steps to extract from the beginning.

        Returns:
            Dictionary containing:
                - 'values': numpy array of shape (n_steps, n_channels)
                - 'columns': list of column names
                - 'timestamps': list of timestamp strings
        """
        n_channels = min(n_channels, len(self.df.columns))
        n_steps = min(n_steps, len(self.df))
        columns = list(self.df.columns[:n_channels])
        timestamps = list(self.dm.df['timestamp'].iloc[:n_steps].astype(str))
        values = self.df.iloc[:n_steps, :n_channels].values
        return {
            'values': values,
            'columns': columns,
            'timestamps': timestamps,
        }

    def run(self, period: int | None = None) -> tuple[dict, dict]:
        """
        Run all diagnostics and return results with sample data.

        Args:
            period: Seasonal period for STL decomposition. If None, defaults to 7.

        Returns:
            Tuple of (result dict, sample data dict).
        """
        result = {}
        result['seasonality'] = self.measure_seasonality(period=period)
        data = self.sample()
        return result, data
