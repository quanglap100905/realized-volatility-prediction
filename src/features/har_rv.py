import numpy as np
import pandas as pd


def realized_volatility(series):
    """Compute realized volatility from log returns"""
    return np.sqrt(np.sum(series ** 2))


def compute_har_rv_features(df):
    """Compute multi-scale realized volatility and volatility trend features"""

    # Realized volatility at different time scales
    df_300 = df[df["seconds_in_bucket"] >= 300]
    df_450 = df[df["seconds_in_bucket"] >= 450]

    rv_600 = (
        df.groupby("time_id")["log_return1"]
        .apply(realized_volatility)
        .rename("har_rv_600")
    )

    rv_300 = (
        df_300.groupby("time_id")["log_return1"]
        .apply(realized_volatility)
        .rename("har_rv_300")
    )

    rv_150 = (
        df_450.groupby("time_id")["log_return1"]
        .apply(realized_volatility)
        .rename("har_rv_150")
    )

    har_features = pd.concat([rv_600, rv_300, rv_150], axis=1).fillna(0)

    # Relative volatility across time scales
    har_features["har_momentum_150_600"] = (
        har_features["har_rv_150"] / (har_features["har_rv_600"] + 1e-8)
    )

    har_features["har_momentum_300_600"] = (
        har_features["har_rv_300"] / (har_features["har_rv_600"] + 1e-8)
    )

    # Volatility across four 150-second intervals
    df["quarter"] = pd.cut(
        df["seconds_in_bucket"],
        bins=[-1, 150, 300, 450, 600],
        labels=[1, 2, 3, 4]
    )

    quarterly_rv = (
        df.groupby(["time_id", "quarter"], observed=False)["log_return1"]
        .apply(realized_volatility)
        .unstack()
        .fillna(0)
    )

    # OLS slope of quarterly volatility
    x = np.array([1, 2, 3, 4])
    x_mean = x.mean()
    denominator = np.sum((x - x_mean) ** 2)

    def calculate_slope(y):
        if np.all(y == 0):
            return 0

        return np.sum((x - x_mean) * (y - np.mean(y))) / denominator

    har_features["volatility_slope"] = quarterly_rv.apply(
        calculate_slope, axis=1
    )

    return har_features.reset_index()