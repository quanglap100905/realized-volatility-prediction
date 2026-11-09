import numpy as np
import pandas as pd
from .har_rv import compute_har_rv_features, realized_volatility


def aggregate_features(df, agg_dict):
    """Aggregate features over the full window and recent time windows"""
    features = []

    for sec in [600, 300, 150]:
        data = df if sec == 600 else df[df["seconds_in_bucket"] >= 600 - sec]

        agg = data.groupby("time_id").agg(agg_dict)
        agg.columns = ["_".join(col).strip("_") for col in agg.columns]
        
        if sec != 600:
            agg = agg.add_suffix(f"_{sec}")

        features.append(agg)

    result = features[0]

    for sec, agg in zip([300, 150], features[1:]):
        result = result.join(agg, how="left")

    return result.reset_index()


def process_book_data(file_path):
    """Feature Engineering for Limit Order Book data including OFI, HAR-RV, Volatility slope """
    df = pd.read_parquet(file_path)

    # WAP and returns
    df["wap1"] = (
        df["bid_price1"] * df["ask_size1"] +
        df["ask_price1"] * df["bid_size1"]
    ) / (df["bid_size1"] + df["ask_size1"])

    df["wap2"] = (
        df["bid_price2"] * df["ask_size2"] +
        df["ask_price2"] * df["bid_size2"]
    ) / (df["bid_size2"] + df["ask_size2"])

    for i in [1, 2]:
        df[f"log_return{i}"] = df.groupby("time_id")[f"wap{i}"].transform(
            lambda x: np.log(x).diff()
        )

    # Spreads and volume
    df["wap_balance"] = abs(df["wap1"] - df["wap2"])
    df["price_spread"] = (
        (df["ask_price1"] - df["bid_price1"]) /
        ((df["ask_price1"] + df["bid_price1"]) / 2)
    )
    df["total_volume"] = df[
        ["ask_size1", "ask_size2", "bid_size1", "bid_size2"]
    ].sum(axis=1)

    df["volume_imbalance"] = abs(
        df["ask_size1"] + df["ask_size2"] -
        df["bid_size1"] - df["bid_size2"]
    )

    # OFI
    bid_price = df.groupby("time_id")["bid_price1"].shift(1)
    bid_size = df.groupby("time_id")["bid_size1"].shift(1)
    ask_price = df.groupby("time_id")["ask_price1"].shift(1)
    ask_size = df.groupby("time_id")["ask_size1"].shift(1)

    # OFI = bid pressure - ask pressure
    df["ofi"] = (
        np.where(df["bid_price1"] > bid_price, df["bid_size1"],
        np.where(df["bid_price1"] == bid_price, df["bid_size1"] - bid_size, -bid_size))
        -
        np.where(df["ask_price1"] < ask_price, df["ask_size1"],
        np.where(df["ask_price1"] == ask_price, df["ask_size1"] - ask_size, -ask_size))
    )

    # HAR-RV features
    har_features = compute_har_rv_features(df)

    agg_dict = {
        "wap1": ["mean"],
        "wap2": ["mean"],
        "log_return1": [realized_volatility],
        "log_return2": [realized_volatility],
        "wap_balance": ["mean"],
        "price_spread": ["mean"],
        "total_volume": ["sum"],
        "volume_imbalance": ["mean"],
        "ofi": ["var", "sum", realized_volatility]
    }

    features = aggregate_features(df, agg_dict)
    features = features.merge(har_features, on="time_id", how="left")
    features["stock_id"] = int(file_path.split("=")[1])

    return features


def process_trade_data(file_path):
    """Feature Engineering for Trading data"""
    df = pd.read_parquet(file_path)

    df["log_return"] = df.groupby("time_id")["price"].transform(
        lambda x: np.log(x).diff()
    )

    agg_dict = {
        "log_return": [realized_volatility],
        "size": ["sum"],
        "order_count": ["sum"]
    }

    features = aggregate_features(df, agg_dict)
    features["stock_id"] = int(file_path.split("=")[1])

    return features