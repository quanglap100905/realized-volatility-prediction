import pandas as pd
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


def apply_tsne_regime(train_df, test_df):
    """Add t-SNE-based time and market volatility features"""

    print("Applying t-SNE time reconstruction...")

    # Combine train and test data
    combined_df = pd.concat([
        train_df[["time_id", "stock_id", "log_return1_realized_volatility"]],
        test_df[["time_id", "stock_id", "log_return1_realized_volatility"]]
    ], ignore_index=True)

    # Create a time_id x stock_id volatility matrix
    vol_matrix = combined_df.pivot(
        index="time_id",
        columns="stock_id",
        values="log_return1_realized_volatility"
    )
    vol_matrix = vol_matrix.fillna(vol_matrix.mean())

    # Scale features before t-SNE
    vol_matrix_scaled = StandardScaler().fit_transform(vol_matrix)

    tsne = TSNE(
        n_components=1,
        perplexity=400,
        random_state=42,
        n_iter=2000,
        init="pca",
        learning_rate="auto"
    )

    time_coords = tsne.fit_transform(vol_matrix_scaled).flatten()

    tsne_df = pd.DataFrame({
        "time_id": vol_matrix.index,
        "tsne_time_coord": time_coords
    })

    # Estimate market volatility along the reconstructed timeline
    tsne_df = tsne_df.sort_values("tsne_time_coord").reset_index(drop=True)

    market_vol = combined_df.groupby("time_id")[
        "log_return1_realized_volatility"
    ].mean()

    tsne_df["market_vol"] = tsne_df["time_id"].map(market_vol)
    tsne_df["reconstructed_rolling_market_vol"] = (
        tsne_df["market_vol"]
        .rolling(window=5, center=True, min_periods=1)
        .mean()
    )

    # Add reconstructed time features to train and test
    features = tsne_df[[
        "time_id",
        "tsne_time_coord",
        "reconstructed_rolling_market_vol"
    ]]

    train_df = train_df.merge(features, on="time_id", how="left")
    test_df = test_df.merge(features, on="time_id", how="left")

    return train_df, test_df