import os
import glob
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
import catboost as cb
from sklearn.model_selection import GroupKFold

warnings.simplefilter("ignore", FutureWarning)

from config import *
from metrics import rmspe, feval_rmspe
from data_loader import process_all_files
from features.microstructure import process_book_data, process_trade_data
from features.macro_tsne import apply_tsne_regime


def main():
    # Load data
    print("Loading data...")
    train = pd.read_csv(TRAIN_CSV)
    test = pd.read_csv(TEST_CSV)

    if "target" not in test.columns:
        test["target"] = 0.0

    train["row_id"] = train["stock_id"].astype(str) + "-" + train["time_id"].astype(str)
    test["row_id"] = test["stock_id"].astype(str) + "-" + test["time_id"].astype(str)

    # Get file paths
    book_train_paths = glob.glob(f"{BOOK_TRAIN_DIR}/*/*")
    trade_train_paths = glob.glob(f"{TRADE_TRAIN_DIR}/*/*")
    book_test_paths = glob.glob(f"{BOOK_TEST_DIR}/*/*")
    trade_test_paths = glob.glob(f"{TRADE_TEST_DIR}/*/*")

    # Extract order book features
    print("Processing order book data...")
    df_book_train = process_all_files(book_train_paths, process_book_data)
    df_book_test = process_all_files(book_test_paths, process_book_data)

    # Extract trade features
    print("Processing trade data...")
    df_trade_train = process_all_files(trade_train_paths, process_trade_data)
    df_trade_test = process_all_files(trade_test_paths, process_trade_data)

    # Merge features
    print("Merging features...")
    train_df = train.merge(df_book_train, on=["stock_id", "time_id"], how="left")
    train_df = train_df.merge(df_trade_train, on=["stock_id", "time_id"], how="left")

    test_df = test.merge(df_book_test, on=["stock_id", "time_id"], how="left")
    test_df = test_df.merge(df_trade_test, on=["stock_id", "time_id"], how="left")

    # Add market-level features
    train_df, test_df = apply_tsne_regime(train_df, test_df)

    # Model features
    print("\nStarting cross-validation...")
    features = [
        c for c in train_df.columns
        if c not in ["time_id", "target", "stock_id", "row_id"]
    ]

    kf = GroupKFold(n_splits=5)

    oof_lgb = np.zeros(len(train_df))
    oof_cb = np.zeros(len(train_df))
    test_pred_lgb = np.zeros(len(test_df))
    test_pred_cb = np.zeros(len(test_df))

    train_df["stock_id"] = train_df["stock_id"].astype(int)
    test_df["stock_id"] = test_df["stock_id"].astype(int)

    cb_features = features + ["stock_id"]

    for fold, (trn_ind, val_ind) in enumerate(
        kf.split(train_df, groups=train_df["time_id"])
    ):
        print(f"\nFold {fold + 1}/{kf.n_splits}")

        x_train = train_df.iloc[trn_ind]
        x_val = train_df.iloc[val_ind]
        y_train = train_df["target"].iloc[trn_ind]
        y_val = train_df["target"].iloc[val_ind]

        # Target encoding
        stock_target_mean = x_train.groupby("stock_id")["target"].mean()

        x_train_lgb = x_train.copy()
        x_val_lgb = x_val.copy()
        x_test_lgb = test_df.copy()

        x_train_lgb["stock_id_target_enc"] = x_train["stock_id"].map(stock_target_mean)
        x_val_lgb["stock_id_target_enc"] = x_val["stock_id"].map(stock_target_mean)
        x_test_lgb["stock_id_target_enc"] = (
            test_df["stock_id"].map(stock_target_mean).fillna(y_train.mean())
        )

        lgb_features = features + ["stock_id_target_enc"]

        # Sample weights for RMSPE
        train_weights = 1 / np.square(y_train)
        val_weights = 1 / np.square(y_val)

        # LightGBM
        print("Training LightGBM...")
        train_dataset = lgb.Dataset(
            x_train_lgb[lgb_features],
            label=y_train,
            weight=train_weights
        )
        valid_dataset = lgb.Dataset(
            x_val_lgb[lgb_features],
            label=y_val,
            weight=val_weights
        )

        model_lgb = lgb.train(
            LGB_PARAMS,
            train_dataset,
            valid_sets=[train_dataset, valid_dataset],
            num_boost_round=3000,
            callbacks=[
                lgb.early_stopping(200),
                lgb.log_evaluation(200)
            ],
            feval=feval_rmspe
        )

        oof_lgb[val_ind] = model_lgb.predict(x_val_lgb[lgb_features])
        test_pred_lgb += model_lgb.predict(x_test_lgb[lgb_features]) / kf.n_splits

        # CatBoost
        print("Training CatBoost...")
        train_pool = cb.Pool(
            x_train[cb_features],
            label=y_train,
            weight=train_weights,
            cat_features=["stock_id"]
        )
        valid_pool = cb.Pool(
            x_val[cb_features],
            label=y_val,
            weight=val_weights,
            cat_features=["stock_id"]
        )

        model_cb = cb.CatBoostRegressor(**CB_PARAMS)
        model_cb.fit(
            train_pool,
            eval_set=valid_pool,
            use_best_model=True
        )

        oof_cb[val_ind] = model_cb.predict(x_val[cb_features])
        test_pred_cb += model_cb.predict(test_df[cb_features]) / kf.n_splits

    # Blend predictions
    oof_blend = (oof_lgb + oof_cb) / 2
    test_blend = (test_pred_lgb + test_pred_cb) / 2

    print("\nOOF Results")
    print(f"LightGBM: {rmspe(train_df['target'], oof_lgb):.5f}")
    print(f"CatBoost: {rmspe(train_df['target'], oof_cb):.5f}")
    print(f"Ensemble: {rmspe(train_df['target'], oof_blend):.5f}")

    # Save submission
    test_df["target"] = test_blend
    test_df[["row_id", "target"]].to_csv("submission.csv", index=False)

    print("Saved submission.csv")


if __name__ == "__main__":
    main()