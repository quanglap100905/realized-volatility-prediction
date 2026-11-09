import os

# Paths
DATA_DIR = '../data'
TRAIN_CSV = os.path.join(DATA_DIR, 'train.csv')
TEST_CSV = os.path.join(DATA_DIR, 'test.csv')
BOOK_TRAIN_DIR = os.path.join(DATA_DIR, 'book_train.parquet')
TRADE_TRAIN_DIR = os.path.join(DATA_DIR, 'trade_train.parquet')
BOOK_TEST_DIR = os.path.join(DATA_DIR, 'book_test.parquet')
TRADE_TEST_DIR = os.path.join(DATA_DIR, 'trade_test.parquet')

# LightGBM Hyperparameters
LGB_PARAMS = {
    'objective': 'rmse',
    'learning_rate': 0.05,
    'max_depth': 6,
    'num_leaves': 64,
    'feature_fraction': 0.8,
    'verbosity': -1,
    'random_state': 42
}

# CatBoost Hyperparameters
CB_PARAMS = {
    'loss_function': 'RMSE',
    'eval_metric': 'RMSE',
    'learning_rate': 0.03,
    'depth': 6,
    'l2_leaf_reg': 3,
    'random_seed': 42,
    'od_type': 'Iter',
    'od_wait': 200,
    'verbose': 200
}