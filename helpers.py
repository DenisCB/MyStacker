import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb


def xgb_predictor(
    params,
    train_fold, ytrain_fold,
    valid_fold, yvalid_fold,
    test_fold, test
):
    """
    input:
        params - dictionary of parameters to be passed to xgb.train
        train_fold, valid_fold, test_fold, test - numpy arrays or csr matrices.
            Model is trained on train data, best number of epochs is chosen
            by loss on valid. Test_fold and test - matrices, for which
            predictions are returned.
        ytrain_fold, yvalid_fold - 1-dim numpy arrays, labels for
            train_fold and valid_fold.
    output: two 1-dim numpy arrays with predictions for
        test_fold and test sets
    """
    train_fold = xgb.DMatrix(train_fold, ytrain_fold)
    valid_fold = xgb.DMatrix(valid_fold, yvalid_fold)
    watchlist = [(train_fold, 'train'), (valid_fold, 'eval')]

    booster = xgb.train(
        params, train_fold,
        num_boost_round=params.get('num_rounds', 999999),
        evals=watchlist,
        verbose_eval=False,
        # maximize=maximize_metric,
        early_stopping_rounds=50
        )

    # if we trained a linear model, then it has no ntree_limit parameter
    if params.get('booster', 'gbtree') == 'gbtree':
        fold_pred = booster.predict(
            xgb.DMatrix(test_fold), ntree_limit=booster.best_iteration)
        test_pred = booster.predict(
            xgb.DMatrix(test), ntree_limit=booster.best_iteration)
    else:
        fold_pred = booster.predict(xgb.DMatrix(test_fold))
        test_pred = booster.predict(xgb.DMatrix(test))

    return fold_pred, test_pred


def lgb_predictor(
    params,
    train_fold, ytrain_fold,
    valid_fold, yvalid_fold,
    test_fold, test
):
    """
    input:
        params - dictionary of parameters to be passed to xgb.train
        train_fold, valid_fold, test_fold, test - numpy arrays or csr matrices.
            Model is trained on train data, best number of epochs is chosen
            by loss on valid. Test_fold and test - matrices, for which
            predictions are returned.
        ytrain_fold, yvalid_fold - 1-dim numpy arrays, labels for
            train_fold and valid_fold.
    output: two 1-dim numpy arrays with predictions for
        test_fold and test sets
    """
    dtrain = lgb.Dataset(train_fold, ytrain_fold)
    dvalid = lgb.Dataset(valid_fold, yvalid_fold, reference=dtrain)

    gbm = lgb.train(
        params, dtrain,
        num_boost_round=params.get('num_rounds', 10000),
        valid_sets=[dtrain, dvalid],
        verbose_eval=False,
        early_stopping_rounds=50)

    fold_pred = gbm.predict(test_fold, num_iteration=gbm.best_iteration)
    test_pred = gbm.predict(test, num_iteration=gbm.best_iteration)

    return fold_pred, test_pred


def preprocess_input(train, ytrain, features, split_by, test, ytest):

    # If test data is not provided - create it as the first line of train
    # (just not to break anything).
    if test is None:
        test = train[:1].copy()

    # If no features passed - then use all columns from data.
    if isinstance(train, pd.DataFrame):
        if not features:
            features = train.columns
        train = train[features]
    if isinstance(test, pd.DataFrame):
        test = test[features]

    # Turn `split_by` column name into a pandas series.
    # We'll apply KFold to it later.
    if split_by in features:
        split_by = train[split_by].copy()
    elif isinstance(split_by, int):
        split_by = train[:, split_by].copy()

    # If target is in pandas format - turn it to numpy array
    if isinstance(ytrain, (pd.DataFrame, pd.Series)):
        ytrain = ytrain.values
    if isinstance(ytest, (pd.DataFrame, pd.Series)):
        ytest = ytest.values
    return train, ytrain, features, split_by, test, ytest
