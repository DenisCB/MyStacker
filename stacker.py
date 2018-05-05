import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.base import BaseEstimator

from helpers import xgb_predictor, lgb_predictor, preprocess_input


class Stacker:

    def __init__(
        self, train, ytrain,
        metric,
        test=None, ytest=None,
        features=[], features_to_encode=[],
        split_by=None, stratify_folds=False, n_splits=5, split_seed=0
    ):
        """
        metric - function, that accepts (true_values, predicted_values)
            and returns float.
        train, ytrain - train data in pandas format
        test - data to predict. If not given - you still
            can get OOF prediction for train data.
        ytest - if given, stacker will be evaluated on test data.
        features - list of pandas column names to train on.
        features_to_encode (not implemented) - features for target encoding.
            Encoding uses target from train folds only.
        split_by - column name, unique values from which should be found
            in a single fold. Is used to avoid overfitting or leakage.
            For example you may want to put all events corresponding
            to the same `user_id` to a single fold. If None - ignored.
        stratify_folds (bool) - used if split_by is None.
        n_splits - number of splits for train data. In order to get
            one OOF prediciton, model must be fitted n_splits times.
        split_seed - seed for folds

        """
        train, ytrain, features, split_by, test, ytest = preprocess_input(
            train, ytrain, features, split_by, test, ytest)

        self._train = train
        self.ytrain = ytrain
        self.metric = metric
        self._test = test
        self.ytest = ytest

        self.features = features
        self.features_to_encode = features_to_encode

        self.split_by = split_by
        self.stratify_folds = stratify_folds
        self.n_splits = n_splits
        self.split_seed = split_seed
        # Current level of fitting. 1 means fitting on src features.
        # 2 means fitting on meta features.
        self.level = 1

        # Meta dataframes - for storing level 1 predictions.
        self.train_meta = pd.DataFrame()
        self.test_meta = pd.DataFrame()
        # Result dataframes - for storing level 2 predictions.
        self.train_result = pd.DataFrame()
        self.test_result = pd.DataFrame()
        self.folds = self.get_folds()

    @property
    def train(self):
        if self.level == 1:
            return self._train
        elif self.level == 2:
            return self.train_meta
        else:
            raise NotImplementedError('Only levels 1 and 2 are implemented')

    @property
    def test(self):
        if self.level == 1:
            return self._test
        elif self.level == 2:
            return self.test_meta
        else:
            raise NotImplementedError('Only levels 1 and 2 are implemented')

    def get_folds(self, **kwargs):

        train = kwargs.get('train', self.train)
        ytrain = kwargs.get('ytrain', self.ytrain)
        split_by = kwargs.get('split_by', self.split_by)
        n_splits = kwargs.get('n_splits', self.n_splits)

        # Check whether split_by is defined.
        # If yes, create folds for unique values in split_by.
        if isinstance(split_by, pd.Series):
            folds = []
            split_by_unique = split_by.unique()
            kf = KFold(
                n_splits=n_splits,
                shuffle=True,
                random_state=self.split_seed)

            # For each unique split_by value in each fold get indexes of
            # elements in main dataframe with corresponding split_by
            # values and add them into single fold.
            for train_vals, test_vals in kf.split(split_by_unique):
                train_fold_ind = (
                    split_by
                    [split_by.isin(split_by_unique[train_vals])]
                    .index.values
                    )
                test_fold_ind = (
                    split_by
                    [split_by.isin(split_by_unique[test_vals])]
                    .index.values
                    )
                folds.append([train_fold_ind, test_fold_ind])

        # If split_by is not defined - use stratified or common KFold.
        elif self.stratify_folds:
            kf = StratifiedKFold(
                n_splits=n_splits,
                shuffle=True,
                random_state=self.split_seed)
            folds = list(kf.split(train, ytrain))
        else:
            kf = KFold(
                n_splits=n_splits,
                shuffle=True,
                random_state=self.split_seed)
            folds = list(kf.split(train))

        return folds

    def get_valid_fold(
        self, train_fold, ytrain_fold,
        train_ind, test_ind,
        model, valid_size
    ):
        if model in ('xgb', 'lgb', 'nnet') and valid_size > 0:
            if self.split_by is None:
                split_by_fold = None
            else:
                split_by_fold = self.split_by.iloc[train_ind].reset_index(drop=True)
            train_ind, valid_ind = self.get_folds(
                train=train_fold,
                ytrain=ytrain_fold,
                split_by=split_by_fold,
                n_splits=int(1/valid_size)
                )[0]
            valid_fold = train_fold.iloc[valid_ind]
            train_fold = train_fold.iloc[train_ind]
            yvalid_fold = ytrain_fold[valid_ind]
            ytrain_fold = ytrain_fold[train_ind]

        # If no valid_size is given - give to it the whole test fold.
        # Minor overfitting is possible.
        elif model in ('xgb', 'lgb', 'nnet'):
            valid_fold = self.train.iloc[test_ind]
            yvalid_fold = self.ytrain[test_ind]

            train_fold = train_fold.reset_index(drop=True)
            valid_fold = valid_fold.reset_index(drop=True)
        else:
            valid_fold, yvalid_fold = None, None

        return train_fold, ytrain_fold, valid_fold, yvalid_fold

    def fit_sklearn_model(self, train_fold, ytrain_fold, test_ind, model):
            model.fit(train_fold, ytrain_fold)
            # Binary classification.
            try:
                fold_pred = model.predict_proba(
                    self.train.iloc[test_ind])[:, 1]
                test_pred = model.predict_proba(self.test)[:, 1]
            # Regression.
            except:
                fold_pred = model.predict(self.train.iloc[test_ind])
                test_pred = model.predict(self.test)
            return fold_pred, test_pred

    def fit_custom_model(
        self,
        train_fold, ytrain_fold,
        valid_fold, yvalid_fold,
        test_ind, model, model_params,
    ):
        if model == 'lgb':
            fold_pred, test_pred = lgb_predictor(
                model_params,
                train_fold, ytrain_fold,
                valid_fold, yvalid_fold,
                self.train.iloc[test_ind], self.test)
        elif model == 'xgb':
            fold_pred, test_pred = xgb_predictor(
                model_params,
                train_fold, ytrain_fold,
                valid_fold, yvalid_fold,
                self.train.iloc[test_ind], self.test)
        elif model == 'nnet':
            raise NotImplementedError('No neural net for you.')
        return fold_pred, test_pred

    def fit(self, model, colname, model_params=None, valid_size=0, level=1):

        # Temporary arrays to store single model predicitons.
        train_meta = np.zeros(self.train.shape[0])
        test_meta = np.zeros(self.test.shape[0])
        self.level = level

        for train_ind, test_ind in self.folds:

            train_fold = self.train.iloc[train_ind]
            ytrain_fold = self.ytrain[train_ind]

            # Adding validation sets (from train set) for models that require
            # it. Valid set is splitted from part of the train_fold.
            train_fold, ytrain_fold, valid_fold, yvalid_fold = self.get_valid_fold(
                train_fold, ytrain_fold, train_ind, test_ind, model, valid_size)

            # Block for feature target encoding.
            if len(self.features_to_encode) > 0:
                raise NotImplementedError('No feature encoding yet.')

            # Block for working with sklearn models.
            if isinstance(model, BaseEstimator):
                fold_pred, test_pred = self.fit_sklearn_model(
                    train_fold, ytrain_fold, test_ind, model)
            else:
                fold_pred, test_pred = self.fit_custom_model(
                    train_fold, ytrain_fold,
                    valid_fold, yvalid_fold,
                    test_ind, model, model_params
                    )

            train_meta[test_ind] += fold_pred
            test_meta += test_pred / self.n_splits

            # if scale_logloss:
            #     data_oofs[test_ind] += mod_stat(fold_pred, np.mean(fold_pred))
            #     kagg_oofs += mod_stat(kagg_pred, np.mean(kagg_pred))
            # else:
            #     data_oofs[test_ind] += fold_pred
            #     kagg_oofs += kagg_pred

            if self.ytest is None:
                print('Metric on test fold: ',
                      round(self.metric(self.ytrain[test_ind], fold_pred), 4))
            else:
                print('Metric on test: ',
                      round(self.metric(self.ytest, test_pred), 4))

        if self.ytest is None:
            print('Iteration OOF score:',
                  round(self.metric(self.ytrain, train_meta), 4))
        else:
            print('Iteration test score:',
                  round(self.metric(self.ytest, test_meta), 4))
        print()

        if self.level == 1:
            self.train_meta[colname] = train_meta
            self.test_meta[colname] = test_meta
        else:
            self.train_result[colname] = train_meta
            self.test_result[colname] = test_meta

    def get_metrics(self, ytest=None):
        if ytest is None:
            ytest = self.ytest

        if ytest is None:
            for c in self.train_meta.columns:
                metric = self.metric(self.ytrain, self.train_meta[c])
                print('{} - {} on train'.format(
                        round(metric, 4), c))
        else:
            for c in self.train_meta.columns:
                metric = self.metric(ytest, self.test_meta[c])
                print('{} - {} on test'.format(
                    round(metric, 4), c))

    def get_metrics_final(self, ytest=None):
        if ytest is None:
            ytest = self.ytest

        if ytest is None:
            for c in self.train_result.columns:
                metric = self.metric(self.ytrain, self.train_result[c])
                print('{} - {} on train'.format(
                        round(metric, 4), c))
        else:
            for c in self.train_result.columns:
                metric = self.metric(ytest, self.test_result[c])
                print('{} - {} on test'.format(
                    round(metric, 4), c))
