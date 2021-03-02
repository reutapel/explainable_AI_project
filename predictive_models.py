from xgboost.sklearn import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
# from lightgbm.sklearn import LGBMClassifier
from catboost import CatBoostClassifier
import pandas as pd
import logging
import utils
import os


class PredictiveModel:
    def __init__(self, features, model_name, hyper_parameters: dict, model_num: int, fold: int, fold_dir: str,
                 excel_models_results: pd.ExcelWriter, trained_model=None):
        self.features = features
        self.model_name = model_name
        self.model_num = model_num
        self.fold = fold
        self.fold_dir = fold_dir
        self.model_table_writer = pd.ExcelWriter(
            os.path.join(excel_models_results, f'Results_fold_{fold}_model_{model_num}.xlsx'), engine='xlsxwriter')
        self.trained_model = trained_model
        if self.trained_model is not None:
            self.model = trained_model

        else:
            if 'randomforest' in str.lower(model_name):
                self.model = RandomForestClassifier(n_estimators=hyper_parameters['n_estimators'],
                                                    max_depth=hyper_parameters['max_depth'],
                                                    min_samples_split=hyper_parameters['min_samples_split'],
                                                    min_samples_leaf=hyper_parameters['min_samples_leaf'])
            elif 'xgboost' in str.lower(model_name):
                self.model = XGBClassifier(learning_rate=hyper_parameters['learning_rate'],
                                           n_estimators=hyper_parameters['n_estimators'],
                                           max_depth=hyper_parameters['max_depth'],
                                           min_child_weight=hyper_parameters['min_child_weight'],
                                           gamma=hyper_parameters['gamma'],
                                           subsample=hyper_parameters['subsample'])
            # elif 'lightgbm' in str.lower(model_name):
            #     self.model = LGBMClassifier(num_leaves=hyper_parameters['num_leaves'],
            #                                 max_depth=hyper_parameters['num_leaves'],
            #                                 learning_rate=hyper_parameters['learning_rate'],
            #                                 n_estimators=hyper_parameters['n_estimators'],
            #                                 subsample_for_bin=hyper_parameters['subsample_for_bin'], objective='binary',
            #                                 min_child_samples=hyper_parameters['min_child_samples'],
            #                                 reg_alpha=hyper_parameters['reg_alpha'],
            #                                 reg_lambda=hyper_parameters['reg_lambda'])
            elif 'catboost' in str.lower(model_name):
                self.model = CatBoostClassifier(
                    iterations=hyper_parameters['iterations'],
                    depth=hyper_parameters['depth'],
                    learning_rate=hyper_parameters['learning_rate'],
                    l2_leaf_reg=hyper_parameters['l2_leaf_reg'],
                    bootstrap_type='Bayesian',  # Poisson (supported for GPU only);Bayesian;Bernoulli;No
                    bagging_temperature=1,  # for Bayesian bootstrap_type; 1=exp;0=1
                    leaf_estimation_method='Newton',  # Gradient;Newton
                    leaf_estimation_iterations=2,
                    boosting_type='Ordered')  # Ordered-small data sets; Plain
            else:
                logging.error('Model name not in: CatBoost, lightgbm, XGBoost', 'RandomForest')
                print('Model name not in: CatBoost, lightgbm, XGBoost', 'RandomForest')
                raise Exception('Model name not in: CatBoost, lightgbm, XGBoost', 'RandomForest')

    def fit(self, train_x: pd.DataFrame, train_y: pd.Series):
        train_x = train_x[self.features]
        train_x.columns = [' '.join(col).strip() for col in train_x.columns.values]
        self.model = self.model.fit(train_x, train_y.values.ravel())

    def predict(self, validation_x: pd.DataFrame, validation_y: pd.Series):
        validation_x = validation_x[self.features]
        validation_x.columns = [' '.join(col).strip() for col in validation_x.columns.values]
        if 'xgboost' in str.lower(self.model_name):
            validation_x = validation_x[self.model._Booster.feature_names]
        predictions = self.model.predict(validation_x)
        validation_y.columns = ['labels']
        predictions = pd.Series(predictions, index=validation_y.index, name='predictions')
        predictions = pd.DataFrame(predictions).join(validation_y)
        utils.save_model_prediction(model_to_dave=self.model, model_name=self.model_name, data_to_save=predictions,
                                    fold_dir=self.fold_dir, fold=self.fold, model_num=self.model_num,
                                    table_writer=self.model_table_writer, save_model=True)

        return predictions
