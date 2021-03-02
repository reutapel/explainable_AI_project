'''https://iancovert.com/blog/understanding-shap-sage/'''
'''https://github.com/iancovert/sage'''
'''pip install sage-importance'''
import sage
'''https://github.com/slundberg/shap'''
import shap
import xgboost
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import logging
import utils

class XAIMethods:
    def __init__(self, model, X, y, method_name, visualize: bool):
        self.model = model
        self.X = X
        self.y = y
        self.method_name = method_name

        #todo: for now support tree models only, should add if else for different
        if 'shap' in str.lower(method_name):
            # explain the model's predictions using SHAP
            # (same syntax works for LightGBM, CatBoost, scikit-learn and spark models)
            self.explainer = shap.TreeExplainer(model)
            self.shap_values = self.explainer.shap_values(X)

        #todo: this library needs to be debugged - does not work as expected
        elif 'sage' in str.lower(method_name):
            print()
        else:
            logging.error('XAI method name not included. this class supports: SHAP and SAGE')
            print('XAI method name not included. this class supports: SHAP and SAGE')
            raise Exception('XAI method name not included. this class supports: SHAP and SAGE')



    '''https://github.com/slundberg/shap/issues/632'''
    def get_shap_feature_mean_values(self):
        vals = np.abs(self.shap_values).mean(0)
        feature_importance = pd.DataFrame(list(zip(self.X.columns, vals)), columns=['col_name', 'feature_importance_vals'])
        feature_importance.sort_values(by=['feature_importance_vals'], ascending=False, inplace=True)
        return feature_importance

    def visualize_shap(self):
        # visualize the first prediction's explanation (use matplotlib=True to avoid Javascript)
        shap.force_plot(self.explainer.expected_value, self.shap_values[0, :], X.iloc[0, :])
        shap.summary_plot(self.shap_values, self.X)
        shap.summary_plot(self.shap_values, self.X, plot_type="bar")

