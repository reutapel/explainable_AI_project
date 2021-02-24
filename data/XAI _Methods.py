import xgboost
'''https://github.com/slundberg/shap'''
import shap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

'''https://iancovert.com/blog/understanding-shap-sage/'''
'''https://github.com/iancovert/sage'''
'''pip install sage-importance'''
import sage


'''https://github.com/slundberg/shap/issues/632'''
def get_feature_mean_values(X,shap_values):
    vals = np.abs(shap_values).mean(0)
    feature_importance = pd.DataFrame(list(zip(X.columns, vals)), columns=['col_name', 'feature_importance_vals'])
    feature_importance.sort_values(by=['feature_importance_vals'], ascending=False, inplace=True)
    return feature_importance

