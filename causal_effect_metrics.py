from typing import Union, Tuple
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
from os.path import dirname, abspath
import os
import utils


models_num_dict = {'LSTM_TR': {'BERT_O': 183, 'BERT_CF': 16},
                   'LSTM_CR': {'BERT_O': 63, 'BERT_CF': 5}
                   }

base_directory = os.path.abspath(os.curdir)
model_results_dir = os.path.join(base_directory, 'models_results_to_compare')


def get_ATE(original_clf_f_probs: np.ndarray, original_clf_cf_probs: np.ndarray, confidence_intervals: bool = False) -> Union[float, Tuple[float, Tuple[float, float]]]:
    """
        Calculates ATE, given class probabilities predicted by the original classifier for all factual and counterfactual test examples.

        Args:
            original_clf_f_probs: Class probabilities for all factual test examples,
                                  predicted by classifier utilizing original vector representations.
                                  Array shape (n_examples, n_classes)
            original_clf_cf_probs: Class probabilities for all counterfactual test examples,
                                   predicted by classifier utilizing original vector representations.
                                   Array shape (n_examples, n_classes)

        Returns:
            ATE score (float).
    """
    ITE_array = np.absolute(np.subtract(original_clf_f_probs, original_clf_cf_probs)).sum(axis=1)
    ATE = ITE_array.mean()
    if confidence_intervals:
        return ATE, get_confidence_intervals(ITE_array, ATE)
    else:
        return ATE


def get_TReATE(original_clf_probs: np.ndarray, treated_clf_probs: np.ndarray, confidence_intervals: bool = False) ->\
        Union[float, Tuple[float, Tuple[float, float]]]:
    """
        Calculates TReATE, given class probabilities predicted by classifiers
        utilizing original and treated vector representations for all test examples.

        Args:
            original_clf_probs: Class probabilities for all test examples,
                                predicted by classifier utilizing original vector representations.
                                Array shape (n_examples, n_classes)
            treated_clf_probs: Class probabilities for all test examples,
                               predicted by classifier utilizing treated vector representations.
                               Array shape (n_examples, n_classes)

        Returns:
            TReATE score (float).
    """
    TRITE_array = np.absolute(np.subtract(original_clf_probs, treated_clf_probs))
    TReATE = TRITE_array.mean()
    if confidence_intervals:
        return round(TReATE, 4), get_confidence_intervals(TRITE_array, TReATE)
    else:
        return round(TReATE, 4)


def get_CONEXP(original_clf_y_probs: np.ndarray, z_treatment_labels: np.ndarray, confidence_intervals: bool = False) -> Union[float, Tuple[float, Tuple[float, float]]]:
    """
        Calculates CONEXP, given y class probabilities predicted by original classifier and z_treatment labels for all test examples.

        Args:
            original_clf_y_probs: Class probabilities for all test examples,
                                  predicted by classifier utilizing original vector representations.
                                  Array shape (n_examples, n_classes)
            z_treatment_labels: Class probabilities for all test examples,
                               predicted by classifier utilizing treated vector representations.
                               Array shape (n_examples,)

        Returns:
            CONEXP score (float).
    """
    CONEXP_array = np.absolute(original_clf_y_probs[z_treatment_labels == 1, :].mean(axis=0) - original_clf_y_probs[z_treatment_labels == 0, :].mean(axis=0))
    CONEXP = CONEXP_array.sum()
    if confidence_intervals:
        return CONEXP, get_confidence_intervals(CONEXP_array, CONEXP)
    else:
        return CONEXP


def get_TPR_GAP(y_pred: np.ndarray, y_true: np.ndarray, z_treatment: np.ndarray) -> float:
    """
    Adapted from https://github.com/shauli-ravfogel/nullspace_projection/blob/master/notebooks/biasbios_bert.ipynb
    """
    scores = defaultdict(Counter)
    poms_count_total = defaultdict(Counter)

    for y_hat, y, g in zip(y_pred, y_true, z_treatment):

        if y == y_hat:
            scores[y][g] += 1

        poms_count_total[y][g] += 1

    tprs = defaultdict(dict)
    tprs_change = dict()

    for pom, scores_dict in scores.items():
        good_m, good_f = scores_dict[0], scores_dict[1]
        pom_total_f = poms_count_total[pom][1]
        pom_total_m = poms_count_total[pom][0]
        tpr_m = (good_m) / pom_total_m
        tpr_f = (good_f) / pom_total_f

        tprs[pom][0] = tpr_m
        tprs[pom][1] = tpr_f
        tprs_change[pom] = tpr_f - tpr_m
    return np.absolute(list(tprs_change.values())).sum()


def get_confidence_intervals(results_array: np.ndarray, final_result: float) -> Tuple[float, float]:
    results_interval = 1.96 * results_array.std() / np.sqrt(len(results_array))
    return final_result - results_interval, final_result + results_interval


def seq_models_results(model_type: str, bert_type: str, model_num: str):
    print(f'create seq results for all folds for model type {model_type} and bert {bert_type}')
    model_results = pd.DataFrame()
    for fold in range(6):
        fold_results_path = os.path.join(model_results_dir, f'{model_type}_{bert_type}',
                                         f'Results_fold_{fold}_model_{model_num}.xlsx')
        if os.path.exists(fold_results_path):
            fold_results = pd.read_excel(fold_results_path, header=1, sheet_name='Model results')
            fold_results = fold_results.loc[(fold_results.Raisha == 'All_raishas') &
                                            (fold_results.Round == 'All_rounds')]
            model_results = model_results.append(fold_results)

    if model_results.empty:
        print(f'No results for model type {model_type} and bert {bert_type}')
        return model_results
    model_results.columns = map(str.lower, model_results.columns)
    agg_columns = ['rmse', 'bin_4_bins_fbeta_score_macro']
    if model_type == 'LSTM_TR':
        model_results['per_round_fbeta_macro'] = model_results[['per_round_fbeta_score_dm chose hotel',
                                                                'per_round_fbeta_score_dm chose stay home']].mean(axis=1)
        agg_columns += ['per_round_accuracy', 'per_round_fbeta_macro']
    model_results = model_results.groupby(by=['model_num', 'model_type', 'model_name'])[agg_columns].mean().round(2)

    return model_results


def main():
    path = f'{dirname(abspath(__file__))}/CausaLM/Reviews_Features/datasets/causal_graph.csv'
    causal_graph_df = pd.read_csv(path)
    bert_models = causal_graph_df.col_name.unique()

    total_payoff_prediction_column = 'final_total_payoff_prediction'
    per_round_prediction_column = 'per_round_predictions'

    all_model_results = pd.DataFrame()

    TreATE_results = pd.DataFrame()
    for model_type in ['LSTM_TR', 'LSTM_CR']:
        print(f'create seq results for all folds')
        model_results = seq_models_results(model_type=model_type, bert_type='BERT_O',
                                           model_num=str(models_num_dict[model_type]["BERT_O"]))
        all_model_results = all_model_results.append(model_results)
        for feature_num, feature in enumerate(bert_models):
            model_results = seq_models_results(model_type=model_type, bert_type=feature,
                                               model_num=f'{models_num_dict[model_type]["BERT_CF"]}_{feature_num}')
            all_model_results = all_model_results.append(model_results)

        print(f'Calculate TreATE for models {model_type}')
        for fold in range(6):
            original_clf_total = pd.read_excel(
                os.path.join(model_results_dir, f'{model_type}_BERT_O',
                             f'Results_fold_{fold}_model_{models_num_dict[model_type]["BERT_O"]}.xlsx'), header=1)
            if model_type == 'LSTM_TR':
                original_clf_per_round = pd.read_excel(
                    os.path.join(model_results_dir, f'{model_type}_BERT_O',
                                 f'Results_fold_{fold}_model_{models_num_dict[model_type]["BERT_O"]}.xlsx'),
                    header=1, sheet_name=f'Model_{models_num_dict[model_type]["BERT_O"]}_round_fold_{fold}')
                original_prediction_per_round = original_clf_per_round[per_round_prediction_column].values
            for feature_num, feature in enumerate(bert_models):
                model_feature_name = os.path.join(model_results_dir, f'{model_type}_{feature}',
                                                  f'Results_fold_{fold}_model_{models_num_dict[model_type]["BERT_CF"]}_'
                                                  f'{feature_num}.xlsx')
                if os.path.exists(model_feature_name):
                    print(f'Calculate TreATE for feature {feature} and fold {fold}')
                    treated_clf_total = pd.read_excel(model_feature_name, header=1)

                    original_prediction = original_clf_total[total_payoff_prediction_column].values
                    treated_prediction = treated_clf_total[total_payoff_prediction_column].values

                    TReATE_CR, confidence_CR = get_TReATE(original_prediction, treated_prediction,
                                                          confidence_intervals=True)

                    if model_type == 'LSTM_TR':
                        treated_clf_per_round = pd.read_excel(
                            model_feature_name, header=1,
                            sheet_name=f'Model_{models_num_dict[model_type]["BERT_CF"]}_{feature_num}_round_fold_{fold}')
                        treated_prediction_per_round = treated_clf_per_round[per_round_prediction_column].values

                        TReATE_TR, confidence_TR = get_TReATE(original_prediction_per_round,
                                                              treated_prediction_per_round,
                                                              confidence_intervals=True)

                        TreATE_results = TreATE_results.append([[feature, model_type, fold, TReATE_CR, confidence_CR,
                                                                TReATE_TR, confidence_TR]])

                    else:
                        TreATE_results = TreATE_results.append([[feature, model_type, fold, TReATE_CR, confidence_CR]])

    TreATE_results.columns = ['feature', 'model_type', 'fold_number', 'TReATE_choice_rate', 'confidence_choice_rate',
                              'TReATE_per_round', 'confidence_per_round']
    TreATE_results.to_csv(os.path.join(model_results_dir, 'TreATE_results.csv'))
    TreATE_results_agg = TreATE_results.groupby(by=['feature', 'model_type'])[
        'TReATE_choice_rate', 'confidence_choice_rate', 'TReATE_per_round', 'confidence_per_round'].mean()
    TreATE_results_agg.to_csv(os.path.join(model_results_dir, 'TreATE_results_agg.csv'))
    all_model_results.to_csv(os.path.join(model_results_dir, 'all_models_results.csv'))


if __name__ == '__main__':
    main()
