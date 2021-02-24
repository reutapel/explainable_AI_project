import pandas as pd
import logging
import joblib
import os
from collections import defaultdict
import sklearn.metrics as metrics
import numpy as np


def save_model_prediction(model_to_dave, model_name: str, data_to_save: pd.DataFrame, fold_dir: str, fold: int,
                          model_num: int, table_writer, save_model: bool=True,):
    """
    Save the model predictions and the model itself
    :param data_to_save: the data to save
    :param save_model: whether to save the model
    :param sheet_prefix_name: the sheet prefix name to save
    :param fold_dir: the fold we want to save the model in
    :param element_to_save: if we want to save something that is not the model itself: {element_name: element}
    :return:
    """

    # save the model
    if save_model:
        logging.info(f'Save model {model_num}: {model_name}_fold_{fold}.pkl')
        joblib.dump(model_to_dave, os.path.join(
            fold_dir, f'{model_num}_{model_name}_fold_{fold}.pkl'))

    write_to_excel(
        table_writer, f'Model_{model_num}_{model_name}_fold_{fold}',
        headers=[f'Predictions for model {model_num} {model_name} in fold {fold}'], data=data_to_save)


def write_to_excel(table_writer: pd.ExcelWriter, sheet_name: str, headers: list, data: pd.DataFrame):
    """
    This function get header and data and write to excel
    :param table_writer: the ExcelWrite object
    :param sheet_name: the sheet name to write to
    :param headers: the header of the sheet
    :param data: the data to write
    :return:
    """
    if table_writer is None:
        return
    workbook = table_writer.book
    if sheet_name not in table_writer.sheets:
        worksheet = workbook.add_worksheet(sheet_name)
    else:
        worksheet = workbook.get_worksheet_by_name(sheet_name)
    table_writer.sheets[sheet_name] = worksheet

    data.to_excel(table_writer, sheet_name=sheet_name, startrow=len(headers), startcol=0)
    all_format = workbook.add_format({
        'valign': 'top',
        'border': 1})
    worksheet.set_column(0, data.shape[1], None, all_format)

    # headers format
    merge_format = workbook.add_format({
        'bold': True,
        'border': 2,
        'align': 'center',
        'valign': 'vcenter',
        'text_wrap': True,
    })
    for i, header in enumerate(headers):
        worksheet.merge_range(first_row=i, first_col=0, last_row=i, last_col=data.shape[1], data=header,
                              cell_format=merge_format)
        # worksheet_header = pd.DataFrame(columns=[header])
        # worksheet_header.to_excel(table_writer, sheet_name=sheet_name, startrow=0+i, startcol=0)

    return


def load_data(data_path: str, label_name: str, features_families: list,  test_pair_ids: list, train_pair_ids: list=None):
    """
    Load data from data_path and return: train_x, train_y, test_x, test_y
    :param data_path: path to data
    :param label_name: the label column name
    :param features_families: the families of feautres to use
    :param train_pair_ids: the pair ids for train data, if None- return only test data
    :param test_pair_ids: the pair ids for test data
    :return:
    """

    if 'pkl' in data_path:
        data = joblib.load(data_path)
    else:
        data = pd.read_csv(data_path)

    if train_pair_ids is not None:
        train_data = data.loc[data.pair_id.isin(train_pair_ids)]
        train_y = train_data[label_name]
        train_x = train_data[features_families]
    else:
        train_y = None
        train_x = None

    test_data = data.loc[data.pair_id.isin(test_pair_ids)]
    test_y = test_data[label_name]
    test_x = test_data[features_families]

    return train_x, train_y, test_x, test_y


def calculate_predictive_model_measures(all_predictions: pd.DataFrame, predictions_column: str='labels',
                                        label_column: str='predictions',
                                        label_options: list=['DM chose stay home', 'DM chose hotel']):
    """

    :param all_predictions: the predictions and the labels to calculate the measures on
    :param predictions_column: the name of the prediction column
    :param label_column: the name of the label column
    :param label_options: the list of the options to labels
    :return:
    """
    results_dict = defaultdict(dict)
    precision, recall, fbeta_score, support =\
        metrics.precision_recall_fscore_support(all_predictions[label_column], all_predictions[predictions_column])
    accuracy = metrics.accuracy_score(all_predictions[label_column], all_predictions[predictions_column])

    # number of DM chose stay home
    final_labels = list(range(len(support)))
    # get the labels in the all_predictions DF
    true_labels = all_predictions[label_column].unique()
    true_labels.sort()
    for label_index, label in enumerate(true_labels):
        status_size = all_predictions[label_column].where(all_predictions[label_column] == label).dropna().shape[0]
        if status_size in support:
            index_in_support = np.where(support == status_size)[0][0]
            final_labels[index_in_support] = label_options[label_index]

    # create the results to return
    for measure, measure_name in [[precision, 'precision'], [recall, 'recall'], [fbeta_score, 'Fbeta_score']]:
        for i, label in enumerate(final_labels):
            results_dict[f'{measure_name}_{label}'] = round(measure[i]*100, 2)
    results_dict[f'Accuracy'] = round(accuracy*100, 2)

    return results_dict