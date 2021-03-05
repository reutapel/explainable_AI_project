import pandas as pd
import os
from datetime import datetime
import logging
import numpy as np
import random
import scipy.sparse as sparse
import joblib
import copy
import time
import itertools
from collections import defaultdict


base_directory = os.path.abspath(os.curdir)
condition = 'verbal'
models_input = 'models_input'
data_directory = os.path.join(base_directory, condition)
save_data_directory = os.path.join(data_directory, models_input)
if not os.path.exists(save_data_directory):
    os.makedirs(save_data_directory)

logs_directory = os.path.join(base_directory, 'logs')
if not os.path.exists(logs_directory):
    os.makedirs(logs_directory)

log_file_name = os.path.join(logs_directory, datetime.now().strftime('LogFile_create_save_data_%d_%m_%Y_%H_%M_%S.log'))
logging.basicConfig(filename=log_file_name,
                    level=logging.DEBUG,
                    format='%(asctime)s: %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    )
random.seed(1)

# define the alpha for the weighted average of the history features - global and text features
# if alpha == 0: use average
alpha_text = 0.9
alpha_global = 0.8

# define global prefix and suffix names to prevent typos
global_prefix = 'raisha'
global_suffix = 'saifa'
crf_label_col_name = 'labels'
prefix_history_text_col_name = 'history_text'
prefix_history_behave_col_name = 'history_behave'
prefix_future_col_name = 'future'
prefix_suffix_col_name = 'suffix'
curr_round_col_name = 'curr_round_feature'


def rename_review_features_column(review_data: pd.DataFrame, prefix_column_name: str):
    columns_to_rename = review_data.columns.tolist()
    if 'review_id' in columns_to_rename:
        columns_to_rename.remove('review_id')
    if review_data.columns[0] == 'review_id':  # review_id first
        review_data.columns = ['review_id'] + [f'{prefix_column_name}_{i}' for i in columns_to_rename]
    elif review_data.columns[review_data.shape[1]-1] == 'review_id':  # review_if last
        review_data.columns = [f'{prefix_column_name}_{i}' for i in columns_to_rename] + ['review_id']
    else:
        logging.exception(f'in rename_review_features_column with prefix {prefix_column_name}: '
                          f'review_id is not the first or the last column')
        raise Exception(f'in rename_review_features_column with prefix {prefix_column_name}: '
                        f'review_id is not the first or the last column')

    return review_data


def create_average_history_text(rounds: list, temp_reviews: pd.DataFrame):
    """This function get the temp reviews with the review_id, round_number and the features for the reviews as column for
    each feature
    :param rounds: list of the rounds to create history average features for
    :param temp_reviews: pdDataFrame with review_id, round_number, features as columns
    :return:
    """
    history_reviews = pd.DataFrame()
    future_reviews = pd.DataFrame()
    suffix_reviews = pd.DataFrame()
    for round_num in rounds:
        review_id_curr_round = \
            temp_reviews.loc[temp_reviews.subsession_round_number == round_num].review_id
        review_id_curr_round.index = ['review_id']
        history = temp_reviews.loc[temp_reviews.subsession_round_number < round_num]
        future = temp_reviews.loc[temp_reviews.subsession_round_number > round_num]
        suffix = temp_reviews.loc[temp_reviews.subsession_round_number >= round_num]
        # history.shape[1]-2- no need subsession_round_number and review_id
        history_weights = list(pow(alpha_text, round_num - history.subsession_round_number))
        future_weights = list(pow(alpha_text, future.subsession_round_number - round_num))
        history = history.drop(['subsession_round_number', 'review_id'], axis=1)
        future = future.drop(['subsession_round_number', 'review_id'], axis=1)
        suffix = suffix.drop(['subsession_round_number', 'review_id'], axis=1)
        if alpha_text == 0:  # if alpha=0 use average
            history_mean = history.mean(axis=0)  # get the average of each column
            future_mean = future.mean(axis=0)
        else:
            history_mean = history.mul(history_weights, axis=0).mean()
            future_mean = future.mul(future_weights, axis=0).mean()
        suffix_mean = suffix.mean(axis=0)
        # concat the review_id of the current round
        if history.empty:  # round=1, no history
            history_mean = pd.Series(np.repeat(-1, history.shape[1]), index=history.columns)
        if future.empty:
            # print(f'future is empty for round number {round_num}')
            future_mean = pd.Series(np.repeat(-1, future.shape[1]), index=future.columns)
        if suffix.empty:
            suffix_mean = pd.Series(np.repeat(-1, suffix.shape[1]), index=suffix.columns)
        history_mean = history_mean.append(review_id_curr_round)
        history_reviews = pd.concat([history_reviews, history_mean], axis=1, ignore_index=True, sort=False)
        future_mean = future_mean.append(review_id_curr_round)
        future_reviews = pd.concat([future_reviews, future_mean], axis=1, ignore_index=True, sort=False)
        suffix_mean = suffix_mean.append(review_id_curr_round)
        suffix_reviews = pd.concat([suffix_reviews, suffix_mean], axis=1, ignore_index=True, sort=False)

    history_reviews = history_reviews.T
    history_reviews = rename_review_features_column(history_reviews, f'{prefix_history_text_col_name}_avg_feature')
    future_reviews = future_reviews.T
    future_reviews = rename_review_features_column(future_reviews, f'{prefix_future_col_name}_avg_feature')
    suffix_reviews = suffix_reviews.T
    suffix_reviews = rename_review_features_column(suffix_reviews, f'{prefix_suffix_col_name}_avg_feature')

    return history_reviews, future_reviews, suffix_reviews


def flat_reviews_numbers(data: pd.DataFrame, rounds: list, columns_to_drop: list, last_round_to_use: int,
                         first_round_to_use: int, total_payoff_label: bool=True, text_data: bool=False,
                         crf_prefix: bool=False, no_suffix_text: bool=False):
    """
    This function get data and flat it as for each row there is be the history of the relevant features in the data
    :param data: the data to flat
    :param rounds: the rounds to create history for
    :param columns_to_drop: the columns to drop, the first should be the column to add at last
    :param last_round_to_use: the last round to create history data for: 10 or 9
    :param first_round_to_use: the first round to use for history: the current round or the round before: 0 ot 1
    :param total_payoff_label: if the label is the decision of a single round
    :param text_data: if the data  we flat is the text representation
    :param crf_prefix: if we create data for crf with prefix features
    :param no_suffix_text: if we don't want to use the text of the suffix rounds
    :return:
    """
    all_history = pd.DataFrame()
    all_history_dict = dict()
    for round_num in rounds:
        id_curr_round = data.loc[data.subsession_round_number == round_num][columns_to_drop[0]]
        id_curr_round.index = ['review_id']
        # this features are not relevant for the last round because they are post treatment features
        data_to_flat = data.copy(deep=True)
        data_to_flat = data_to_flat.reset_index(drop=True)
        # the last round is not relevant for the history of any other rounds
        data_to_flat = data_to_flat.loc[data_to_flat.subsession_round_number <= last_round_to_use]
        data_to_flat = data_to_flat.drop(columns_to_drop, axis=1)
        # for current and next rounds put -1 --> use also current if first_round_to_use=0
        # and not use if first_round_to_use=1
        # if we predict all the future payoff, so the future text can be seen
        # if we want the text only from the prefix rounds and the current round - put -1 for the suffix
        if (not total_payoff_label and text_data) or (not text_data) or (text_data and no_suffix_text):
            data_to_flat.iloc[list(range(round_num - first_round_to_use, last_round_to_use))] = -1
        if crf_prefix:  # for suffix review put list of -1
            columns_to_use = data_to_flat.columns.tolist()
            columns_to_use.remove('review_features')
            for i in range(round_num - first_round_to_use, last_round_to_use):
                data_to_flat.at[i, 'review_features'] = [-1] * data.iloc[0]['review_features'].shape[0]
            prefix_data_list = list()
            for index, row in data_to_flat.iterrows():
                review_features = data_to_flat.review_features.iloc[index]
                if type(review_features) != list:
                    review_features = review_features.tolist()
                review_features.extend(data_to_flat[columns_to_use].iloc[index].to_list())
                prefix_data_list.extend(review_features)
            all_history_dict[round_num] = prefix_data_list
        else:
            data_to_flat.index = data_to_flat.index.astype(str)
            # data_to_flat.columns = data_to_flat.columns.astype(str)
            data_to_flat = data_to_flat.unstack().to_frame().sort_index(level=1).T
            data_to_flat.columns = [f'{str(col)}_{i}' for col, i in zip(data_to_flat.columns.get_level_values(0),
                                                                        data_to_flat.columns.get_level_values(1))]
            # concat the review_id of the current round
            data_to_flat = data_to_flat.assign(review_id=id_curr_round.values)
            all_history = pd.concat([all_history, data_to_flat], axis=0, ignore_index=True, sort=False)
    if crf_prefix:
        all_history = pd.Series(all_history_dict)
        all_history = pd.DataFrame(all_history)

    return all_history


def split_pairs_to_data_sets(load_file_name: str, k_folds: int=6, only_train_val: bool=False, id_column: str='pair_id',
                             directory: str=data_directory):
    """
    Split all the pairs to data sets: train, validation, test for 6 folds
    :param load_file_name: the raw data file name
    :param k_folds: number of folds to split the data
    :param only_train_val: if we want to split data to only train and validation, without test data
    (if test data is in a seperate data set)
    :param id_column: the name of the ID column
    :param directory: the directory the data is saved in
    :return:
    """
    print(f'Start create and save data for file: {os.path.join(directory, f"{load_file_name}")}')
    if 'csv' in load_file_name:
        data = pd.read_csv(os.path.join(directory, f'{load_file_name}'))
    elif 'pkl' in load_file_name:
        data = joblib.load(os.path.join(directory, f'{load_file_name}'))
    else:
        raise NameError('file type must be csv or pkl')
    if 'status' in data.columns:
        data = data.loc[(data.status == 'play') & (data.player_id_in_group == 2)]
    data = data.drop_duplicates()
    if len(data.columns.names) == 2:
        data.columns = data.columns.droplevel()
    pairs = pd.DataFrame(data[id_column].unique(), columns=[id_column])
    pairs = pairs.sample(frac=1)
    pairs = pairs.assign(fold_number=0)
    paris_list = pairs[id_column].unique()
    for k in range(k_folds):
        pairs.loc[pairs[id_column].isin([x for i, x in enumerate(paris_list) if i % k_folds == k]), 'fold_number'] = k

    # split pairs to folds - train, test, validation in each fold
    for k in range(k_folds):
        if only_train_val:
            pairs.loc[pairs.fold_number == k, f'fold_{k}'] = 'validation'
            pairs.loc[pairs.fold_number != k, f'fold_{k}'] = 'train'
        else:
            pairs.loc[pairs.fold_number == k, f'fold_{k}'] = 'test'
            if k != k_folds-1:
                pairs.loc[pairs.fold_number == k + 1, f'fold_{k}'] = 'validation'
                pairs.loc[~pairs.fold_number.isin([k, k + 1]), f'fold_{k}'] = 'train'
            else:
                pairs.loc[pairs.fold_number == 0, f'fold_{k}'] = 'validation'
                pairs.loc[~pairs.fold_number.isin([k, 0]), f'fold_{k}'] = 'train'

    return pairs


class CreateSaveData:
    """
    This class load the data, create the seq data and save the new data with different range of K
    """
    def __init__(self, load_file_name: str, features_files_dict: dict, total_payoff_label: bool=True,
                 features_file_list: list=list(), use_all_history: bool = False, label: str='total_payoff',
                 use_all_history_text_average: bool = False,
                 use_all_history_text: bool=False, use_all_history_average: bool = False,
                 use_prefix_suffix_setting: bool=False, features_to_drop: list=None,
                 suffix_average_text: bool=False, no_suffix_text: bool=False,
                 non_nn_turn_model: bool=False, transformer_model: bool=False,
                 prefix_data_in_sequence: bool=False, data_type='train_data', no_decision_features: bool=False,
                 suffix_no_current_round_average_text: bool=False):
        """
        :param load_file_name: the raw data file name
        :param features_files_dict: dict of features files and types
        :param total_payoff_label: if the label is the total payoff of the expert or the next rounds normalized payoff
        :param label: the name of the label
        :param features_file_list: if using fix features- the name of the features file
        :param use_all_history: if to add some numeric features regarding the history decisions and lottery
        :param use_all_history_average: if to add some numeric features regarding the history decisions and lottery as
        average over the history
        :param use_all_history_text: if to use all the history text features
        :param use_all_history_text_average: if to use the history text as average over all the history
        :param use_prefix_suffix_setting: if we create data for crf model with fixed prefix
        :param features_to_drop: a list of features to drop
        :param suffix_average_text:  if we want to add the suffix average text features
        :param no_suffix_text: if we don't want to use the text of the suffix rounds
        :param non_nn_turn_model: non neural networks models that predict a label for each round
        :param transformer_model: create data for transformer model --> create features for prefix rounds too
        :param prefix_data_in_sequence: if the prefix data is not in the suffix features but in the seq
        :param no_decision_features: if we want to check models without decision features
        :param suffix_no_current_round_average_text: if we want the average of all suffix text
        """
        print(f'Start create and save data for file: '
              f'{os.path.join(data_directory, f"{load_file_name}_{data_type}.csv")}')
        logging.info('Start create and save data for file: {}'.
                     format(os.path.join(data_directory, f'{load_file_name}_{data_type}.csv')))

        self.data = pd.read_csv(os.path.join(data_directory, f'{load_file_name}_{data_type}.csv'))  # , usecols=columns_to_use)
        print(f'Number of rows in data: {self.data.shape[0]}')
        self.data = self.data.loc[(self.data.status == 'play') & (self.data.player_id_in_group == 2)]
        print(f'Number of rows in data: {self.data.shape[0]} after keep only play and decision makers')
        self.data = self.data.drop_duplicates()
        print(f'Number of rows in data: {self.data.shape[0]} after drop duplicates')

        # get manual text features
        reviews_features_files_list = list()
        print(f'Load features from: {features_file_list}')
        for features_file in features_file_list:
            features_file_type = features_files_dict[features_file]
            if features_file_type == 'pkl':
                reviews_features_files_list.append(joblib.load(os.path.join(
                    data_directory, f'{features_file}_{data_type}.{features_file_type}')))
            elif features_file_type == 'xlsx':
                features = pd.read_excel(os.path.join(
                    data_directory, f'{features_file}_{data_type}.{features_file_type}'))
                if data_type == 'test_data':  # change order to be the same as in the train data
                    train_features = pd.read_excel(
                        os.path.join(data_directory, f'{features_file}_train_data.{features_file_type}'))
                    features = features[train_features.columns]
                reviews_features_files_list.append(features)

            else:
                print('Features file type is has to be pkl or xlsx')
                return
        # get manual text features
        for index, reviews_features_file in enumerate(reviews_features_files_list):
            if 'review' in reviews_features_file:
                reviews_features_file = reviews_features_file.drop('review', axis=1)
            if 'score' in reviews_features_file:
                reviews_features_file = reviews_features_file.drop('score', axis=1)

            if reviews_features_file.shape[1] == 2:  # Bert features -> flat the vectors
                reviews = pd.DataFrame()
                for i in reviews_features_file.index:
                    temp = pd.DataFrame(reviews_features_file.at[i, 'review_features']).append(
                        pd.DataFrame([reviews_features_file.at[i, 'review_id']], index=['review_id']))
                    reviews = pd.concat([reviews, temp], axis=1, ignore_index=True)

                reviews_features_files_list[index] = reviews.T
            else:  # manual features
                if features_to_drop is not None:
                    reviews_features_files_list[index] = reviews_features_file.drop(features_to_drop, axis=1)

        if len(reviews_features_files_list) == 1:  # hand crafted features
            self.reviews_features = reviews_features_files_list[0]
        elif len(reviews_features_files_list) == 2:  # BERT features
            self.reviews_features = reviews_features_files_list[0].merge(reviews_features_files_list[1],
                                                                         on='review_id')
        else:
            print(f"Can't create reviews features with {len(reviews_features_files_list)} feature types")

        # calculate expert total payoff --> the label
        self.data['exp_payoff'] = self.data.group_receiver_choice.map({1: 0, 0: 1})
        total_exp_payoff = self.data.groupby(by='pair_id').agg(
            total_exp_payoff=pd.NamedAgg(column='exp_payoff', aggfunc=sum))
        self.data = self.data.merge(total_exp_payoff, how='left', right_index=True, left_on='pair_id')
        self.data['10_result'] = np.where(self.data.group_lottery_result == 10, 1, 0)
        self.data = self.data[['pair_id', 'total_exp_payoff', 'subsession_round_number', 'group_sender_answer_reviews',
                               'exp_payoff', 'group_lottery_result', 'review_id', 'previous_round_lottery_result',
                               'previous_round_decision', 'group_average_score',
                               'lottery_result_low', 'lottery_result_med1', 'previous_round_lottery_result_low',
                               'previous_round_lottery_result_high', 'previous_average_score_low',
                               'previous_average_score_high', 'previous_round_lottery_result_med1',
                               'group_sender_payoff', 'lottery_result_high',
                               'chose_lose', 'chose_earn', 'not_chose_lose', 'not_chose_earn',
                               'previous_score', 'group_sender_answer_scores', '10_result']]
        # 'time_spent_low', 'time_spent_high',
        self.final_data = pd.DataFrame()
        self.pairs = pd.Series(self.data.pair_id.unique())
        self.total_payoff_label = total_payoff_label
        self.label = label
        self.data_type = data_type
        self.number_of_rounds = 10
        self.features_file_list = features_file_list
        self.use_all_history = use_all_history
        self.use_all_history_average = use_all_history_average
        self.use_all_history_text_average = use_all_history_text_average
        self.use_all_history_text = use_all_history_text
        self.suffix_average_text = suffix_average_text
        self.suffix_no_current_round_average_text = suffix_no_current_round_average_text
        self.no_suffix_text = no_suffix_text
        self.non_nn_turn_model = non_nn_turn_model
        self.transformer_model = transformer_model
        self.prefix_data_in_sequence = prefix_data_in_sequence
        self.decisions_payoffs_columns = ['exp_payoff', 'lottery_result_high', 'lottery_result_low',
                                          'lottery_result_med1', 'chose_lose', 'chose_earn', 'not_chose_lose',
                                          'not_chose_earn']
        if no_decision_features:
            self.decisions_payoffs_columns = list()
        print(f'Number of pairs in data: {self.pairs.shape}')

        self.history_columns = list()
        if self.use_all_history_average:
            self.set_all_history_average_measures()

        # create file names:
        file_name_component = [f'{self.label}_label_',
                               'prefix_suffix_' if use_prefix_suffix_setting else '',
                               'non_nn_turn_model_' if self.non_nn_turn_model else '',
                               'transformer_' if self.transformer_model else '',
                               f'all_history_features_' if self.use_all_history else '',
                               f'all_history_features_avg_with_global_alpha_{alpha_global}_'
                               if self.use_all_history_average else '',
                               f'all_history_text_avg_with_alpha_{alpha_text}_' if
                               self.use_all_history_text_average else '',
                               'prefix_in_seq_' if prefix_data_in_sequence else '',
                               f'no_suffix_text_' if self.no_suffix_text else '',
                               f'all_suffix_text_average_' if self.suffix_average_text else '',
                               f'all_history_text_' if self.use_all_history_text else '',
                               f'{self.features_file_list}_',
                               'no_decision_features_' if no_decision_features else 'use_decision_features_',
                               f'{condition}_{data_type}']
        self.base_file_name = ''.join(file_name_component)
        print(f'Create data for: {self.base_file_name}')
        return

    def set_all_history_average_measures(self):
        """
        This function calculates some measures about all the history per round for each pair
        :return:
        """

        print('Start set_all_history_average_measures')
        columns_to_calc = self.decisions_payoffs_columns
        # Create only for the experts and then assign to both players
        columns_to_chose = columns_to_calc + ['pair_id', 'subsession_round_number']
        data_to_create = self.data[columns_to_chose]

        pairs = data_to_create.pair_id.unique()
        history_data = pd.DataFrame()
        for pair in pairs:
            pair_data = data_to_create.loc[data_to_create.pair_id == pair]
            for round_num in range(1, 11):
                history_data_dict = defaultdict(dict)
                history_data_dict[pair]['subsession_round_number'] = round_num
                history = pair_data.loc[pair_data.subsession_round_number < round_num]
                weights = pow(alpha_global, round_num - history.subsession_round_number)
                for column in columns_to_calc:
                    if column == 'lottery_result':
                        j = 1
                    else:
                        j = 1
                    if alpha_global == 0:  # if alpha == 0: use average
                        history_data_dict[pair][f'{prefix_history_behave_col_name}_{column}'] =\
                            round(history[column].mean(), 2)
                    else:
                        history_data_dict[pair][f'{prefix_history_behave_col_name}_{column}'] =\
                            (pow(history[column], j) * weights).mean()

                    # for the first round put -1 for the history
                    if round_num == 1:
                        history_data_dict[pair][f'{prefix_history_behave_col_name}_{column}'] = -1
                pair_history_data = pd.DataFrame.from_dict(history_data_dict).T
                history_data = history_data.append(pair_history_data)

        history_data['pair_id'] = history_data.index
        self.history_columns = list(set(history_data.columns) - set(['pair_id', 'subsession_round_number']))
        self.data = self.data.merge(history_data, on=['pair_id', 'subsession_round_number'],  how='left')

        print('Finish set_all_history_average_measures')

        return

    def set_text_average(self, rounds, reviews_features, data):
        """
        Create data frame with the history and future average text features and the current round text features
        :param rounds: the rounds to use
        :param reviews_features: the reviews features of this pair
        :param data: the data we want to merge to
        :return:
        """
        history_reviews, future_reviews, suffix_reviews =\
            create_average_history_text(rounds, reviews_features)
        # add the current round reviews features
        reviews_features = reviews_features.drop('subsession_round_number', axis=1)
        reviews_features = rename_review_features_column(reviews_features, curr_round_col_name)
        if not self.suffix_no_current_round_average_text:
            data = data.merge(reviews_features, on='review_id', how='left')
        data = data.merge(history_reviews, on='review_id', how='left')
        if self.suffix_average_text:
            data = data.merge(future_reviews, on='review_id', how='left')
        elif self.suffix_no_current_round_average_text:
            data = data.merge(suffix_reviews, on='review_id', how='left')

        return data

    def create_features_per_review(self):
        """
        This function create for each review:
        1. label: proportion of DMs choose hotel when read this review
        2. HC features
        :return:
        """
        print(f'Start creating information for each review')
        logging.info('Start creating information for each review')

        meta_data_columns = ['review_id', 'group_sender_answer_reviews']
        data_per_review = self.data.groupby(by=['review_id']). \
            agg(review=pd.NamedAgg(column='group_sender_answer_reviews', aggfunc=sum),
                label=pd.NamedAgg(column='exp_payoff', aggfunc='mean'))
        data_per_review.rename(columns={'label': self.label}, inplace=True)
        self.final_data = data_per_review.merge(self.reviews_features, on='review_id')
        feature_columns = self.reviews_features.columns.tolist()
        feature_columns.remove('review_id')
        self.final_data.review_id = self.final_data.review_id.map(str)
        self.final_data.review_id =\
            self.final_data.review_id.str.cat([self.data_type]*len(self.final_data.review_id), sep='_')
        multi_level_column_names = list()
        multi_level_column_names.extend([('meta_data', column) for column in meta_data_columns])
        multi_level_column_names.append(('label', self.label))
        multi_level_column_names.extend([('text_features', column) for column in feature_columns])
        self.final_data.columns = pd.MultiIndex.from_tuples(multi_level_column_names, names=('high', 'low'))

        # save final data
        file_name = f'all_data_{self.base_file_name}'
        self.final_data.to_csv(os.path.join(save_data_directory, f'{file_name}.csv'), index=False)
        joblib.dump(self.final_data, os.path.join(save_data_directory, f'{file_name}.pkl'))

        print(f'Finish creating manual features data: {file_name}.pkl')
        logging.info('Finish creating manual features data')

        return

    def create_info_df_per_pair_round(self):
        """
        This function create for each round of each pair a df with the following information:
        1. Vector of history features (behavioral + text)
        2. Handcrafted textual features vector for text in round r
        3. Plain text for text in round r
        4. Label y (decision in round r)

        :return:
        """

        print(f'Start creating information for each round of each pair')
        logging.info('Start creating information for each round of each pair')

        meta_data_columns = ['subsession_round_number', 'pair_id', 'sample_id']
        history_features_columns = self.history_columns
        rounds = list(range(1, 11))

        columns_to_use = ['review_id', 'group_sender_answer_reviews', 'subsession_round_number']

        if self.use_all_history_average:
            columns_to_use = columns_to_use + history_features_columns

        for pair in self.pairs:
            data = self.data.loc[self.data.pair_id == pair][columns_to_use]

            # concat history numbers
            if self.use_all_history:
                temp_numbers = self.data.loc[self.data.pair_id == pair][self.decisions_payoffs_columns +
                                                                        ['subsession_round_number']]
                temp_numbers = temp_numbers.reset_index(drop=True)
                all_history = flat_reviews_numbers(temp_numbers, rounds, columns_to_drop=['subsession_round_number'],
                                                   last_round_to_use=9, first_round_to_use=1)
                # first_round_to_use=1 because the numbers of the current round should not be in the features
                all_history.rename(columns={'review_id': 'subsession_round_number'}, inplace=True)
                data = data.merge(all_history, on='subsession_round_number', how='left')

            # first merge for the review_id for the current round
            temp_reviews = data[['review_id', 'subsession_round_number']].copy(deep=True)
            temp_reviews = temp_reviews.merge(self.reviews_features, on='review_id', how='left')
            if self.use_all_history_text_average:
                data = self.set_text_average(rounds, temp_reviews, data)

            if self.use_all_history_text:
                history_reviews = flat_reviews_numbers(
                    temp_reviews, rounds, columns_to_drop=['review_id', 'subsession_round_number'],
                    last_round_to_use=10, first_round_to_use=0, text_data=True,
                    total_payoff_label=self.total_payoff_label, no_suffix_text=self.no_suffix_text)
                data = data.merge(history_reviews, on='review_id', how='left')
            if self.no_suffix_text and not self.suffix_average_text:
                # remove curr_round_features because we have all suffix features
                curr_round_features_columns = [column for column in data.columns if curr_round_col_name in column]
                data = data.drop(curr_round_features_columns, axis=1)

            if not self.use_all_history_text and not self.use_all_history_text_average:  # no history text
                temp_reviews =\
                    self.data.loc[self.data.pair_id == pair][['review_id', 'subsession_round_number']]
                data = temp_reviews.merge(self.reviews_features, on='review_id', how='left')

            data = data.drop('review_id', axis=1)

            # add metadata
            self.data['sample_id'] = self.data[meta_data_columns[1]] + '_' + (
                        self.data['subsession_round_number'] - 1).map(str)
            # add sample ID column
            data[meta_data_columns[1]] = pair
            data[meta_data_columns[2]] = data[meta_data_columns[1]] + '_' + data[meta_data_columns[0]].map(str)
            if self.label == 'single_round':
                # the label is the exp_payoff of the current round - 1 or -1
                data = data.merge(self.data[['exp_payoff', 'sample_id']], how='left', on='sample_id')
                data.rename(columns={'exp_payoff': self.label}, inplace=True)
                # if 1 not in data.subsession_round_number.values:
                #     data[self.label] =\
                #         self.data.loc[(self.data.pair_id == pair) & (self.data.subsession_round_number > 1)][
                #             'exp_payoff'].reset_index(drop=True)
                # else:
                #     data[self.label] = self.data.loc[(self.data.pair_id == pair)]['exp_payoff'].reset_index(drop=True)
                data[self.label] = np.where(data[self.label] == 1, 1, -1)

            else:  # the label is the total payoff
                for i in rounds:  # the prefixs are 0-9
                    data.loc[data[global_prefix] == i-1, self.label] =\
                        self.data.loc[(self.data.pair_id == pair) &
                                      (self.data.subsession_round_number >= i)].group_sender_payoff.sum() /\
                        (self.number_of_rounds + 1 - i)
            # concat to all data
            self.final_data = pd.concat([self.final_data, data], axis=0, ignore_index=True)

        # sort columns according to the round number
        if self.use_all_history_text or self.use_all_history:
            columns_to_sort = self.reviews_features.columns.values.tolist() + self.decisions_payoffs_columns
            if 'review_id' in columns_to_sort:
                columns_to_sort.remove('review_id')
            if 'review' in columns_to_sort:
                columns_to_sort.remove('review')
            if self.use_all_history_average or self.use_all_history_text_average:
                columns_order = [column for column in self.final_data.columns if 'history' in column]
            else:
                columns_order = list()
            for round_num in rounds:
                for column in columns_to_sort:
                    if f'{column}_{round_num-1}' in self.final_data.columns:
                        columns_order.append(f'{column}_{round_num-1}')

            meta_data_columns.append(self.label)
            columns_order.extend(meta_data_columns)
            self.final_data = self.final_data[columns_order]

        # create multi level columns to make it easier to use the df later
        history_behave_features =\
            [column for column in self.final_data.columns if prefix_history_behave_col_name in column]
        history_text_features = [column for column in self.final_data.columns if prefix_history_text_col_name in column]
        current_round_text_features = [column for column in self.final_data.columns if curr_round_col_name in column]
        final_data_columns = meta_data_columns + history_behave_features + history_text_features + \
                             current_round_text_features + ['group_sender_answer_reviews', self.label]
        self.final_data = self.final_data[final_data_columns]
        # create tuples for multi level columns
        multi_level_column_names = list()
        multi_level_column_names.extend([('meta_data', column) for column in meta_data_columns])
        multi_level_column_names.extend([('history_behave_features', column) for column in history_behave_features])
        multi_level_column_names.extend([('history_text_features', column) for column in history_text_features])
        multi_level_column_names.extend([('current_text_features', column) for column in current_round_text_features])
        multi_level_column_names.append(('plain_text', 'group_sender_answer_reviews'))
        multi_level_column_names.append(('label', self.label))
        self.final_data.columns = pd.MultiIndex.from_tuples(multi_level_column_names, names=('high', 'low'))
        # # save column names to get the features later
        # file_name = f'features_{self.base_file_name}'
        # features_columns = self.final_data.columns.tolist()
        # columns_to_drop = meta_data_columns + ['subsession_round_number', self.label]
        # for column in columns_to_drop:
        #     if column in features_columns:
        #         features_columns.remove(column)
        # pd.DataFrame(features_columns).to_excel(os.path.join(save_data_directory, f'{file_name}.xlsx'), index=True)

        # save final data
        file_name = f'all_data_{self.base_file_name}'
        self.final_data.to_csv(os.path.join(save_data_directory, f'{file_name}.csv'), index=False)
        joblib.dump(self.final_data, os.path.join(save_data_directory, f'{file_name}.pkl'))

        print(f'Finish creating manual features data: {file_name}.pkl')
        logging.info('Finish creating manual features data')

        return

    def split_data(self):
        """
        Split the pairs into train-validation-test data
        :return:
        """

        print(f'Start split data to train-test-validation data and save for k=1-10 and k=1-9')
        logging.info('Start split data to train-test-validation data and save for k=1-10 and k=1-9')

        train_pairs, validation_pairs, test_pairs = np.split(self.pairs.sample(frac=1),
                                                           [int(.6 * len(self.pairs)), int(.8 * len(self.pairs))])
        for data_name, pairs in [['train', train_pairs], ['validation', validation_pairs], ['test', test_pairs]]:
            data = self.final_data.loc[self.final_data.pair_id.isin(pairs)]
            # save 10 sequences per pair

            file_name = f'{data_name}_data_1_{self.number_of_rounds}_{self.base_file_name}'
            data.to_csv(os.path.join(save_data_directory, f'{file_name}.csv'), index=False)
            joblib.dump(data, os.path.join(save_data_directory, f'{file_name}.pkl'))
            # save 9 sequences per pair
            if not self.label == 'single_round':
                seq_len_9_data = data.loc[data[global_prefix] != self.number_of_rounds]
                columns_to_drop = [column for column in seq_len_9_data.columns
                                   if str(self.number_of_rounds - 1) in column]
                seq_len_9_data = seq_len_9_data.drop(columns_to_drop, axis=1)
                seq_len_9_data.to_csv(os.path.join(
                    save_data_directory, f'{data_name}_data_1_{self.number_of_rounds-1}_{self.base_file_name}.csv'),
                    index=False)
                joblib.dump(seq_len_9_data,
                            os.path.join(save_data_directory,
                                         f'{data_name}_data_1_{self.number_of_rounds-1}_{self.base_file_name}.pkl'))

        print(f'Finish split data to train-test-validation data and save for k=1-10 and k=1-9')
        logging.info('Finish split data to train-test-validation data and save for k=1-10 and k=1-9')


def main():
    features_files = {
        'hand_crafted_features': 'xlsx',
        'bert_embedding': 'pkl',
    }
    # features_to_use can be: ['hand_crafted_features'], ['bert_embedding'], or any combination of them
    features_to_use = ['hand_crafted_features']
    conditions_dict = {
        'verbal': {
                   'use_all_history_average': False,  # use all the average of the all the prefix trials
                   'use_all_history': False,  # use all the behavioral features of the all the prefix trials
                   'use_all_history_text_average': False,  # use all the average of the all the prefix trials texr
                   'use_all_history_text': False,  # use all the text of the all the prefix trials
                   'suffix_average_text': False,  # use the suffix trials average textual features
                   'no_suffix_text': False,  # don't use the suffix text
                   'no_decision_features': False,  # if we want to check models without decision features
                   'non_nn_turn_model': False,  # non neural networks models that predict a label for each round
                   'transformer_model': False,   # for transformer models
                   'prefix_data_in_sequence': False,  # if the prefix data is not in the suffix features but in the seq
                   'suffix_no_current_round_average_text': False,  # use the average text of all suffix trials in SVM-CR
                   'label': 'proportion',  # single_round, future_total_payoff, proportion for review's label
                   'per_review_proportion': True,
                   }
    }
    use_prefix_suffix_setting = False  # relevant to all sequential models in the prefix_suffix setting
    data_type = ['train_data', 'test_data']  # it can be train_data or test_data
    total_payoff_label = False if conditions_dict[condition]['label'] == 'single_round' else True
    features_to_drop = []  # if we don't want to use some of the features
    only_split_data = False  # if we just want to split data into folds and not create data
    if only_split_data:
        pairs_folds = split_pairs_to_data_sets('raw', only_train_val=True)
        pairs_folds.to_csv(os.path.join(save_data_directory, 'pairs_folds_new_test_data.csv'))
        return

    all_data = pd.DataFrame()
    for dtype in data_type:
        create_save_data_obj = CreateSaveData(
            'raw', total_payoff_label=total_payoff_label,
            label=conditions_dict[condition]['label'], features_files_dict=features_files,
            features_file_list=features_to_use,
            use_prefix_suffix_setting=use_prefix_suffix_setting,
            use_all_history=conditions_dict[condition]['use_all_history'],
            use_all_history_average=conditions_dict[condition]['use_all_history_average'],
            use_all_history_text_average=conditions_dict[condition]['use_all_history_text_average'],
            use_all_history_text=conditions_dict[condition]['use_all_history_text'],
            features_to_drop=features_to_drop,
            suffix_average_text=conditions_dict[condition]['suffix_average_text'],
            no_suffix_text=conditions_dict[condition]['no_suffix_text'],
            non_nn_turn_model=conditions_dict[condition]['non_nn_turn_model'],
            transformer_model=conditions_dict[condition]['transformer_model'],
            prefix_data_in_sequence=conditions_dict[condition]['prefix_data_in_sequence'],
            data_type=dtype,
            no_decision_features=conditions_dict[condition]['no_decision_features'],
            suffix_no_current_round_average_text=conditions_dict[condition]['suffix_no_current_round_average_text'])

        if conditions_dict[condition]['per_review_proportion']:
            create_save_data_obj.create_features_per_review()
            all_data = all_data.append(create_save_data_obj.final_data)
        else:
            create_save_data_obj.create_info_df_per_pair_round()  # SVM-TR, MED, AVG

    if conditions_dict[condition]['per_review_proportion']:
        file_name = f'all_data_{create_save_data_obj.base_file_name}'
        all_data.to_csv(os.path.join(save_data_directory, f'{file_name}.csv'), index=False)
        joblib.dump(all_data, os.path.join(save_data_directory, f'{file_name}.pkl'))
        pairs_folds = split_pairs_to_data_sets(load_file_name=f'{file_name}.pkl', directory=save_data_directory,
                                               id_column='review_id')
        pairs_folds.to_csv(os.path.join(save_data_directory, 'reviews_folds.csv'))


if __name__ == '__main__':
    main()
