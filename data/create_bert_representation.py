import torch
from pytorch_transformers import *
import pandas as pd
import datetime
import numpy as np
import math
import os
import joblib


def average_vectors(vectors_list):
    """ assumes tensors are in the same dimensions"""
    return np.mean(vectors_list, axis=0)


def variable_initializer(is_list):
    if is_list:
        return list()
    else:
        return str()


def split_list_by_token(list_input, token):

    index_list = [index+1 for index, value in enumerate(list_input) if value == token]
    partition = np.split(list_input, index_list)

    return [list(x) for x in partition if len(x) > 0]


def split_text_by_max_size(text, max_size, split_token, is_list=False):
    """
    splits text or list to parts by split_token , wheres part is maximum max_size number of tokens
    :param text:
    :param max_size:
    :param split_token: split text by its instances
    :param is_list: if list manage differently
    :return: list of parts of text
    """
    # list of text sub strings of maximum length of max_size
    text_parts_list = list()
    # assuming sentences in text are divided by split token
    if is_list:
        sentence_list = split_list_by_token(text, split_token)
    else:
        sentence_list = text.split(split_token)

    # current part of text to concatenate sentences
    current_part = variable_initializer(is_list)

    # length of current part
    current_part_len = 0
    while sentence_list:
        # take next sentence
        current_sentence = sentence_list.pop(0)
        # check if empty string
        if len(current_sentence) == 0:
            continue
        # length of current sentence
        if is_list:
            current_sentence_tokens_count = len(current_sentence)
        else:
            current_sentence_tokens_count = len(current_sentence.split(' '))
        if current_part_len + current_sentence_tokens_count <= max_size:
            current_part = current_part + current_sentence
            current_part_len += current_sentence_tokens_count
        else:
            # insert last legitimate part
            text_parts_list.append(current_part) if current_part_len > 0 else None
            # check that current sentence is in legitimate length
            if len(current_sentence) <= max_size:
                # update incremental part with last legitimate length of current sentence
                current_part = current_sentence  # TODO: debug with string and with long comments, make sure doesnt loose parts in the middle
                current_part_len = current_sentence_tokens_count
            else:
                # split current sentence to valid lengths
                num_split = math.ceil(len(current_sentence) / max_size)
                splited_current_sentence = [l.tolist() for l in np.array_split(current_sentence, num_split)]
                text_parts_list += splited_current_sentence
                current_part = variable_initializer(is_list)
                current_part_len = 0
    # in case entire text never crossed max_size or last part didn't..
    if current_part in text_parts_list:
        return text_parts_list
    else:
        text_parts_list.append(current_part) if current_part_len > 0 else None
        return text_parts_list


class BertTransformer:
    """ creates embedded version of submission titles and clusters them """

    def __init__(self, pretrained_weights_shortcut='bert-base-cased', take_bert_pooler=True,
                 bert_state_dict_path: str=None):

        self.Pretrained_weights_shortcut = pretrained_weights_shortcut
        self.take_bert_pooler = take_bert_pooler
        self.poolers_df = None

        # TODO: change to be able to choose which model / tokenizer for downstream tasks
        # Load pre trained model/tokenizer

        self.tokenizer = BertTokenizer.from_pretrained(self.Pretrained_weights_shortcut)

        if bert_state_dict_path is not None:
            fine_tuned_state_dict = torch.load(bert_state_dict_path)
            self.model = BertModel.from_pretrained(self.Pretrained_weights_shortcut, state_dict=fine_tuned_state_dict)
        else:
            self.model = BertModel.from_pretrained(self.Pretrained_weights_shortcut)

    def bert_text_encoding(self, text, take_bert_pooler=True):
        """
        Embeded text with BERT pre trained model and tokenizer that is defined in init
        :param text: text to embedd
        :param take_bert_pooler: take pooler - sum of sentence or rest of embedded tensor for sentence tokens
        :return: requested embedded part
        """
        # Encode text
        print('starting BERT encoding', datetime.datetime.now())
        tokenized = self.tokenizer.tokenize(text)
        tokenized.insert(0, 'CLS')
        input_ids = self.tokenizer.convert_tokens_to_ids(tokenized)
        with torch.no_grad():
            if take_bert_pooler:
                pooler = self.model(torch.tensor([input_ids]))[0][0][0]
                print('finished BERT encoding, returning BERT pooler', datetime.datetime.now())
                return pooler
            else:
                # TODO: validate tokens [1:]
                tokens_embedding = self.model(torch.tensor([input_ids]))[0][0][1:]
                print('finished BERT encoding, returning BERT sentence', datetime.datetime.now())
                return tokens_embedding

    def get_text_average_pooler(self, text, max_size, take_bert_pooler=True):
        """
        this method gets a text, splits it to allowed size, encode, gets the poolers and average them
        :param text:
        :param max_size:
        :param take_bert_pooler:
        :return: average pooler
        """
        poolers_list = list()
        # get splited text
        text_parts_list = split_text_by_max_size(text, max_size, split_token='.', is_list=False)
        # encode each text part
        for text_part in text_parts_list:
            text_part_pooler = self.bert_text_encoding(text_part, take_bert_pooler)
            poolers_list.append(pd.Series(text_part_pooler))
            # poolers_df = pd.DataFrame(poolers_list)
        # average all poolers of text
        average_pooler = average_vectors(poolers_list)
        return average_pooler

    def get_text_average_pooler_split_bert(self, text, max_size, split_id=1012):
        """
        this method gets a text, splits it to allowed size, encode, gets the poolers and average them
        :param text:
        :param max_size:
        :param split_id: the BERT token id to split by before transforming the parts of the sentence, for example 1012
        is the id of '.'
        :return: average pooler
        """
        poolers_list = list()

        # Encode text
        # print('starting BERT encoding', datetime.datetime.now())
        tokenized = self.tokenizer.tokenize(text)
        # tokenized.insert(0, 'CLS')
        input_ids = self.tokenizer.convert_tokens_to_ids(tokenized)
        input_ids_list = list()
        input_ids_list.append(input_ids)
        print(f'input_ids length is {len(input_ids)}')

        if len(input_ids) > max_size:
            if split_id in input_ids:

                input_ids_list = split_text_by_max_size(text=input_ids, max_size=max_size, split_token=split_id,
                                                        is_list=True)
            else:
                # splitting just by max size since there is no split_id in
                # TODO: validate list of lists
                input_ids_list = [input_ids[x:x + max_size] for x in range(0, len(input_ids), max_size)]

        # encode each text part
        for input_ids_sub in input_ids_list:
            with torch.no_grad():
                # add id 100 to get pooler
                input_ids_sub.insert(0, 100)
                pooler = self.model(torch.tensor([input_ids_sub]).to(torch.int64))[0][0][0]
                # print('finished BERT encoding, returning BERT pooler', datetime.datetime.now())
                poolers_list.append(pd.Series(pooler))
        # average all poolers of text
        average_pooler = average_vectors(poolers_list)
        return average_pooler


def main():
    base_directory = os.path.abspath(os.curdir)
    data_directory = os.path.join(base_directory, 'verbal')
    for data_type in ['train', 'test']:
        reviews = pd.read_excel(os.path.join(data_directory, f'hand_crafted_features_{data_type}_data.xlsx'),
                                usecols=['review_id', 'review'])
        # reviews = pd.read_csv('labeledTrainData.csv')
        models_path = os.path.join(base_directory, 'bert_models')
        bert_models = ['topic_price_positive']
        for model in bert_models:
            bert_state_dict_path = os.path.join(models_path, model, 'model', 'pytorch_model.bin')
            bert_model = BertTransformer(bert_state_dict_path=bert_state_dict_path)
            reviews = reviews.assign(review_features='')
            for index, row in reviews.iterrows():
                # print(f'Start create BERT embedding to review ID: {row.review_id}, num of chars: {len(row.review)}, '
                #       f'review: {row.review}')
                reviews.at[index, 'review_features'] = bert_model.get_text_average_pooler_split_bert(row.review,
                                                                                                     max_size=450)

            # reviews.to_excel(os.path.join(data_directory, 'bert_embedding_test_data.xlsx'))
            joblib.dump(reviews, (os.path.join(data_directory,
                                               f'bert_embedding_for_feature_{model}_{data_type}_data.pkl')))

    # print(type(reviews))
    # joblib.dump(reviews, 'labeledTrainData_bert_embedding.pkl')
    # reviews.to_csv('labeledTrainData_bert_embedding.csv')


if __name__ == '__main__':
    main()
