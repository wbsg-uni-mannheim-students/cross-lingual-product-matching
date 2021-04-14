import ast
import datetime
import os
import pathlib
import re
from pathlib import Path
from typing import List

import nltk
import numpy as np
import pandas as pd
import py_entitymatching as em
import torch
import transformers
from nltk.corpus import stopwords
from sklearn.metrics import f1_score
from torch import FloatTensor
from torch.nn import CrossEntropyLoss
from transformers import Trainer
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

nltk.download("stopwords")



class CustomDataset(torch.utils.data.Dataset):
    """
    Custom implementation of abstract torch.utils.data.Dataset class
    """

    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


class CustomTrainer(Trainer):
    """
    Subclass of Trainer-Class to define custom loss function using class weights for unbalanced data.
    """

    def compute_loss(self, model, inputs, return_outputs=False):
        # copy inputs, because, weirdly, they are otherwise mutated outside the function
        inputs_backup = inputs.copy()
        labels = inputs_backup.pop("labels")
        outputs = model(**inputs_backup)
        logits = outputs[0]

        # Convert labels tensor from float to long as required by CrossEntropyLoss
        labels_long = labels.type(torch.LongTensor).cuda() if torch.cuda.is_available() \
            else labels.type(torch.LongTensor)

        # Check if loss function has already been defined, otherwise define it
        try:
            # Compute loss
            loss = self.loss_function(logits, labels_long)
        except AttributeError:
            y = self.train_dataset.labels
            # Compute class weights inversely proportional to size so as to render weights for larger classes small
            class_weights = FloatTensor(len(y) / (len(set(y)) * np.bincount(y)))
            self.loss_function = CrossEntropyLoss(weight=class_weights).cuda() if torch.cuda.is_available() \
                else CrossEntropyLoss(weight=class_weights)
            loss = self.loss_function(logits, labels_long)

        return (loss, outputs) if return_outputs else loss

    def _sorted_checkpoints(self, checkpoint_prefix=PREFIX_CHECKPOINT_DIR, use_mtime=False) -> List[str]:
        ordering_and_checkpoint_path = []

        glob_checkpoints = [str(x) for x in Path(self.args.output_dir).glob(f"**/{checkpoint_prefix}-*")]

        for path in glob_checkpoints:
            if use_mtime:
                ordering_and_checkpoint_path.append((os.path.getmtime(path), path))
            else:
                regex_match = re.match(f".*{checkpoint_prefix}-([0-9]+)", path)
                if regex_match and regex_match.groups():
                    ordering_and_checkpoint_path.append((int(regex_match.groups()[0]), path))

        checkpoints_sorted = sorted(ordering_and_checkpoint_path)
        checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]

        # Make sure we don't delete the best model.
        if self.state.best_model_checkpoint is not None:
            best_model_index = checkpoints_sorted.index(str(Path(self.state.best_model_checkpoint)))
            checkpoints_sorted[best_model_index], checkpoints_sorted[-1] = (
                checkpoints_sorted[-1],
                checkpoints_sorted[best_model_index],
            )
        return checkpoints_sorted


def compute_metrics(pred):
    """
    Computes macro f1

    :param pred: Prediction object of transformer
    :type pred: transformers.trainer_utils.PredictionOutput
    :return: Dictionary containing the macro f1
    :rtype: dict
    """
    # distinguish between transformer results and baseline results
    if isinstance(pred, (transformers.trainer_utils.PredictionOutput, transformers.trainer_utils.EvalPrediction)):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
    else:
        labels = pred["labels"]
        preds = pred["predictions"]

    f1 = f1_score(labels, preds, average='macro')
    return {
        'f1': f1,
    }


def prep_data_pair(
        train_data,
        test_data,
        use_description):
    """
    Runs simple preprocessing and encoding of test and training data
    :param train_data:
    :param test_data:
    :return: training_set, test_set
    """

    train_data["content_1"] = train_data["title_1"]
    train_data["content_2"] = train_data["title_2"]
    test_data["content_1"] = test_data["title_1"]
    test_data["content_2"] = test_data["title_2"]

    if use_description:
        train_data.loc[train_data.description_1.notna(), "content_1"] = train_data.loc[
                                                                            train_data.description_1.notna(), "title_1"] + ' ' + \
                                                                        train_data.loc[
                                                                            train_data.description_1.notna(), "description_1"]
        train_data.loc[train_data.description_2.notna(), "content_2"] = train_data.loc[
                                                                            train_data.description_2.notna(), "title_2"] + ' ' + \
                                                                        train_data.loc[
                                                                            train_data.description_2.notna(), "description_2"]
        test_data.loc[test_data.description_1.notna(), "content_1"] = test_data.loc[
                                                                          test_data.description_1.notna(), "title_1"] + ' ' + \
                                                                      test_data.loc[
                                                                          test_data.description_1.notna(), "description_1"]
        test_data.loc[test_data.description_2.notna(), "content_2"] = test_data.loc[
                                                                          test_data.description_2.notna(), "title_2"] + ' ' + \
                                                                      test_data.loc[
                                                                          test_data.description_2.notna(), "description_2"]

    # remove MPN and EAN from titles
    train_data["content_1"] = train_data.apply(
        lambda row: remove_identifier_from_content_pair(row, "category_1", "content_1", "ean_1"), axis=1)
    train_data["content_2"] = train_data.apply(
        lambda row: remove_identifier_from_content_pair(row, "category_2", "content_2", "ean_2"), axis=1)

    test_data["content_1"] = test_data.apply(
        lambda row: remove_identifier_from_content_pair(row, "category_1", "content_1", "ean_1"), axis=1)
    test_data["content_2"] = test_data.apply(
        lambda row: remove_identifier_from_content_pair(row, "category_2", "content_2", "ean_2"), axis=1)

    return train_data, test_data


def prep_data_pair_mallegan(dataset, use_description):
    """
    Preprocessing for MAGELLAN framework
    :param dataset: train or test dataset
    :return: combined MAGELLAN-ready dataset
    """

    # Preprocess pairs (A is left side of the pair, B is right side)
    A = dataset[['content_1']]
    # A = A.reset_index()
    A = A.rename(columns={'content_1': "title"})
    A["l_id"] = A.index
    em.set_key(A, 'l_id')

    B = dataset[['content_2']]
    # B = B.reset_index()
    B = B.rename(columns={'content_2': "title"})
    B["r_id"] = len(A) + B.index
    em.set_key(B, 'r_id')

    # join A and B to retrieve complete pairwise dataset
    G = A.join(B, lsuffix='_left', rsuffix='_right')
    G["label"] = dataset[["label"]]
    G["id"] = G.index
    em.set_key(G, 'id')
    em.set_ltable(G, A)
    em.set_rtable(G, B)
    em.set_fk_ltable(G, 'l_id')
    em.set_fk_rtable(G, 'r_id')

    return A, B, G


def prep_data_multi(train_data,
                    test_data,
                    use_description):
    """
    Runs simple preprocessing and encoding of test and training data
    :param train_data:
    :param test_data:
    :return: training_set, test_set
    """

    train_data["content"] = train_data["title"]
    test_data["content"] = test_data["title"]

    if use_description:
        train_data.loc[train_data.description.notna(), "content"] = train_data.loc[
                                                                        train_data.description.notna(), "title"] + ' ' + \
                                                                    train_data.loc[
                                                                        train_data.description.notna(), "description"]
        test_data.loc[test_data.description.notna(), "content"] = test_data.loc[
                                                                      test_data.description.notna(), "title"] + ' ' + \
                                                                  test_data.loc[
                                                                      test_data.description.notna(), "description"]

    # remove MPN and EAN from content
    train_data.loc[:, "content"] = train_data.apply(lambda row: remove_identifier_from_content_multi(row), axis=1)
    test_data.loc[:, "content"] = test_data.apply(lambda row: remove_identifier_from_content_multi(row), axis=1)

    # assign new, consecutive labels
    train_data_final, test_data_final, label_dict_inv = assign_consecutive_labels(train_data, test_data)

    return train_data_final, test_data_final, label_dict_inv


def remove_identifier_from_content_multi(row):
    """
    For offers, that contain MPN or EAN in their title/description, remove it
    :return:
    """

    # remove MPN and EAN (use different logic for toy and phone)
    row.content = re.sub(re.escape(str(row.ean)), '', row.content)
    if row.category == 'toy':
        row.content = re.sub(r"\d{5}|\d{4}", '', row.content)
    elif row.category == 'phone':
        row.content = re.sub(r"\bM[A-Z]{1,3}\d*[A-Z]*\d{1,2}[A-Z]{1,2}[A-Z]|SM.{0,1}[A-Z][\d]{3}[A-Z]{1,2}|GA\d{5}",
                             '',
                             row.content)

    return row.content


def remove_identifier_from_content_pair(row, category, content, ean):
    """
    For offers, that contain MPN or EAN in their title/description, remove it
    :return:
    """

    # remove MPN and EAN (use different logic for toy and phone)
    row[content] = re.sub(re.escape(str(row[ean])), '', row[content])

    if row[category] == 'toy':
        row[content] = re.sub(r"\d{5}|\d{4}", '', row[content])
    elif row[category] == 'phone':
        row[content] = re.sub(r"\bM[A-Z]{1,3}\d*[A-Z]*\d{1,2}[A-Z]{1,2}[A-Z]|SM.{0,1}[A-Z][\d]{3}[A-Z]{1,2}|GA\d{5}",
                              '',
                              row[content])
    return row[content]


def remove_stopwords(df, pairwise=False):
    """
    Remove domain-unspecific stopwords from description
    :param df:
    :param pairwise: Boolean to check for pairwise case
    :return:
    """
    lang_dict = {
        'en': 'english',
        'de': 'german',
        'es': 'spanish',
        'fr': 'french'
    }

    if pairwise:
        for i in [1, 2]:
            for lang in df[f"lang_{i}"].unique():
                # Check for translated data (use stopwords for target language)
                if '-' in lang:
                    lang = lang.split('-')[1]
                stop = stopwords.words(lang_dict[lang])
                df.loc[(df[f"lang_{i}"] == lang) & (df[f'description_{i}'].notna()), f'description_{i}'] = df[
                    (df[f"lang_{i}"] == lang) & (df[f'description_{i}'].notna())][f'description_{i}'].apply(
                    lambda x: ' '.join([word for word in x.split() if word.lower() not in stop]))

    else:
        # Multiclass case
        for lang in df.lang.unique():
            # Check for translated data (use stopwords for target language)
            if '-' in lang:
                lang = lang.split('-')[1]
            stop = stopwords.words(lang_dict[lang])
            df.loc[(df.lang == lang) & (df['description'].notna()), 'description'] = df[
                (df.lang == lang) & (df['description'].notna())]['description'].apply(
                lambda x: ' '.join([word for word in x.split() if word.lower() not in stop]))

    return df


def assign_consecutive_labels(train_data, test_data):
    """
    PyTorch requires the targets labels to be consecutively labels. Therefore, this function backups the old
    label and assigns new ones consecutively.
    :param train_data:
    :param test_data:
    :return:
    """

    # backup label id
    train_data.loc[:, 'old_label_id'] = train_data['label']
    test_data.loc[:, 'old_label_id'] = test_data['label']

    label_dict = dict()
    for i, label in enumerate(train_data["label"].unique()):
        label_dict[label] = i

    train_data.label.replace(label_dict, inplace=True)
    test_data.label.replace(label_dict, inplace=True)

    # Get the inverse of the label dict, to revert the mapping later on
    label_dict_inv = {v: k for k, v in label_dict.items()}

    return train_data, test_data, label_dict_inv


def output_and_store_results(settings_data, setting_key, category, train_langs_str, lang, result, all_scores,
                             hyperparameters, input_path, predictions):
    """
    Print results in console and store them in csv (merging with previous results)
    :param settings_data: settings used in current run
    :param setting_key: settings key used in current run
    :param category: Category used in the current run (phone or toy)
    :param train_langs_str: languages (train) used in the current run
    :param lang: languages (test) used in the current run
    :param result: averaged result of the current run (f1)
    :param all_scores: all results of the current runs (f1)
    :param hyperparameters: tuned hyperparameters retrieved from model
    :param input_path: base path to determine results.csv path
    :param predictions: exact predictons of the model
    :return:
    """
    timestamp = datetime.datetime.now().strftime("%d-%m-%Y %H-%M-%S")
    if len(pd.DataFrame(predictions)) == 1:
        pred_df = pd.DataFrame(predictions).transpose()
    else:
        pred_df = pd.DataFrame(predictions)
    pred_df.to_csv(
        f'{str(pathlib.Path(input_path).parent) + f"/results/{setting_key}_{category}_{lang}_{timestamp}.csv"}')

    if "model_parameters" in settings_data:
        model_parameters = str(settings_data["model_parameters"])
        settings_data.pop("model_parameters", None)
    else:
        model_parameters = ""
    results_df = pd.DataFrame.from_dict(settings_data, orient='index').transpose()

    results_df["experiment_id"] = setting_key
    results_df["category"] = category
    results_df["train_lang"] = train_langs_str
    results_df["eval_lang"] = lang
    results_df["f1_averaged"] = result
    results_df["f1_all"] = str(all_scores)
    results_df["hyperparameters"] = hyperparameters
    results_df["model_parameters"] = model_parameters
    results_df["timestamp"] = timestamp
    results_df = results_df.set_index(["experiment_id", "category", "eval_lang"])
    print('-----------------------------------------------------------')
    print('RESULTS')
    print('-----------------------------------------------------------')
    print(f"Runs: {all_scores})")
    print("Average Score:")
    print(results_df[['train_lang', 'f1_averaged']])
    print('-----------------------------------------------------------')

    if os.path.isfile(f'{str(pathlib.Path(input_path).parent) + f"/results/results.csv"}'):
        prev_results = pd.read_csv(f'{str(pathlib.Path(input_path).parent) + f"/results/results.csv"}',
                                   index_col=["experiment_id", "category", "eval_lang"])
        results_df = results_df.append(prev_results)
        # Check if there are previous runs of the same experiment in results.csv
        if len(all_scores) < 3 and len(results_df[results_df.index.duplicated()]) >= 1\
                and settings_data["model_type"] == "transformer":
            # Number of current run
            i = len(all_scores) - 1
            prev_run = results_df[results_df.index.duplicated()]
            # Parse string as list
            prev_scores = ast.literal_eval(prev_run.loc[:, "f1_all"].values[0])
            # Append score of current run
            prev_scores.append(all_scores[i])
            # Save updated scores in DataFrame
            results_df.loc[results_df.index.duplicated(keep='last'), "f1_all"] = str(prev_scores)
            results_df.loc[results_df.index.duplicated(keep='last'), "f1_averaged"] = np.mean(prev_scores)

        results_df = results_df[~results_df.index.duplicated()].sort_index()

    results_df.to_csv(f'{str(pathlib.Path(input_path).parent) + f"/results/results.csv"}')


def create_config_key(settings):
    """
    Automatically creates keys to identify the individual runs of the experiments
    :param settings:
    :return:
    """
    replacement_dict = {
        'model': {'bert-base-multilingual-uncased': 'mbert', 'bert-base-uncased': 'bert',
                  'xlm-roberta-base': 'XLM-R'},
        'use_description': {True: 'all', False: 'title'}
    }
    components = []
    for component in ['category', 'problem_type', 'model', 'dataset_size', 'vectorization', 'use_description',
                      'train_lang']:
        if component in replacement_dict.keys() and settings.get(component) in replacement_dict[component].keys():
            components.append(replacement_dict[component][settings.get(component)])
        elif component == 'train_lang':
            components.append('+'.join(settings.get(component)))
        elif type(settings.get(component)) == list:
            components.append(''.join(settings.get(component)))
        else:
            components.append(str(settings.get(component)))

    return '_'.join(components).replace('None', '')
