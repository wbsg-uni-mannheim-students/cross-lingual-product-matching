from typing import List
import random
import pandas as pd
import numpy as np
import json
import pathlib
import torch

from util import CustomDataset, CustomTrainer, compute_metrics, prep_data_multi, output_and_store_results, create_config_key
from argparse import ArgumentParser

import transformers
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
from transformers import Trainer, TrainingArguments
from transformers.trainer_callback import EarlyStoppingCallback
from sklearn.model_selection import train_test_split

# Define string constants
SMALL = "small"
LARGE = "large"

# Set seed for reproducibility
#set_seed(2021)

def run_transformer_multi_new(input_path: str, setting_keys: List[str] = None):
    """

    :param input_path: Path to settings.json (must be placed in the same directory as the "datasets"-folder)
    :param setting_keys: Selected setting keys if only some of the model configurations in settings.json are to be used
    :return:
    """
    # Read settings file
    with open(f'{input_path}') as file:
        settings = json.load(file)

    for setting_key, setting_data in settings.items():
        # Only run the setting if the key is in the list of settings or no setting_keys are provided
        if setting_keys is None:
            pass
        elif setting_keys is not None and setting_key not in setting_keys:
            continue

        # Get name of settings
        settings_name = create_config_key(setting_data)
        # Get the relevant data from the settings
        # Set model global to use it inside model_init function
        global model
        model = setting_data.get("model")
        n_runs = setting_data.get("n_runs")
        use_description = setting_data.get("use_description")
        run_parameter_search = setting_data.get("hyperparameter_search")
        train_langs = setting_data.get("train_lang")
        test_langs = setting_data.get("eval_lang")
        category = setting_data.get("category")
        params = setting_data.get("model_parameters")
        resume_training = setting_data.get("resume_training")
        n_hp_runs = setting_data.get("n_hp_runs")

        # Process the categories separately
        dataset_p = pathlib.Path(input_path).parent.joinpath("datasets")

        train_data_p = dataset_p.joinpath(f'multi_class_train_set_{category}.csv')
        test_data_p = dataset_p.joinpath(f'multi_class_test_set_{category}.csv')

        # Read the data
        train_data = pd.read_csv(train_data_p)
        test_data = pd.read_csv(test_data_p)

        # Filter the train data:
        train_data = train_data.loc[train_data["lang"].isin(train_langs)]
        # Prepare the train and test data for the experiments and get the mapping of the labels
        train_data, test_data, label_dict_inv = prep_data_multi(train_data, test_data, use_description)

        # Tokenize the text features
        # Instantiate Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model)

        # Encode the text features for training (test data is encoded later)
        train_encodings = tokenizer(train_data.content.tolist(), truncation=True, padding=True)

        # Create Trainset
        train_set = CustomDataset(train_encodings, train_data.label.tolist())

        # Load transformer config
        # Set model global to use it inside model_init function
        global model_config
        model_config = AutoConfig.from_pretrained(model, num_labels=train_data["label"].nunique())

        # Set compute warmup steps
        params['warmup_steps'] = np.ceil(len(train_data) / (params['per_device_train_batch_size'] * params['gradient_accumulation_steps']))

        # Run hyperparameter tuning, if specified by settings.json
        if run_parameter_search:
            best_lr = 0
            best_f1 = 0
            lr_file = pathlib.Path(input_path).parent.joinpath("results", "lr.txt")

            # In case we don´t do the complete hyperparameter tuning, we need to read the best lr from file
            if n_hp_runs < 5:
                f = open(lr_file, "r")
                content = f.readline().split(",")
                best_f1, best_lr = float(content[0]), float(content[1])

            # Set learning rates
            lrs = np.array([5e-6, 1e-5, 5e-5, 1e-4, 3e-5])
            lrs = lrs[-n_hp_runs:]

            # Prepare train and valdiation set for hyperparameter tuning
            tune_data, val_data, tune_label, val_label = train_test_split(
                train_data, train_data.label,
                test_size=0.2, stratify=train_data.label,
                random_state=42
            )

            tune_encodings = tokenizer(tune_data.content.tolist(), truncation=True, padding=True)

            val_encodings = tokenizer(val_data.content.tolist(), truncation=True, padding=True)

            tune_set = CustomDataset(tune_encodings, tune_data.label.tolist())
            val_set = CustomDataset(val_encodings, val_data.label.tolist())

            # We stop early, if we do not improve on the validation set
            early_stopping = EarlyStoppingCallback(early_stopping_patience=3)

            # Set model parameters for tuning with TrainingArguments-object
            training_args = TrainingArguments(
                output_dir=f'./model',
                overwrite_output_dir=params.get('overwrite_output_dir'),
                num_train_epochs=params.get('num_train_epochs'),
                save_total_limit=params.get('save_total_limit'),
                per_device_train_batch_size=params.get('per_device_train_batch_size'),
                per_device_eval_batch_size=params.get('per_device_eval_batch_size'),
                gradient_accumulation_steps=params.get('gradient_accumulation_steps'),
                warmup_steps=params.get('warmup_steps'),
                weight_decay=params.get('weight_decay'),
                evaluation_strategy=params.get('evaluation_strategy'),
                load_best_model_at_end=params.get('load_best_model_at_end'),
                metric_for_best_model=params.get('metric_for_best_model'),
            )

            trainer = CustomTrainer(
                model_init=model_init,
                train_dataset=tune_set,
                eval_dataset=val_set,
                compute_metrics=compute_metrics,
                callbacks=[early_stopping]
            )

            for lr in lrs:
                print(f'Learning rate set to: {lr}')
                # Set current learning rate
                setattr(training_args, 'learning_rate', lr)
                # Set training args
                setattr(trainer, 'args', training_args)

                # Train and get the current f1
                trainer.train()
                eval_score = trainer.evaluate()
                print(f'Eval scores: {eval_score}')

                if eval_score["eval_f1"] > best_f1:
                    best_f1 = eval_score["eval_f1"]
                    best_lr = lr
                    # Write to disk
                    f = open(lr_file, "w")
                    f.write(','.join([str(best_f1), str(best_lr)]))
                    f.close()

        # Create dict to save scores for every run
        scores_per_lang = dict((lang, list()) for lang in test_langs)
        avg_scores_per_lang = dict()
        results_per_lang = dict((lang, list()) for lang in test_langs)

        # Run every Experiment n-times
        for i in range(n_runs):
            # Change args and reinstantiate trainer for training on whole trainset (no early stopping here)
            # Set new seed for different results in each run

            training_args = TrainingArguments(
                output_dir=f'./model',
                overwrite_output_dir=params.get('overwrite_output_dir'),
                num_train_epochs=params.get('num_train_epochs'),
                learning_rate=params.get('learning_rate'),
                save_total_limit=params.get('save_total_limit'),
                per_device_train_batch_size=params.get('per_device_train_batch_size'),
                per_device_eval_batch_size=params.get('per_device_eval_batch_size'),
                gradient_accumulation_steps=params.get('gradient_accumulation_steps'),
                warmup_steps=params.get('warmup_steps'),
                weight_decay=params.get('weight_decay'),
                seed=random.randint(0, 2021)
            )

            trainer = CustomTrainer(
                model_init=model_init,
                args=training_args,
                train_dataset=train_set,
                compute_metrics=compute_metrics
            )

            # Use best parameters for new trainer, if parameter search was run
            if run_parameter_search:
                setattr(trainer.args, 'learning_rate', best_lr)
                print(f'Learning rate set to: {best_lr}')
            else:
                print(f'Learning rate set to: {params.get("learning_rate")}')

            # Train the model
            trainer.train()

            # Run predictions
            for lang in test_langs:
                # Subset the test data
                test_data_lang = test_data.loc[test_data['lang'] == lang]

                # Encode Text Features for testing
                test_encodings_lang = tokenizer(test_data_lang.content.tolist(), truncation=True, padding=True)

                # Create Test Set
                test_set_lang = CustomDataset(test_encodings_lang, test_data_lang.label.tolist())

                # Predict and compute metrics to measure performance of model
                pred = trainer.predict(test_set_lang)
                # Map the predictions back to cluster ids
                pred_cl_id = np.array([label_dict_inv[x] for x in pred[0].argmax(-1)])
                scores_per_lang[lang].append(pred[2]['eval_f1'])
                results_per_lang[lang].append(pred_cl_id)

                # Output results
                all_scores = scores_per_lang[lang]
                avg_scores_per_lang[lang] = np.mean(scores_per_lang[lang])
                output_and_store_results(setting_data, settings_name, category, str(train_langs), lang,
                                         avg_scores_per_lang[lang], all_scores,
                                         str({"learning_rate": trainer.args.learning_rate}),
                                         input_path, results_per_lang[lang])

def tune_hyperparameters(trainer, tokenizer, train_data):
    """
    Runs hyperparameter tuning and returns the best parameter configuration found.
    :param trainer:
    :param tokenizer:
    :param train_data:
    :return:
    """
    tune_data, val_data, tune_label, val_label = train_test_split(
        train_data, train_data.label,
        test_size=0.2, stratify=train_data.label,
        random_state=42
    )

    tune_encodings = tokenizer(tune_data.content.tolist(), truncation=True, padding=True)

    val_encodings = tokenizer(val_data.content.tolist(), truncation=True, padding=True)

    tune_set = CustomDataset(tune_encodings, tune_data.label.tolist())
    val_set = CustomDataset(val_encodings, val_data.label.tolist())

    # Hand tuning and validation set to trainer
    setattr(trainer, 'train_dataset', tune_set)
    setattr(trainer, 'eval_dataset', val_set)

    # Search Parameters
    best_run = trainer.hyperparameter_search(
        hp_space=hp_space,
        n_trials=5,
        direction="maximize",
        compute_objective=lambda metrics: metrics['eval_f1']
    )
    return best_run


def model_init():
    """
    Function to reinitialize the model during hyperparameter tuning.
    :return:
    """
    return AutoModelForSequenceClassification.from_pretrained(model, config=model_config)


def hp_space(trial):
    """
    Function to define the hyperparameter space searched during tuning.
    :param trial:
    :return:
    """
    return {
        # Only tune learning rate for now
        "learning_rate": trial.suggest_float("learning_rate", 5e-6, 1e-4, log=True),
    }

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-i", "--input", type=str,
                        help="path to project", metavar="path")
    args = parser.parse_args()
    input_path = args.input
    run_transformer_multi_new(input_path)