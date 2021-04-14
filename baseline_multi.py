from typing import List
import numpy as np
import pandas as pd
import json
import pathlib

from util import CustomDataset, compute_metrics, prep_data_multi, output_and_store_results, create_config_key
from argparse import ArgumentParser

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import LinearSVC

# Define string constants
BINARY = "binary"
TFIDF = "tf-idf"
LOGIT = "logistic-regression"
RAFO = "random-forest"
SVM = "svm"
STANDARD = "standard"
RANDOM_STATE = 42


def run_baseline_multi(input_path: str, setting_keys: List[str] = None):
    # Read settings file
    with open(f'{input_path}') as file:
        settings = json.load(file)

    for setting_key, settings_data in settings.items():
        # Only run the setting if the key is in the list of settings or no setting_keys are provided
        if setting_keys is None:
            pass
        elif setting_keys is not None and setting_key not in setting_keys:
            continue

        # Get name of settings
        settings_name = create_config_key(settings_data)
        # Get the relevant data from the settings
        model = settings_data.get("model")
        vectorization = settings_data.get("vectorization")
        use_description = settings_data.get("use_description")
        train_langs = settings_data.get("train_lang")
        test_langs = settings_data.get("eval_lang")
        category = settings_data.get("category")

        # Create a string of the train languages
        train_langs_str = ", ".join(train_langs)

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

        # Compute the feature embedding
        if vectorization == BINARY:
            vectorizer = CountVectorizer(analyzer="word",
                                         encoding='utf-8',
                                         tokenizer=None,
                                         preprocessor=None,
                                         stop_words=None,
                                         ngram_range=(1, 2),
                                         max_features=5000,
                                         binary=True)

            train_data_embeddings = vectorizer.fit_transform(train_data['content']).toarray()

        elif vectorization == TFIDF:
            vectorizer = TfidfVectorizer()

            train_data_embeddings = vectorizer.fit_transform(train_data['content']).toarray()

        else:
            # Other vectorizations are not implemented
            raise AssertionError

        # Fit the models
        if model == LOGIT:
            est = LogisticRegression()
            # Description needs more time to converge
            parameters = {
                    'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000],
                    'class_weight': ['balanced'],
                    'max_iter': [800],
                    'n_jobs': [-2]
            }

        elif model == RAFO:
            est = RandomForestClassifier()
            parameters = {
                'n_estimators': [100],
                'max_features': ['sqrt', 'log2', None],
                'max_depth': [2, 4, 7, 10],
                'min_samples_split': [2, 5, 10, 20],
                'min_samples_leaf': [1, 2, 4, 8],
                'class_weight': ['balanced_subsample'],
                'n_jobs': [-2]
            }

        elif model == SVM:
            est = LinearSVC()
            parameters = {
                'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000],
                'class_weight': ['balanced']
            }
        else:
            # Other models are not implemented
            raise AssertionError

        print(model)

        # Define grid search and fit model
        rs = RandomizedSearchCV(estimator=est, param_distributions=parameters, scoring="f1_macro", cv=5,
                                n_jobs=-2, verbose=1, n_iter=100, refit=True)

        rs.fit(train_data_embeddings, train_data["label"].astype(int))

        # Run predictions
        scores_per_lang = {}
        for lang in test_langs:
            # Subset the test data
            test_data_lang = test_data.loc[test_data['lang'] == lang]

            # Retrieve representations & word co-occurence vectors for test set
            test_data_embeddings_lang = vectorizer.transform(test_data_lang['content']).toarray()

            # prediction and computation of metrics to measure performance of model
            pred = rs.best_estimator_.predict(test_data_embeddings_lang)

            # Map the predictions back to cluster ids
            pred_cl_id = np.array([label_dict_inv[x] for x in pred])
            # Map the true labels back to cluster ids
            true_cl_id = test_data_lang["old_label_id"].to_numpy()

            scores_per_lang[lang] = compute_metrics({"labels": true_cl_id, "predictions": pred_cl_id}).get("f1")
            output_and_store_results(settings_data=settings_data, setting_key=settings_name, category=category,
                                     train_langs_str=train_langs_str, lang=lang, result=scores_per_lang[lang],
                                     all_scores="", hyperparameters=[rs.best_params_], input_path=input_path,
                                     predictions=pred_cl_id)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-i", "--input", type=str,
                        help="path to project", metavar="path")
    args = parser.parse_args()
    input_path = args.input
    run_baseline_multi(input_path)
