import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
import seaborn as sns
import collections

N_CLASSES = 10
LABELS_DICT = {"Clap": 0, "Cry": 1, "Disappoint": 2, "Explode": 3, "FacePalm": 4,
    "Hands": 5, "Neutral": 6, "Shrug": 7, "Think": 8, "Upside": 9}
LABELS = ["Clap", "Cry", "Disappoint", "Explode", "FacePalm", 
    "Hands", "Neutral", "Shrug", "Think", "Upside"]
TRAIN_DATASET_PATH = "data/train_raw.txt"
DEV_DATASET_PATH = "data/dev_raw.txt"
TEST_DATASET_PATH = "data/test_raw.txt"
OUTPUT_FILENAME = "test_raw_output.txt"

def load_dataset(path):
    data = {}
    data["sentence"] = []
    data["emoji"] = []
    with open(path) as f:
        for line in f:
            splited = line.split("\t", 2)
            data["sentence"].append(splited[2])
            data["emoji"].append(LABELS_DICT[splited[1]])
    df = pd.DataFrame.from_dict(data)
    return df.sample(frac=1).reset_index(drop=True)

def load_test_dataset(path):
    data = {}
    data["id"] = []
    data["sentence"] = []
    with open(path) as f:
        for line in f:
            splited = line.split("\t", 2)
            data["id"].append(splited[0])
            data["sentence"].append(splited[2])
    df = pd.DataFrame.from_dict(data)
    return df.sample(frac=1).reset_index(drop=True)

def predict_and_output(estimator):
    # predict
    test_df = load_test_dataset(TEST_DATASET_PATH)
    test_input_fn = tf.estimator.inputs.pandas_input_fn(test_df, shuffle=False)
    predictions = estimator.predict(input_fn=test_input_fn)
    emoji_ids = []
    for pred_dict in predictions:
        class_id = pred_dict['class_ids'][0]
        emoji_ids.append(class_id)
    test_df["emoji"] = emoji_ids

    # output
    test_dict = test_df.set_index('id').T.to_dict('list')
    test_dict_keys = list(test_dict.keys())
    test_dict_keys.sort(key=int)
    with open(OUTPUT_FILENAME, "w") as f:
        for key in test_dict_keys:
            f.write("{}\t{}\t{}".format(key, 
                LABELS[test_dict[key][1]], test_dict[key][0]))

def get_predictions(estimator, input_fn):
    return [x["class_ids"][0] for x in estimator.predict(input_fn=input_fn)]

def train_and_evaluate_with_module(hub_module, 
    train_input_fn, predict_train_input_fn, predict_dev_input_fn, 
    train_module=False):
    embedded_text_feature_column = hub.text_embedding_column(
        key="sentence", module_spec=hub_module, trainable=train_module)

    estimator = tf.estimator.DNNClassifier(
        hidden_units=[500, 100],
        feature_columns=[embedded_text_feature_column],
        n_classes=N_CLASSES,
        optimizer=tf.train.AdagradOptimizer(learning_rate=0.003))

    estimator.train(input_fn=train_input_fn, steps=1000)

    train_eval_result = estimator.evaluate(input_fn=predict_train_input_fn)
    dev_eval_result = estimator.evaluate(input_fn=predict_dev_input_fn)
    training_set_accuracy = train_eval_result["accuracy"]
    dev_set_accuracy = dev_eval_result["accuracy"]

    return {
        "training accuracy": training_set_accuracy,
        "development accuracy": dev_set_accuracy
    }

def main():
    # reduce logging output
    tf.logging.set_verbosity(tf.logging.ERROR)

    # load train and dev data
    train_df = load_dataset(TRAIN_DATASET_PATH)
    dev_df = load_dataset(DEV_DATASET_PATH)
    train_df.head()

    # training input on the whole training set with no limit on training epochs
    train_input_fn = tf.estimator.inputs.pandas_input_fn(
        train_df, train_df["emoji"], num_epochs=None, shuffle=True)

    # feature column
    embedded_text_feature_column = hub.text_embedding_column(
        key="sentence", 
        module_spec="https://tfhub.dev/google/nnlm-en-dim128/1")

    # deep neural network classifier
    estimator = tf.estimator.DNNClassifier(
        hidden_units=[500, 100],
        feature_columns=[embedded_text_feature_column],
        n_classes=N_CLASSES,
        optimizer=tf.train.AdagradOptimizer(learning_rate=0.003))

    # train
    estimator.train(input_fn=train_input_fn, steps=1000)

    # predict and output
    predict_and_output(estimator)
    
    # evaluate
    predict_train_input_fn = tf.estimator.inputs.pandas_input_fn(
        train_df, train_df["emoji"], shuffle=False)
    predict_dev_input_fn = tf.estimator.inputs.pandas_input_fn(
        dev_df, dev_df["emoji"], shuffle=False)
    train_eval_result = estimator.evaluate(input_fn=predict_train_input_fn)
    dev_eval_result = estimator.evaluate(input_fn=predict_dev_input_fn)
    print("training set accuracy: {accuracy}".format(**train_eval_result))
    print("development set accuracy: {accuracy}".format(**dev_eval_result))

    # confusion matrix on dev input
    with tf.Graph().as_default():
        cm = tf.confusion_matrix(dev_df["emoji"], 
                                get_predictions(estimator, predict_dev_input_fn))
        with tf.Session() as session:
            cm_out = session.run(cm)
    cm_out = cm_out.astype(float) / cm_out.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_out, annot=True, xticklabels=LABELS, yticklabels=LABELS)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

    # transfer learning analysis
    results = {}
    results["nnlm-en-dim128"] = train_and_evaluate_with_module(
        "https://tfhub.dev/google/nnlm-en-dim128/1", 
        train_input_fn, predict_train_input_fn, predict_dev_input_fn)
    results["nnlm-en-dim128-with-module-training"] = train_and_evaluate_with_module(
        "https://tfhub.dev/google/nnlm-en-dim128/1", 
        train_input_fn, predict_train_input_fn, predict_dev_input_fn, True)
    results["random-nnlm-en-dim128"] = train_and_evaluate_with_module(
        "https://tfhub.dev/google/random-nnlm-en-dim128/1", 
        train_input_fn, predict_train_input_fn, predict_dev_input_fn)
    results["random-nnlm-en-dim128-with-module-training"] = train_and_evaluate_with_module(
        "https://tfhub.dev/google/random-nnlm-en-dim128/1", 
        train_input_fn, predict_train_input_fn, predict_dev_input_fn, True)
    print(results)
 
# main
if __name__ == "__main__":
    main()
