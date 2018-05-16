import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
import seaborn as sns
import collections

labels = {"Clap": 0, "Cry": 1, "Disappoint": 2, "Explode": 3, "FacePalm": 4,
    "Hands": 5, "Neutral": 6, "Shrug": 7, "Think": 8, "Upside": 9}

label_ids = ["Clap", "Cry", "Disappoint", "Explode", "FacePalm", 
    "Hands", "Neutral", "Shrug", "Think", "Upside"]

def load_train_dataset(path):
    data = {}
    data["sentence"] = []
    data["emoji"] = []
    with open(path) as f:
        for line in f:
            splited = line.split("\t", 2)
            data["sentence"].append(splited[2])
            data["emoji"].append(labels[splited[1]])
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

def main():
    # reduce logging output
    tf.logging.set_verbosity(tf.logging.ERROR)

    # load train data
    train_df = load_train_dataset("data/train_raw.txt")
    train_df.head()

    # training input on the whole training set with no limit on training epochs
    train_input_fn = tf.estimator.inputs.pandas_input_fn(
        train_df, train_df["emoji"], num_epochs=None, shuffle=True)

    # feature column
    embedded_text_feature_column = hub.text_embedding_column(
        key="sentence", 
        module_spec="https://tfhub.dev/google/nnlm-en-dim128/1")

    # dnnclassifier
    estimator = tf.estimator.DNNClassifier(
        hidden_units=[500, 100],
        feature_columns=[embedded_text_feature_column],
        n_classes=10,
        optimizer=tf.train.AdagradOptimizer(learning_rate=0.003))

    # train
    estimator.train(input_fn=train_input_fn, steps=1000)

    # predict
    test_df = load_test_dataset("data/test_raw.txt")
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
    with open("test_raw_output.txt", "w") as f:
        for key in test_dict_keys:
            f.write("{}\t{}\t{}".format(key, 
                label_ids[test_dict[key][1]], test_dict[key][0]))
    
    # evaluate
    predict_train_input_fn = tf.estimator.inputs.pandas_input_fn(
        train_df, train_df["emoji"], shuffle=False)
    train_eval_result = estimator.evaluate(input_fn=predict_train_input_fn)
    print("Training set accuracy: {accuracy}".format(**train_eval_result))


if __name__ == "__main__":
    main()
