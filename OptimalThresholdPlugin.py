import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_curve, auc, average_precision_score
from sklearn.metrics import matthews_corrcoef, f1_score, balanced_accuracy_score, auc, precision_score, recall_score
import pickle

def find_threshold_one_fold(df, score_name, label_name):
    precision, recall, thresholds = precision_recall_curve(df[label_name], df[score_name])
    max_matthews = 0
    optimal_threshold = 0
    labels = df[label_name]
    for thr in thresholds:
        pred_labels = df[score_name].apply(lambda x: int(x>thr))
        matthews = matthews_corrcoef(labels, pred_labels)
        if matthews>max_matthews:
            max_matthews = matthews
            optimal_threshold = thr
    return optimal_threshold, max_matthews

def find_optimal_threshold(df, score_name, label_name, reverse_sign=True):
    """
    df - dataframe with binding scores
    score_name - column name with binding scores
    label name - column name with true labels
    if reverse_sign, than we assume that lower values represent the better score
    ----
    return: optimal threshold
    """
    df = df.copy()
    if reverse_sign:
        df[score_name] = - df[score_name]

    all_thresholds = []
    all_matthews = []
    shuffled = df.sample(frac=1, random_state=15)
    all_chunks = np.array_split(shuffled, 10)
    labels = df[label_name]
    for cv_df in all_chunks:
        cv_optimal, cv_matthews = find_threshold_one_fold(cv_df, score_name, label_name)
        all_thresholds.append(cv_optimal)
        all_matthews.append(cv_matthews)

    optimal_threshold = np.mean(all_thresholds)
    pred_labels = df[score_name].apply(lambda x: int(x>optimal_threshold))
    balanced_accuracy = balanced_accuracy_score(labels, pred_labels)
    f1 = f1_score(labels, pred_labels)
    precision = precision_score(labels, pred_labels)
    recall = recall_score(labels, pred_labels)

    if reverse_sign:
        all_thresholds = [-x for x in all_thresholds]
        optimal_threshold = np.mean(all_thresholds)

    print(f"Optimal threshold: {optimal_threshold}; BA={balanced_accuracy}, F1={f1}, Precision={precision}; Recall={recall}")
    return all_thresholds, all_matthews, [balanced_accuracy, f1, precision, recall]

import PyPluMA
import PyIO

class OptimalThresholdPlugin:
    def input(self, inputfile):
       self.parameters = PyIO.readParameters(inputfile)
    def run(self):
        pass
    def output(self, outputfile):
        SCORES_MASIF=PyPluMA.prefix()+"/"+self.parameters["csvfile"]#"data/masif_test/MaSIF-Search_scores.csv"
        other_labels = PyIO.readSequential(PyPluMA.prefix()+"/"+self.parameters["other_labels"])
        if (SCORES_MASIF.endswith('csv')):
           scores_masif = pd.read_csv(SCORES_MASIF)
        else:
            oscore = open(SCORES_MASIF, "rb")
            scores_masif = pickle.load(oscore)
            for mylabel in other_labels:
                scores_masif[mylabel] = scores_masif[mylabel].astype(float)

        all_thresholds = []
        for i in range(len(other_labels)):
         thresholds, matthews, metrics = find_optimal_threshold(scores_masif, other_labels[i], 'label')
         all_thresholds.append([self.parameters["label"]]+thresholds)

        thresholds_df = pd.DataFrame(all_thresholds, columns=['Method']+list(range(1,11)))
        thresholds_df.to_csv(outputfile, index=False)
        thresholds_df
