from os import SCHED_OTHER
from numpy.lib.function_base import percentile
import pandas as pd
import numpy as np
import math
# calculate mean, and variance
# "fixed acidity";
# "volatile acidity";
# "citric acid";
# "residual sugar";
# "chlorides";
# "free sulfur dioxide";
# "total sulfur dioxide";
# "density";
# "pH";
# "sulphates";
# "alcohol";
# "quality"
winequality = pd.read_csv("winequality-white.csv", sep=';', encoding='utf-8-sig')
# winequality.columns = winequality.columns.str.strip()
def divide_in_sets(main_set, k):
    winequality_sets = []
    count = int(len(main_set) / k)
    while k > 0:
        winequality_set = main_set.iloc[(k-1)*count:k*count]
        # winequality_set = winequality_set.reset_index(drop=True)
        winequality_sets.append(winequality_set)
        k -= 1
    winequality_sets.reverse()
    return winequality_sets

def create_sets_60_40(sets):
    list_of_train_sets = []
    list_of_test_sets = []
    for set1 in sets:
        concat_list_train = []
        concat_list_test = []
        concat_list_train.append(set1)
        for set2 in sets:
            if set2.equals(set1):
                continue
            concat_list_train.append(set2)
            for set3 in sets:
                if set3.equals(set1) or set3.equals(set2):
                    continue
                if len(concat_list_train) != 3:
                    concat_list_train.append(set3)
                possible_train_set = pd.concat(concat_list_train)
                if possible_train_set['quality'].nunique() == 7:
                    for set in sets:
                        if set.equals(set1) or set.equals(set2) or set.equals(set3):
                            continue
                        concat_list_test.append(set)
                    possible_test_set = pd.concat(concat_list_test)
                    if possible_test_set['quality'].nunique() == 7:
                        list_of_train_sets.append(possible_train_set)
                        list_of_test_sets.append(possible_test_set)
                        concat_list_test = []
                        concat_list_train.pop()
                    else:
                        concat_list_test = []
                        concat_list_train.pop()
        concat_list_train.pop()
    return list_of_train_sets, list_of_test_sets
    

def get_mean_var_from_sets(sets):
    means_and_vars = []
    for set in sets:
        mean = np.array(set.groupby(['quality']).mean())
        var = np.array(set.groupby(['quality']).var())
        slot = mean, var
        means_and_vars.append(slot)
    return means_and_vars

def GaussianNaiveBayes(value, mean, var):
    result = 1/(np.sqrt(2 * np.pi * var)) * np.exp(-((value - mean)**2)/(2*var))
    return result

winequality_sets = divide_in_sets(winequality, 5)
sets = create_sets_60_40(winequality_sets)
train_sets = sets[0]
test_sets = sets[1]
def choose_best_model(train_sets, test_sets):
    means_and_vars = get_mean_var_from_sets(train_sets)
    i = 0
    best_score = 0
    for test_set in test_sets:
        score = 0
        quality_results = [0,0,0,0,0,0,0]
        prediction_results = [0,0,0,0,0,0,0]
        wrong_results = [0,0,0,0,0,0,0]
        count_all = test_set.shape[0]
        for index, row in test_set.iterrows():
            expected_quality = row[11]
            probs = []
            j = 0
            while j < 7:
                count_q = test_set[test_set['quality'] == j + 3].shape[0]
                p_quality = count_q/count_all
                n = 0
                for value in row:
                    if n < 11:
                        p_quality = p_quality * GaussianNaiveBayes(value, means_and_vars[i][0][j][n], means_and_vars[i][1][j][n])
                        ### i - set ; j - quality ; n - category ; 0 - mean ; 1- var
                        n += 1
                    elif n == 11:
                        probs.append(p_quality)
                j += 1
            prediction_quality = probs.index(max(probs)) + 3
            prediction_results[int(prediction_quality) - 3] += 1
            if prediction_quality == expected_quality:
                score += 1
                quality_results[int(expected_quality) - 3] += 1
            else:
                wrong_results[int(prediction_quality) - 3] += 1
        if score > best_score:
            best_score = score
            best_set = i
            best_quality_results = quality_results
            best_prediction_results = prediction_results
            best_wrong_results = wrong_results
        i += 1
    return best_score, best_set, best_quality_results, best_prediction_results, best_wrong_results

