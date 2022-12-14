import numpy as np
import math
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import mean_squared_error, log_loss
from utils import * 

def manual_logistic_regression(inputs):

    def sigmoid(wsi):
        prob = [1.0/(1 + math.exp(-w)) for w in wsi]
        predicted = []
        for p in prob:
            if p >= 0.5:
                predicted.append(1)
            else:
                predicted.append(0)
        return prob, predicted

    weights = inputs[OUTOUT_DICT_WEIGHT_KEY]
    b = inputs[OUTPUT_DICT_B_KEY] ########################
    test_size = inputs[OUTPUT_DICT_TEST_SIZE_KEY]
    x = inputs[OUTPUT_DICT_DATA_POINT_KEY]
    y = inputs[OUTPUT_DICT_LABELS_KEY]
    y = y.reshape(-1)

    if not (0 < test_size < 1):
        raise Exception("Input test size must be in range 0-1.")
    
    weighted_sum_input = np.add(np.matmul(x, weights), b)
    pred_prob, predicted = sigmoid(weighted_sum_input)
    cost = log_loss(y, predicted)

    return {MODEL_OUTPUT_DICT_COST_KEY: cost,
            MODEL_OUTPUT_DICT_RESULTS_KEY: predicted,
            MODEL_OUTPUT_DICT_PROB_KEY: pred_prob, ##########################
            MODEL_OUTPUT_DICT_FITTED_WEIGHT_KEY: weights
            }

