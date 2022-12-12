import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from utils import * 

def auto_logistic_regression(inputs):
    test_size = inputs[OUTPUT_DICT_TEST_SIZE_KEY]
    C = 1 / inputs[OUTPUT_DICT_REG_STRENGTH_KEY]
    x = inputs[OUTPUT_DICT_DATA_POINT_KEY]
    y = inputs[OUTPUT_DICT_LABELS_KEY]

    if not (0 < test_size < 1):
        raise Exception("Input test size must be in range 0-1.")
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=0)
    model = LogisticRegression(C=C)
    model.fit(x_train, y_train)
    model_coefficient = model.coef_
    cost = mean_squared_error(y_train, model.predict(x_train))
    test_results = model.predict(x_test)
    test_metrics = classification_report(y_test, test_results, output_dict=True)

    return {MODEL_OUTPUT_DICT_COST_KEY: cost,
            MODEL_OUTPUT_DICT_RESULTS_KEY: test_results,
            MODEL_OUTPUT_DICT_METRICS_KEY: test_metrics,
            MODEL_OUTPUT_DICT_FITTED_WEIGHT_KEY: model_coefficient}




