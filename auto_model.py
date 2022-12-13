import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from utils import * 
import matplotlib.pyplot as plt
from scipy.special import expit


def plot_logistic_curve(x, y, model):
    figure_x_values = np.sum(np.multiply(x, model.coef_), axis=1) + model.intercept_
    figure_y_values = expit(figure_x_values).ravel()
    predict_result  = model.predict(x)
    wrong_mask = (predict_result != y)
    predict_result[wrong_mask] = 2
    class0_mask = (predict_result == 0)
    class1_mask = (predict_result == 1)
    class0_x = figure_x_values[class0_mask]
    class1_x = figure_x_values[class1_mask]
    wrong_x = figure_x_values[wrong_mask]

    plt.figure(1, figsize=(4, 4))
    plt.scatter(class0_x, [0] * len(class0_x), label="class 0", color="black", zorder=20)
    plt.scatter(class1_x, [1] * len(class1_x), label="class 1", color="blue", zorder=20)
    plt.scatter(wrong_x, y[wrong_mask], label="wrong predict", color="red", zorder=20, marker='x')

    plt.plot(figure_x_values, figure_y_values, label="Logistic Regression Model")
    plt.ylabel("Probability(sigmoid(x))")
    plt.xlabel("Measurement(x)")
    plt.legend(
    loc="lower right",
    fontsize="small",)
    # plt.show()
    return plt.gcf()


def auto_logistic_regression(inputs):
    test_size = inputs[OUTPUT_DICT_TEST_SIZE_KEY]
    C = 1 / inputs[OUTPUT_DICT_REG_STRENGTH_KEY]
    x = inputs[OUTPUT_DICT_DATA_POINT_KEY]
    y = inputs[OUTPUT_DICT_LABELS_KEY]
    y = y.reshape(-1)

    if not (0 < test_size < 1):
        raise Exception("Input test size must be in range 0-1.")
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=0)
    model = LogisticRegression(C=C)
    model.fit(x_train, y_train)
    model_coefficient = model.coef_
    cost = mean_squared_error(y_train, model.predict(x_train))
    test_results = model.predict(x_test)
    test_metrics = classification_report(y_test, test_results, output_dict=True)

    fig = plot_logistic_curve(x, y, model)

    return {MODEL_OUTPUT_DICT_COST_KEY: cost,
            MODEL_OUTPUT_DICT_RESULTS_KEY: test_results,
            MODEL_OUTPUT_DICT_METRICS_KEY: test_metrics,
            MODEL_OUTPUT_DICT_FITTED_WEIGHT_KEY: model_coefficient,
            MODEL_OUTPUT_DICT_FIGURE: fig}

# def readCSV2Ndarray(filePath):
#     if not os.path.exists(filePath):
#         raise Exception("File: " + filePath + " does not exist.")

#     npArray = None
#     try:
#         npArray = np.genfromtxt(filePath, delimiter=",")
#         if len(npArray.shape) == 1:
#             npArray = np.expand_dims(npArray, axis=1)
#     except Exception:
#         raise Exception("File: " + filePath +
#                         " is not a valid CSV File. (Numpy Parsing Failed)")

#     return npArray
    
# def test():
#     data = readCSV2Ndarray('mock/5x20/data_point.csv')
#     label = readCSV2Ndarray('mock/5x20/label.csv')

#     inputs = {OUTPUT_DICT_DATA_POINT_KEY: data,
#                     OUTPUT_DICT_LABELS_KEY: label,
#                     OUTPUT_DICT_REG_STRENGTH_KEY: 1,
#                     OUTPUT_DICT_TEST_SIZE_KEY: 0.1}

#     auto_logistic_regression(inputs)


# if __name__ == '__main__':

#     test()