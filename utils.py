OUTPUT_DICT_DATA_POINT_KEY = 'features'
OUTPUT_DICT_LABELS_KEY = 'labels'
OUTOUT_DICT_WEIGHT_KEY = 'weights'
OUTPUT_DICT_REG_STRENGTH_KEY = 'regularization_strength'
OUTPUT_DICT_TEST_SIZE_KEY = 'test_size'
OUTPUT_DICT_B_KEY = 'intercept'


MODEL_OUTPUT_DICT_RESULTS_KEY = 'results' # (n_samples, ) Vector containing the class labels for each sample.
MODEL_OUTPUT_DICT_COST_KEY = 'cost' # float
MODEL_OUTPUT_DICT_METRICS_KEY = 'metrics' 
# Dictionary: Text summary of the precision, recall, F1 score for each class.
# {'label 1': {'precision':0.5,
#              'recall':1.0,
#              'f1-score':0.67,
#              'support':1},
#  'label 2': { ... },
#   ...
# }
MODEL_OUTPUT_DICT_FITTED_WEIGHT_KEY = 'fitted_weights' # (1, n_features)
MODEL_OUTPUT_DICT_FIGURE = 'figure'
MODEL_OUTPUT_DICT_PROB_KEY = 'pred_prob_manual'
