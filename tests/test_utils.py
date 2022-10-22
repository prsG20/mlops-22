import sys
sys.path.append('.')
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import svm, metrics, datasets
from joblib import dump, load
import pdb

from utils import preprocess_digits, h_param_tuning, data_visualization, train_dev_test_split, pred_image_visualization, get_all_h_params_combo, train_saved_model

#test to check if all hyperparameter combos are indeed getting created
def test_get_h_param_combo():
    gamma_list = [0.01, 0.005, 0.001, 0.0005, 0.0001 ]
    c_list = [0.1, 0.2, 0.5, 0.7, 1, 2, 5, 7, 10]
    
    params = {}
    params['gamma'] = gamma_list
    params['C'] = c_list

    h_para_comb = get_all_h_params_combo(params)

    assert len(h_para_comb) == len(gamma_list)*len(c_list)

#test to validate if the models are getting saved or not

#step1: test on small dataset, provide a path to save trained model
#step2: assert if the file exists at the provided path
#step3: assert if the file is a scikit-learn model
#step4: optinally checksome validate md5

def test_check_model_saving():
    
    #PART: Load dataset
    digits = datasets.load_digits()
    #PART: Visualize dataset
    data_visualization(digits)
    #PART: Preprocess dataset
    data, label = preprocess_digits(digits)
    sample_data = data[:500]
    sample_label = label[:500]

    #model hyperparameters
    gamma_list = [0.01, 0.0005, 0.0001 ]
    c_list = [0.1, 0.2, 5, 7, 10]

    params = {}
    params['gamma'] = gamma_list
    params['C'] = c_list

    train_frac, test_frac, dev_frac = 0.8, 0.1, 0.1
    assert train_frac + test_frac + dev_frac == 1.0

    h_para_comb = get_all_h_params_combo(params)
    #pdb.set_trace()
    X_train, y_train, X_dev, y_dev, X_test, y_test = train_dev_test_split(data, label, train_frac, test_frac)
    #define model, create classifier: Support Vector Classifier
    clf = svm.SVC()
    #define metric
    metric = metrics.accuracy_score
    #PART: Hyperparameter tuning
    best_h_params, best_model, best_metric = h_param_tuning(h_para_comb, clf, X_train, y_train, X_dev, y_dev, metric)

    best_param_config = "_".join([h +"="+ str(best_h_params[h]) for h in best_h_params])
    model_path = str("svm_" + best_param_config + ".joblib")
    #train_saved_model(X_train, y_train, X_dev, y_dev, data,label,train_frac,dev_frac,model_path,h_para_comb, best_h_params, best_model)
    actual_model_path = train_saved_model(sample_data, sample_label, sample_data, sample_label, data, label, train_frac, dev_frac, model_path, h_para_comb, best_h_params, best_model)
    
    assert actual_model_path == model_path
    assert os.path.exists(model_path)

    loaded_model = load(model_path)
    assert type(loaded_model) == type(clf)