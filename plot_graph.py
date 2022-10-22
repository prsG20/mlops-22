# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# License: BSD 3 clause

#PART: Library Dependencies
import matplotlib.pyplot as plt
# Import datasets, classifier and performance metrics
from sklearn import datasets, svm, metrics
#from sklearn.model_selection import train_test_split
from utils import preprocess_digits, h_param_tuning, data_visualization, train_dev_test_split, pred_image_visualization, get_all_h_params_combo, train_saved_model
from joblib import dump, load

train_frac, test_frac, dev_frac = 0.8, 0.1, 0.1
assert train_frac + test_frac + dev_frac == 1.0

#model hyperparameters
gamma_list = [0.01, 0.005, 0.001, 0.0005, 0.0001 ]
c_list = [0.1, 0.2, 0.5, 0.7, 1, 2, 5, 7, 10]

params = {}
params['gamma'] = gamma_list
params['C'] = c_list

h_para_comb = get_all_h_params_combo(params)

#PART: Load dataset
digits = datasets.load_digits()
#PART: Visualize dataset
data_visualization(digits)
#PART: Preprocess dataset
data, label = preprocess_digits(digits)

del digits

X_train, y_train, X_dev, y_dev, X_test, y_test = train_dev_test_split(data, label, train_frac, test_frac)

#define model, create classifier: Support Vector Classifier
clf = svm.SVC()
#define metric
metric = metrics.accuracy_score
#PART: Hyperparameter tuning
best_h_params, best_model, best_metric = h_param_tuning(h_para_comb, clf, X_train, y_train, X_dev, y_dev, metric)

model_path = train_saved_model(X_train, y_train, X_dev, y_dev, data, label, train_frac, dev_frac, None, h_para_comb, best_h_params, best_model)

#2.Load the best_model from the disk
best_model = load(model_path)

#PART: Prediction on test data
predicted = best_model.predict(X_test)

#PART: Sanity check of predictions/ prediction visualization
pred_image_visualization(X_test, predicted)



#PART: Compute Evaluation metrics
print(
    f"Classification report for classifier {clf}:\n"
    f"{metrics.classification_report(y_test, predicted)}\n"
)

#disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predicted)
#disp.figure_.suptitle("Confusion Matrix")
#print(f"Confusion matrix:\n{disp.confusion_matrix}")

print("Best hyperparams were : " + str(best_h_params))