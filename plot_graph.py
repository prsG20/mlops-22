# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# License: BSD 3 clause

#PART: Library Dependencies
import matplotlib.pyplot as plt
# Import datasets, classifier and performance metrics
from sklearn import datasets, svm, metrics, tree
#from sklearn.model_selection import train_test_split
from utils import get_all_h_params_combo_tree, preprocess_digits, data_visualization, train_dev_test_split, pred_image_visualization, get_all_h_params_combo, tune_and_save
import numpy as np


train_frac, test_frac, dev_frac = 0.8, 0.1, 0.1
assert train_frac + test_frac + dev_frac == 1.0

#model hyperparameters
gamma_list = [0.01, 0.005, 0.001, 0.0005, 0.0001 ]
c_list = [0.1, 0.2, 0.5, 0.7, 1, 2, 5, 7, 10]
svm_params = {}
svm_params['gamma'] = gamma_list
svm_params['C'] = c_list
svm_h_para_comb = get_all_h_params_combo(svm_params)

max_depth_list = [10, 20, 30, 40, 50]
criterion_list = ["gini", "entropy"]
tree_params = {}
tree_params['max_depth'] = max_depth_list
tree_params['criterion'] = criterion_list 
tree_h_para_comb = get_all_h_params_combo_tree(tree_params)

h_para_comb = {'svm': svm_h_para_comb, 'decision_tree': tree_h_para_comb}

#PART: Load dataset
digits = datasets.load_digits()
#PART: Visualize dataset
data_visualization(digits)
#PART: Preprocess dataset
data, label = preprocess_digits(digits)

del digits

train_fracs_ls = [0.9, 0.8, 0.7, 0.6, 0.5]
metrics_ls = []

#define model, create classifier: Support Vector Classifier
#clf = svm.SVC()
#define model, create Decision Tree classifier
#clf2 = tree.DecisionTreeClassifier()

model_of_choice = {'svm': svm.SVC(), 'decision_tree': tree.DecisionTreeClassifier()}
#model_of_choice = [svm.SVC(), tree.DecisionTreeClassifier()]
#define metric

metric = metrics.accuracy_score

results = {}
for i in range(5):

    dev_frac = (1 - train_fracs_ls[i])/2
    X_train, y_train, X_dev, y_dev, X_test, y_test = train_dev_test_split(data, label, train_fracs_ls[i], dev_frac)

    for clf in model_of_choice:

        best_model = tune_and_save(h_para_comb[clf], model_of_choice[clf], X_train, y_train, X_dev, y_dev, metric, clf, data, label, train_frac, dev_frac)

        #PART: Prediction on test data
        predicted = best_model.predict(X_test)

        if not clf in results:
            results[clf] = []
        
        results[clf].append(metric(predicted, y_test))

        #PART: Compute Evaluation metrics
        print(
            f"Classification report for classifier {clf}:\n"
            f"{metrics.classification_report(y_test, predicted)}\n"
        )

print("Accuracy" + str(results))

mean, std_dev = {'svm': [], 'decision_tree': []}, {'svm': [], 'decision_tree': []}

for clf in results:
    if not clf in mean and std_dev:
        mean[clf] = None
        std_dev[clf] = None
    mean[clf] = np.mean(results[clf])
    std_dev[clf] = np.std(results[clf])

print("Mean:" + str(mean))
#print(mean)
print("Standard Deviation:" + str(std_dev))
#print(std_dev)


#PART: Sanity check of predictions/ prediction visualization
#pred_image_visualization(X_test, predicted)


#disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predicted)
#disp.figure_.suptitle("Confusion Matrix")
#print(f"Confusion matrix:\n{disp.confusion_matrix}")

#print("Best hyperparams were : " + str(best_h_params))