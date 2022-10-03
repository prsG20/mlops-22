# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# License: BSD 3 clause

#PART-1: Library Dependencies

# Standard scientific Python imports
import matplotlib.pyplot as plt

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split


gamma_list = [0.01, 0.005, 0.001, 0.0005, 0.0001 ]
c_list = [0.1, 0.2, 0.5, 0.7, 1, 2, 5, 7, 10]

h_para_comb = [{'gamma': g, 'C': c}for g in gamma_list for c in c_list]

assert len(h_para_comb) == len(gamma_list)*len(c_list)

#model hyperparameters
GAMMA = 0.001
C = 0.5
train_frac = 0.8
test_frac = 0.1
dev_frac = 0.1



def preprocess_digits(dataset):
    n_samples = len(dataset.images)
    data = dataset.images.reshape((n_samples, -1))
    label = dataset.target
    return data, label

def data_visualization(dataset):
    _, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
    for ax, image, label in zip(axes, dataset.images, dataset.target):
        ax.set_axis_off()
        ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
        ax.set_title("Training: %i" % label)

def train_dev_test_split(data, label, train_frac, dev_frac):
    dev_test_frac = 1 - train_frac
    X_train, X_dev_test, y_train, y_dev_test = train_test_split(
        data, label, test_size= dev_test_frac, shuffle=True
    )
    X_test, X_dev, y_test, y_dev = train_test_split(
        X_dev_test, y_dev_test, test_size=(dev_frac)/(dev_test_frac), shuffle=True
    )

    return X_train, y_train, X_dev, y_dev, X_test, y_test

def h_param_tuning(h_para_comb, clf, X_train, y_train, X_dev, y_dev, metric):
    best_metric = -1.0
    best_model = None
    best_h_params = None

    for cur_h_params in h_para_comb:

        #setting up the hyperparameters
        hyper_params = cur_h_params
        clf.set_params(**hyper_params)

        #train the model
        #learn the digits on the train subset
        clf.fit(X_train, y_train)

        #get test set predictions
        #predict the value of the digit on the test subset
        predicted_dev = clf.predict(X_dev)

        #compute accuracy on validation set
        cur_metric = metric(y_pred= predicted_dev, y_true= y_dev)

        if cur_metric > best_metric:
            best_metric = cur_metric
            best_model = clf
            best_h_params = cur_h_params
            print("Best metric : " + str(cur_metric))
            print("Best hyperparameters : " + str(cur_h_params))

    return best_h_params, best_model, best_metric


digits = datasets.load_digits()
data_visualization(digits)
data, label = preprocess_digits(digits)

del digits

X_train, y_train, X_dev, y_dev, X_test, y_test = train_dev_test_split(data, label, train_frac, dev_frac)



#define model, create classifier: Support Vector Classifier
clf = svm.SVC()
metric = metrics.accuracy_score
best_h_params, best_model, best_metric = h_param_tuning(h_para_comb, clf, X_train, y_train, X_dev, y_dev, metric)


predicted = best_model.predict(X_test)

#PART-8: Sanity check of predictions
_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, prediction in zip(axes, X_test, predicted):
    ax.set_axis_off()
    image = image.reshape(8, 8)
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title(f"Prediction: {prediction}")


#PART-9: Compte Evaluation metrics
print(
    f"Classification report for classifier {clf}:\n"
    f"{metrics.classification_report(y_test, predicted)}\n"
)

#disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predicted)
#disp.figure_.suptitle("Confusion Matrix")
#print(f"Confusion matrix:\n{disp.confusion_matrix}")

print("Best hyperparams were : " + str(best_h_params))