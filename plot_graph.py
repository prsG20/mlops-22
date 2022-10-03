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

hyperpara_comb = [{'gamma': g, 'C': c}for g in gamma_list for c in c_list]


#model hyperparameters
GAMMA = 0.001
C = 0.5
train_frac = 0.8
test_frac = 0.1
dev_frac = 0.1



#PART-2: LOAD DATASETS s
digits = datasets.load_digits()


#PART-2.1: Sanity Check Visualization of the data

_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, label in zip(axes, digits.images, digits.target):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title("Training: %i" % label)



#PART-3: Data Preprocessing -- to remove some noise, normalize data, 
# transformation to format the data to be consumed by the model

# flatten the images
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))
#model cannot take a 2d image as input so we convert every images into single array


#PART-4: Define the model

# Create a classifier: a support vector classifier
clf = svm.SVC() #support vector machine

#PART-4.1: Setting up the hyperparameters
hyper_params = {'gamma' : GAMMA}
clf.set_params(**hyper_params)

#PART-5: Define train/test/dev splits of experiment protocol
#train to train model
#dev to set hyperparameters of the model
#test to evaluate performance of the model (test on unseen data, to avoid overestimation of performance of the model)


dev_test_frac = 1 - train_frac
#Split into 80:10:10 :: train:dev:test
X_train, X_dev_test, y_train, y_dev_test = train_test_split(
    data, digits.target, test_size= dev_test_frac, shuffle=True
)
#Resplit
#Split dev_test variables
X_test, X_dev, y_test, y_dev = train_test_split(
    X_dev_test, y_dev_test, test_size=(dev_frac)/(dev_test_frac), shuffle=True
)

#PART-6: Train the model
# Learn the digits on the train subset
clf.fit(X_train, y_train)

#PART-7: Get test set predictions
# Predict the value of the digit on the test subset
predicted = clf.predict(X_test)


#PART-8: Sanity check of predictions
_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, prediction in zip(axes, X_test, predicted):
    ax.set_axis_off()
    image = image.reshape(8, 8)
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title(f"Prediction: {prediction}")


#PART-9: Compte Evaluation matrices
print(
    f"Classification report for classifier {clf}:\n"
    f"{metrics.classification_report(y_test, predicted)}\n"
)


disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predicted)
disp.figure_.suptitle("Confusion Matrix")
print(f"Confusion matrix:\n{disp.confusion_matrix}")

plt.show()