"""
================================
Recognizing hand-written digits
================================

This example shows how scikit-learn can be used to recognize images of
hand-written digits, from 0-9.

"""

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
clf.set_params(**hyperpara_comb)

#model hyperparameters
GAMMA = 0.001
C = 0.5
train_frac = 0.8
test_frac = 0.1
dev_frac = 0.1


#PART-2: LOAD DATASETS 

###############################################################################
# Digits dataset
# --------------
#
# The digits dataset consists of 8x8
# pixel images of digits. The ``images`` attribute of the dataset stores
# 8x8 arrays of grayscale values for each image. We will use these arrays to
# visualize the first 4 images. The ``target`` attribute of the dataset stores
# the digit each image represents and this is included in the title of the 4
# plots below.
#
# Note: if we were working from image files (e.g., 'png' files), we would load
# them using :func:`matplotlib.pyplot.imread`.

digits = datasets.load_digits()

#PART-2.1: Sanity Check Visualization of the data

_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, label in zip(axes, digits.images, digits.target):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title("Training: %i" % label)

###############################################################################
# Classification
# --------------
#
# To apply a classifier on this data, we need to flatten the images, turning
# each 2-D array of grayscale values from shape ``(8, 8)`` into shape
# ``(64,)``. Subsequently, the entire dataset will be of shape
# ``(n_samples, n_features)``, where ``n_samples`` is the number of images and
# ``n_features`` is the total number of pixels in each image.
#
# We can then split the data into train and test subsets and fit a support
# vector classifier on the train samples. The fitted classifier can
# subsequently be used to predict the value of the digit for the samples
# in the test subset.

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
'''
X_test, X_dev, y_test, y_dev = train_test_split(
    X_dev_test, y_dev_test, test_size=(dev_frac)/(test_frac + dev_frac), shuffle=True
)
or
X_dev, X_test, y_dev, y_test = train_test_split(
    X_dev_test, y_dev_test, test_size=(test_frac)/(test_frac + dev_frac), shuffle=True
)
'''

#PART-6: Train the model
# Learn the digits on the train subset
clf.fit(X_train, y_train)

#PART-7: Get test set predictions
# Predict the value of the digit on the test subset
predicted = clf.predict(X_test)

###############################################################################
# Below we visualize the first 4 test samples and show their predicted
# digit value in the title.

#PART-8: Sanity check of predictions
_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, prediction in zip(axes, X_test, predicted):
    ax.set_axis_off()
    image = image.reshape(8, 8)
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title(f"Prediction: {prediction}")

###############################################################################
# :func:`~sklearn.metrics.classification_report` builds a text report showing
# the main classification metrics.

#PART-9: Compte Evaluation matrices
print(
    f"Classification report for classifier {clf}:\n"
    f"{metrics.classification_report(y_test, predicted)}\n"
)

###############################################################################
# We can also plot a :ref:`confusion matrix <confusion_matrix>` of the
# true digit values and the predicted digit values.

disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predicted)
disp.figure_.suptitle("Confusion Matrix")
print(f"Confusion matrix:\n{disp.confusion_matrix}")

plt.show()
