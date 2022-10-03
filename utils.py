import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def preprocess_digits(dataset):
    n_samples = len(dataset.images)
    data = dataset.images.reshape((n_samples, -1))
    label = dataset.target
    return data, label

#other types of preprocessing
#image resize, image flattening
#normalize data: mean normalization, min-max normalization
#smoothening of image: blur on image

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

def pred_image_visualization(X_test, predictions):
    _, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
    for ax, image, prediction in zip(axes, X_test, predictions):
        ax.set_axis_off()
        image = image.reshape(8, 8)
        ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
        ax.set_title(f"Prediction: {prediction}")