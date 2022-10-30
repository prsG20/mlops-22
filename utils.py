import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import svm, metrics, datasets
from joblib import dump, load
import pdb

def get_all_h_params_combo(params):
    h_para_comb = [{'gamma': g, 'C': c}for g in params['gamma'] for c in params['C']]
    return h_para_comb

def get_all_h_params_combo_tree(params):
    h_para_comb = [{'max_depth' : m, 'criterion' : c} for m in params['max_depth'] for c in params['criterion']]
    return h_para_comb

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

    print("Train:Dev:Test :: " + str(train_frac*100) + " : "+ str(dev_frac*100) + " : "+ str(dev_frac*100))
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

def train_saved_model(X_train, y_train, X_dev, y_dev, data,label,train_frac,dev_frac,model_path,h_para_comb, best_h_params, best_model, clf):

    #1.Save the best_model to the disk
    best_param_config = "_".join([h +"="+ str(best_h_params[h]) for h in best_h_params])
    if model_path is None:
        model_path = str(clf + "_"+ best_param_config + ".joblib")
    dump(best_model, model_path)

    return model_path

def tune_and_save(h_para_comb, model_of_choice, X_train, y_train, X_dev, y_dev, metric, clf, data, label, train_frac, dev_frac):
    #PART: Hyperparameter tuning
    best_h_params, best_model, best_metric = h_param_tuning(h_para_comb, model_of_choice, X_train, y_train, X_dev, y_dev, metric)

    model_path = train_saved_model(X_train, y_train, X_dev, y_dev, data, label, train_frac, dev_frac, None, h_para_comb, best_h_params, best_model, clf)

    #2.Load the best_model from the disk
    best_model = load(model_path)

    return best_model