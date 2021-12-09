import numpy as np
from hyperopt import hp, tpe, fmin, space_eval, Trials
from hyperopt.pyll.base import scope
import pickle
import xgboost as xgb
from skmultilearn.model_selection.iterative_stratification import iterative_train_test_split
from sklearn.utils import class_weight
from sklearn.multioutput import MultiOutputClassifier
from tensorflow.keras.losses import BinaryCrossentropy
import gc

from data_set_params import DataSetParams

def tune_xgb(params, feat_train, labels, trials_filename):
    """
    Tunes the XGBoost with hyperopt.

    Parameters
    ------------
    params : DataSetParams
        Parameters of the model.
    feat_train : ndarray
        Array containing the spectrogram features for each training window of the audio file.
    labels : ndarray
        Class labels in one-hot encoding for each training window of the audio files.
    trials_filename : String
        Name of the file where the previous iterations of hyperopt are saved.
    """

    print("\n tune xgboost")
    space_xgb = {   'eta': hp.choice('eta', np.logspace(-4, -0.522879, num=20)),
                    'min_child_weight': hp.choice('min_child_weight', [1,2,3]),
                    'max_depth': scope.int(hp.quniform('max_depth', 3, 15, 1)),
                    'n_estimators': hp.choice('n_estimators', [100, 500, 1000, 2000]),
                    'gamma_xgb': hp.choice('gamma_xgb', [0, 0.0001, 0.005, 0.001, 0.005, 0.01]),
                    'subsample': hp.choice('subsample', [0.7, 0.8, 0.9, 1]),
                    'scale_pos_weight': hp.choice('scale_pos_weight', [0, 0.25, 0.5, 1, 1.5]),
                    'objective': 'binary:logistic',
                    'eval_metric': 'mlogloss',
                    'model': "xgboost",
                    'feat_train': feat_train,
                    'labels': labels
                }

    # load the saved trials
    try:
        trials = pickle.load(open(trials_filename+".hyperopt", "rb"))
        max_trials = len(trials.trials) + 1
    # create a new trials
    except:
        max_trials = 1
        trials = Trials()

    # optimise the objective function with the defined set of XGBoost parameters
    best_space_indices = fmin(obj_func_xgb, space_xgb, trials=trials, algo=tpe.suggest, max_evals=max_trials)
    best_space = space_eval(space_xgb, best_space_indices)
    best_space = {k: best_space[k] for k in best_space.keys() - {'feat_train', 'labels'}}
    with open(trials_filename + ".hyperopt", "wb") as f:
        pickle.dump(trials, f)

    params.eta = best_space['eta']
    params.min_child_weight = best_space['min_child_weight']
    params.max_depth = best_space['max_depth']
    params.n_estimators = best_space['n_estimators']
    params.gamma_xgb = best_space['gamma_xgb']
    params.subsample = best_space['subsample']
    params.scale_pos_weight = best_space['scale_pos_weight']


def obj_func_xgb(args):
    """
    Fits and returns the best loss of an XGBoost model with given parameters.

    Parameters
    -----------
    args : dict
        Dictionnary of all the parameters needed to fit an XGBoost.

    Returns
    --------
    loss : float
        minimum value of the loss during training of the XGBoost.
    """

    # split dataset into training and validation set
    train_feat, train_labels, val_feat, val_labels = iterative_train_test_split(args['feat_train'], args['labels'], 0.1)

    # Fit XGBoost and compute minimum loss
    params_xgb = DataSetParams()
    params_xgb.eta = args['eta']
    params_xgb.min_child_weight = args['min_child_weight']
    params_xgb.max_depth = args['max_depth']
    params_xgb.n_estimators = args['n_estimators']
    params_xgb.gamma_xgb = args['gamma_xgb']
    params_xgb.subsample = args['subsample']
    params_xgb.scale_pos_weight = args['scale_pos_weight']
    print_params_xgb(params_xgb)
    xgb_clf = xgb.XGBClassifier(eta=params_xgb.eta, min_child_weight=params_xgb.min_child_weight, max_depth=params_xgb.max_depth,
                            n_estimators=params_xgb.n_estimators, gamma=params_xgb.gamma_xgb, subsample=params_xgb.subsample,
                            scale_pos_weight=params_xgb.scale_pos_weight, objective="binary:logistic",
                            tree_method='gpu_hist', predictor='gpu_predictor')
    clf = MultiOutputClassifier(xgb_clf)
    sample_w = class_weight.compute_sample_weight('balanced',train_labels)
    clf.fit(train_feat, train_labels, sample_weight=sample_w)
    y_pred_train = clf.predict_proba(val_feat)
    y_pred_train = np.array(y_pred_train)[:,:,1].T
    sample_w = class_weight.compute_sample_weight('balanced',val_labels)
    bce = BinaryCrossentropy()
    loss = bce(val_labels, y_pred_train, sample_weight=sample_w).numpy()
    
    # Free memory
    for clf_estimator in clf.estimators_:
        clf_estimator._Booster.__del__()
    gc.collect()
    return loss

def print_params_xgb(params):
    """
    Prints the parameters of XGBoost.

    Parameters
    ------------
    params : DataSetParams
        Parameters of the model.
    """
    
    dic = {}
    dic["eta"] = params.eta
    dic['min_child_weight'] = params.min_child_weight
    dic["max_depth"] = params.max_depth
    dic["n_estimators"] = params.n_estimators
    dic["gamma_xgb"] = params.gamma_xgb
    dic["subsample"] = params.subsample
    dic["scale_pos_weight"] = params.scale_pos_weight
    print("params used for XGBoost =", dic)