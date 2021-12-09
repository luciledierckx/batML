import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, regularizers
from sklearn.utils import class_weight
from hyperopt import hp, tpe, fmin, space_eval, Trials
import pickle
import gc

from data_set_params import DataSetParams
from models_params_helper import params_to_dict

#the code for the resnet blocs and architecture was taken from https://androidkt.com/resnet-implementation-in-tensorflow-keras/
def identity_block(params, input_tensor, kernel_size, filters, nb_cnn):
    """
    Block that has no conv layer at shortcut.

    Parameters
    -----------
    params : DataSetParams
        Parameters of the model.
    input_tensor : tensor
        input tensor of the block.
    kernel_size : int
        the kernel size of middle conv layer at main path.
    filters : list
        list of integers, the filters of 3 conv layer at main path.
    nb_cnn : String
        "_1" or "_2" if the double resnet architecture and "" otherwise.
    
    Returns
    --------
    x : tensor
        The tensor for the block.
    """
    filters1, filters2, filters3 = filters
    batch_norm_decay = getattr(params, "batch_norm_decay"+nb_cnn)
    L2_weight_decay = getattr(params, "L2_weight_decay"+nb_cnn)
    batch_norm_epsilon = getattr(params, "batch_norm_epsilon"+nb_cnn) 
    bn_axis = 3

    x = layers.Conv2D(filters1, (1, 1), use_bias=False,
                      kernel_initializer='he_normal',
                      kernel_regularizer=regularizers.l2(L2_weight_decay))(input_tensor)

    x = layers.BatchNormalization(axis=bn_axis,
                                  momentum=batch_norm_decay,
                                  epsilon=batch_norm_epsilon)(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters2, kernel_size,
                      padding='same', use_bias=False,
                      kernel_initializer='he_normal',
                      kernel_regularizer=regularizers.l2(L2_weight_decay))(x)

    x = layers.BatchNormalization(axis=bn_axis,
                                  momentum=batch_norm_decay,
                                  epsilon=batch_norm_epsilon)(x)

    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters3, (1, 1), use_bias=False,
                      kernel_initializer='he_normal',
                      kernel_regularizer=regularizers.l2(L2_weight_decay))(x)

    x = layers.BatchNormalization(axis=bn_axis,
                                  momentum=batch_norm_decay,
                                  epsilon=batch_norm_epsilon)(x)

    x = layers.add([x, input_tensor])
    x = layers.Activation('relu')(x)
    return x

def conv_block(params, input_tensor, kernel_size, filters, nb_cnn, strides=(2, 2)):
    """
    Block that has a conv layer at shortcut.

    Parameters
    -----------
    params : DataSetParams
        Parameters of the model.
    input_tensor : tensor
        input tensor of the block.
    kernel_size : int
        the kernel size of middle conv layer at main path.
    filters : list
        list of integers, the filters of 3 conv layer at main path.
    nb_cnn : String
        "_1" or "_2" if the double resnet architecture and "" otherwise.
    strides : tuple
        strides of the second conv layer and of the shortcut.
        
    Returns
    --------
    x : tensor
        The tensor for the block.
    """

    filters1, filters2, filters3 = filters
    batch_norm_decay = getattr(params, "batch_norm_decay"+nb_cnn)
    L2_weight_decay = getattr(params, "L2_weight_decay"+nb_cnn)
    batch_norm_epsilon = getattr(params, "batch_norm_epsilon"+nb_cnn)
    bn_axis = 3

    x = layers.Conv2D(filters1, (1, 1), use_bias=False,
                      kernel_initializer='he_normal',
                      kernel_regularizer=regularizers.l2(L2_weight_decay))(input_tensor)
    x = layers.BatchNormalization(axis=bn_axis,
                                  momentum=batch_norm_decay,
                                  epsilon=batch_norm_epsilon)(x)
    x = layers.Activation('relu')(x)


    x = layers.Conv2D(filters2, kernel_size, strides=strides, padding='same',
                      use_bias=False, kernel_initializer='he_normal',
                      kernel_regularizer=regularizers.l2(L2_weight_decay))(x)
    x = layers.BatchNormalization(axis=bn_axis,
                                  momentum=batch_norm_decay,
                                  epsilon=batch_norm_epsilon)(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters3, (1, 1), use_bias=False,
                      kernel_initializer='he_normal',
                      kernel_regularizer=regularizers.l2(L2_weight_decay))(x)
    x = layers.BatchNormalization(axis=bn_axis,
                                  momentum=batch_norm_decay,
                                  epsilon=batch_norm_epsilon)(x)

    shortcut = layers.Conv2D(filters3, (1, 1), strides=strides, use_bias=False,
                             kernel_initializer='he_normal',
                             kernel_regularizer=regularizers.l2(L2_weight_decay))(input_tensor)
    shortcut = layers.BatchNormalization(axis=bn_axis,
                                         momentum=batch_norm_decay,
                                         epsilon=batch_norm_epsilon)(shortcut)

    x = layers.add([x, shortcut])
    x = layers.Activation('relu')(x)
    return x

def build_resnet(params, ip_size, nb_output, nb_cnn):
    """
    Builds a ResNet50 with the specified parameters.

    Parameters
    -----------
    params : DataSetParams
        Parameters of the model.
    ip_size : numpy array
        Dimention of the input to the ResNet.
    nb_output : int
        Number of output nodes.
    nb_cnn : String
        "_1" or "_2" if the double resnet architecture and "" otherwise.

    Returns
    --------
    net : Model
        The ResNet50 model.
    """
    batch_norm_decay = getattr(params, "batch_norm_decay"+nb_cnn)
    L2_weight_decay = getattr(params, "L2_weight_decay"+nb_cnn)
    batch_norm_epsilon = getattr(params, "batch_norm_epsilon"+nb_cnn)

    input_l = layers.Input(shape=(ip_size[0], ip_size[1], 1))

    # channels_last
    x = input_l
    bn_axis = 3

    # Conv1 (7x7,64,stride=2)
    x = layers.ZeroPadding2D(padding=(3, 3))(x)

    x = layers.Conv2D(64, (7, 7),
                        strides=(2, 2),
                        padding='valid', use_bias=False,
                        kernel_initializer='he_normal',
                        kernel_regularizer=regularizers.l2(L2_weight_decay))(x)
    x = layers.BatchNormalization(axis=bn_axis,
                                    momentum=batch_norm_decay,
                                    epsilon=batch_norm_epsilon)(x)
    x = layers.Activation('relu')(x)
    x = layers.ZeroPadding2D(padding=(1, 1))(x)

    # 3x3 max pool,stride=2
    x = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

    # Conv2_x
    # 1×1, 64
    # 3×3, 64
    # 1×1, 256

    x = conv_block(params, x, 3, [64, 64, 256], nb_cnn, strides=(1, 1))
    x = identity_block(params, x, 3, [64, 64, 256], nb_cnn)
    x = identity_block(params, x, 3, [64, 64, 256], nb_cnn)

    # Conv3_x
    # 1×1, 128
    # 3×3, 128
    # 1×1, 512

    x = conv_block(params, x, 3, [128, 128, 512], nb_cnn)
    x = identity_block(params, x, 3, [128, 128, 512], nb_cnn)
    x = identity_block(params, x, 3, [128, 128, 512], nb_cnn)
    x = identity_block(params, x, 3, [128, 128, 512], nb_cnn)

    # Conv4_x
    # 1×1, 256
    # 3×3, 256
    # 1×1, 1024
    x = conv_block(params, x, 3, [256, 256, 1024], nb_cnn)
    x = identity_block(params, x, 3, [256, 256, 1024], nb_cnn)
    x = identity_block(params, x, 3, [256, 256, 1024], nb_cnn)
    x = identity_block(params, x, 3, [256, 256, 1024], nb_cnn)
    x = identity_block(params, x, 3, [256, 256, 1024], nb_cnn)
    x = identity_block(params, x, 3, [256, 256, 1024], nb_cnn)

    # 1×1, 512
    # 3×3, 512
    # 1×1, 2048
    x = conv_block(params, x, 3, [512, 512, 2048], nb_cnn)
    x = identity_block(params, x, 3, [512, 512, 2048], nb_cnn)
    x = identity_block(params, x, 3, [512, 512, 2048], nb_cnn)

    # average pool, softmax
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(
        nb_output, activation='sigmoid',
        kernel_regularizer=regularizers.l2(L2_weight_decay),
        bias_regularizer=regularizers.l2(L2_weight_decay))(x)

    # Create model.
    return tf.keras.models.Model(input_l, x, name='resnet50')
    
def network_fit(params, features, labels, labels_not_merged, nb_output, nb_cnn=''):
    """
    Build and fit the ResNet.

    Parameters
    ------------
    params : DataSetParams
        Parameters of the model.
    features : ndarray
        Array containing the spectrogram features for each window of the audio file.
    labels : ndarray
        Class labels in one-hot encoding for each position of the audio files.
    labels_not_merged : ndarray
        Array containing one class label per call instead of per position in one-hot encoding.
        (Used to compute the class weights.)
    nb_output : int
        Number of output nodes.
    nb_cnn : String
        "_1" or "_2" if the double resnet architecture and "" otherwise.
    
    Returns
    --------
    network : Model
        Fit ResNet.
    history : list
        History of the monitored metrics for each epoch.
    """
    
    tf.keras.backend.clear_session()
    gc.collect()

    print("ResNet params= ", params_to_dict(params))
    learn_rate_adam = getattr(params, "learn_rate_adam"+nb_cnn)
    beta_1 = getattr(params, "beta_1"+nb_cnn)
    beta_2 = getattr(params, "beta_2"+nb_cnn)
    epsilon = getattr(params, "epsilon"+nb_cnn)
    min_delta = getattr(params, "min_delta"+nb_cnn)
    patience = getattr(params, "patience"+nb_cnn)
    batchsize = getattr(params, "batchsize"+nb_cnn)

    # Build the resnet
    network = build_resnet(params, features.shape[2:], nb_output, nb_cnn)
    if nb_output==2: loss_fn = "sparse_categorical_crossentropy"
    else: loss_fn = "binary_crossentropy"
    opti = tf.keras.optimizers.Adam( learning_rate=learn_rate_adam, beta_1=beta_1, beta_2=beta_2,
                                    epsilon=epsilon, name="Adam")
    network.compile(optimizer=opti, loss=loss_fn, metrics=['accuracy'])
    if nb_output!=2: labels_not_merged = np.argmax(labels_not_merged, axis=1)
    class_w = class_weight.compute_class_weight('balanced', classes=np.unique(labels_not_merged), y=labels_not_merged)
    class_w = dict(enumerate(class_w))
    callback = tf.keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=min_delta, patience=patience,
                                                verbose=1, restore_best_weights=params.restore_best_weights)
    features = features.reshape(features.shape[0], features.shape[2], features.shape[3], 1)
    
    # Fit the resnet
    print("Fit the ResNet")

    history = network.fit( features, labels, epochs=params.num_epochs, batch_size=batchsize,
                                shuffle=True, verbose=2, class_weight=class_w,
                                validation_split=params.validation_split, callbacks=[callback])
    return network, history

def obj_func_cnn(args):
    """
    Fits and returns the best loss of a resnet with given parameters.

    Parameters
    -----------
    args : dict
        Dictionnary of all the parameters needed to fit a resnet.

    Returns
    --------
    min_loss : float
        minimum value of the loss during training of the resnet.
    """
    
    params_resnet = DataSetParams()
    # ResNet
    params_resnet.L2_weight_decay = args['L2_weight_decay']
    params_resnet.batch_norm_decay = args['batch_norm_decay']
    params_resnet.batch_norm_epsilon = args['batch_norm_epsilon']
    #Adam
    params_resnet.learn_rate_adam = args['learn_rate_adam']
    params_resnet.beta_1 = args['beta_1']
    params_resnet.beta_2 = args['beta_2']
    params_resnet.epsilon = args['epsilon']
    # early stopping
    params_resnet.min_delta = args['min_delta']
    params_resnet.patience = args['patience']
    # fit
    params_resnet.batchsize = args['batchsize']

    _, history = network_fit(params_resnet, args['features'], args['labels'], args['labels_not_merged'], args['nb_output'])
    min_loss = np.min(history.history['val_loss'])
    return min_loss

def tune_network(params, features, labels, labels_not_merged, trials_filename, goal=None):
    """
    Tunes the network with hyperopt.

    Parameters
    ------------
    params : DataSetParams
        Parameters of the model.
    features : ndarray
        Array containing the spectrogram features for each window of the audio file.
    labels : ndarray
        Class labels in one-hot encoding for each position of the audio files.
    labels_not_merged : ndarray
        Array containing one class label per call instead of per position in one-hot encoding.
        (Used to compute the class weights.)
    trials_filename : String
        Name of the file where the previous iterations of hyperopt are saved.
    goal : String
        Indicates whether the network needs to be tuned for detection or classification.
        Can be either "detection" or "classification".
    """

    print("\n tune resnet")
    nb_output = 8
    if goal == "detection":
        nb_output = 2
    elif goal == "classification": 
        nb_output = 7
    space_resnet = {'L2_weight_decay': hp.choice('L2_weight_decay', [0.1, 0.05,0.01,0.005, 0.001, 0.0005, 0.0001, 0.00005, 0.00001]),
                    'batch_norm_decay': hp.choice('batch_norm_decay', [0.99]), 
                    'batch_norm_epsilon': hp.choice('batch_norm_epsilon', [0.001]),
                    'learn_rate_adam': hp.choice('learn_rate_adam', np.logspace(-5, -2, num=15)),
                    'beta_1': hp.choice('beta_1', [0.8, 0.9, 0.95]),
                    'beta_2': hp.choice('beta_2', [0.95, 0.999]),
                    'epsilon': hp.choice('epsilon', [1e-8]),
                    'min_delta': hp.choice('min_delta', [0.00005, 0.0005, 0.005]),
                    'patience': hp.choice('patience', [5, 10, 15, 20]),
                    'batchsize': hp.choice('batchsize', range(32, 129, 32)),
                    'features': features,
                    'labels': labels,
                    'labels_not_merged': labels_not_merged,
                    'nb_output': nb_output
                }
    
    # load the saved trials
    try:
        trials = pickle.load(open(trials_filename+".hyperopt", "rb"))
        max_trials = len(trials.trials) + 1
    # create a new trials
    except:
        max_trials = 1
        trials = Trials()

    # optimise the objective function with the defined set of resnet parameters
    best_space_indices = fmin(obj_func_cnn, space_resnet, trials=trials, algo=tpe.suggest, max_evals=max_trials)
    best_space = space_eval(space_resnet, best_space_indices)
    best_space = {k: best_space[k] for k in best_space.keys() - {'features', 'labels', 'labels_not_merged'}}
    print("best_space=",best_space)
    with open(trials_filename + ".hyperopt", "wb") as f:
        pickle.dump(trials, f)

    # ResNet
    params.L2_weight_decay = best_space['L2_weight_decay']
    params.batch_norm_decay = best_space['batch_norm_decay']
    params.batch_norm_epsilon = best_space['batch_norm_epsilon']
    # Adam
    params.learn_rate_adam = best_space['learn_rate_adam']
    params.beta_1 = best_space['beta_1']
    params.beta_2 = best_space['beta_2']
    params.epsilon = best_space['epsilon']
    # early stopping
    params.min_delta = best_space['min_delta']
    params.patience = best_space['patience']
    # fit
    params.batchsize = best_space['batchsize']
