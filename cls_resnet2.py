import numpy as np
from scipy.ndimage.filters import gaussian_filter1d
from scipy.io import wavfile
import pyximport; pyximport.install()
from os import path
import time
import pickle

from resnet_helper import obj_func_cnn, network_fit
import nms_cnn2 as nms
from spectrogram import compute_features_spectrogram
from hyperopt import hp, tpe, fmin, space_eval, Trials


class NeuralNet:

    def __init__(self, params_):
        """
        Creates a resnet for detection and a resnet for classification.

        Parameters
        -----------
        params_ : DataSetParams
            Parameters of the model.
        """
        self.params = params_
        self.network_detect = None
        self.network_classif = None

    def train(self, positions, class_labels, files, durations):
        """
        Takes the file names and ground truth call positions and trains model.

        Parameters
        -----------
        positions : ndarray
            Training positions for each training file.
        class_labels : numpy array
            Class label for each training position.
        files : numpy array
            Names of the wav files used to train the model.        
        durations : numpy array
            Durations of the wav files used to train the model.
        """

        # compute or load the features of the training files and the associated class label.
        print("Compute or load features")
        tic = time.time()
        features_detect, labels_detect, _ = self.features_labels_from_file(positions["detect"], class_labels["detect"], files["detect"],
                                                                        durations["detect"], "detection")
        features_classif, labels_classif, labels_not_merged_classif = self.features_labels_from_file(positions["classif"], class_labels["classif"], files["classif"],
                                                                        durations["classif"], "classification")
        toc = time.time()
        self.params.features_computation_time += toc-tic

        # tuning of the hyperparameters of the two CNNs
        if self.params.tune_resnet_2:
            print("Tune resnet detect")
            tic_resnet_2 = time.time()
            best_space_detect = self.tune_network(features_detect, labels_detect, labels_detect, self.params.trials_filename_1, goal="detection")
            toc_resnet_2 = time.time()
            while toc_resnet_2-tic_resnet_2 < self.params.tune_time:
                best_space_detect = self.tune_network(features_detect, labels_detect, labels_detect, self.params.trials_filename_1, goal="detection")
                toc_resnet_2 = time.time()
            print('total tuning time', round(toc_resnet_2-tic_resnet_2, 3), '(secs) =', round((toc_resnet_2-tic_resnet_2)/60,2), r"min \\")
        if self.params.tune_resnet_7:
            print("Tune resnet classif")
            tic_resnet_7 = time.time()
            best_space_classif = self.tune_network(features_classif, labels_classif, labels_not_merged_classif, self.params.trials_filename_2, goal="classification")
            toc_resnet_7 = time.time()
            while toc_resnet_7-tic_resnet_7 < self.params.tune_time:
                best_space_classif = self.tune_network(features_classif, labels_classif, labels_not_merged_classif, self.params.trials_filename_2, goal="classification")
                toc_resnet_7 = time.time()
            print('total tuning time', round(toc_resnet_7-tic_resnet_7, 3), '(secs) =', round((toc_resnet_7-tic_resnet_7)/60,2), r"min \\")
        
        # fit the two resnets
        self.network_detect, _ = network_fit(self.params, features_detect, labels_detect,  labels_detect, 2, '_1')
        self.network_classif, _ = network_fit(self.params, features_classif, labels_classif, labels_not_merged_classif, 7, '_2')

        if self.params.tune_resnet_2:
            print("best_space_detect =", best_space_detect)
        if self.params.tune_resnet_7:
            print("best_space_classif =", best_space_classif)


    def tune_network(self, features, labels, labels_not_merged, trials_filename, goal):
        """
        Tunes the network with hyperopt.

        Parameters
        -----------
        features : ndarray
            Array containing the spectrogram features for each training window of the audio file.
        labels : ndarray
            Class labels in one-hot encoding for each training window of the audio files.
        labels_not_merged : ndarray
            Array containing one class label per call instead of per position in one-hot encoding.
            (Used to compute the class weights.)
        trials_filename : String
            Name of the file where the previous iterations of hyperopt are saved.
        goal : String
            Indicates whether the network needs to be tuned for detection or classification.
            Can be either "detection" or "classification".
        
        Returns
        --------
        best_space : dict
            Best hyperparameters found so far for the CNN.
        """
        
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
                    'nb_output': (2 if goal=="detection" else 7)
                    }
                
        # load the saved trials
        try:
            trials = pickle.load(open(trials_filename+".hyperopt", "rb"))
            max_trials = len(trials.trials) + 1
        # create a new trials
        except:
            max_trials = 1
            trials = Trials()

        # optimise the objective function with the defined set of CNN parameters
        best_space_indices = fmin(obj_func_cnn, space_resnet, trials=trials,algo=tpe.suggest, max_evals=max_trials)
        best_space = space_eval(space_resnet, best_space_indices)
        best_space = {k: best_space[k] for k in best_space.keys() - {'features', 'labels', 'labels_not_merged'}}
        with open(trials_filename + ".hyperopt", "wb") as f:
            pickle.dump(trials, f)

        nb_cnn = (1 if goal=="detection" else 2)
        # resnet
        setattr(self.params, "L2_weight_decay"+str(nb_cnn), best_space['L2_weight_decay'])
        setattr(self.params, "batch_norm_decay"+str(nb_cnn), best_space['batch_norm_decay'])
        setattr(self.params, "batch_norm_epsilon"+str(nb_cnn), best_space['batch_norm_epsilon'])
        # Adam
        setattr(self.params, "learn_rate_adam_"+str(nb_cnn), best_space['learn_rate_adam'])
        setattr(self.params, "beta_1_"+str(nb_cnn), best_space['beta_1'])
        setattr(self.params, "beta_2_"+str(nb_cnn), best_space['beta_2'])
        setattr(self.params, "epsilon_"+str(nb_cnn), best_space['epsilon'])
        # early stopping
        setattr(self.params, "min_delta_"+str(nb_cnn), best_space['min_delta'])
        setattr(self.params, "patience_"+str(nb_cnn), best_space['patience'])
        # fit
        setattr(self.params, "batchsize_"+str(nb_cnn), best_space['batchsize'])

        return best_space

    
    def features_labels_from_file(self, positions, class_labels, files, durations, goal):
        """
        Computes or loads the features of each position of the files
        and indicates the associated class label.

        Parameters
        -----------
        positions : ndarray
            Training positions for each file.
        class_labels : numpy array
            Class label for each position.
        files : numpy array
            Names of the wav files.        
        durations : numpy array
            Durations of the wav files.
        goal : String
            Indicates whether the network needs to be tuned for detection or classification.
            Can be either "detection" or "classification".

        Returns
        --------
        features : ndarray
            Array containing the spectrogram features for each position of the audio files.
        labels : ndarray
            Class labels in one-hot encoding for each training position of the audio files.
        """

        feats = []
        labels = np.array([])
        labels_not_merged = np.array([], dtype=int)
        nb_inds_no_dup = 0
        for i, file_name in enumerate(files):
            if positions[i].shape[0] > 0:
                local_feats = self.create_or_load_features(goal, file_name)

                # convert time in file to integer
                positions_ratio = positions[i] / durations[i]
                train_inds = (positions_ratio*float(local_feats.shape[0])).astype('int')

                if goal=="detection":
                    feats.append(local_feats[train_inds, :, :, :])
                    labels = np.concatenate((labels,class_labels[i]))
                elif goal == "classification":
                    # one-hot encoding of the class labels
                    local_class = np.zeros((class_labels[i].size, 7), dtype=int)
                    rows = np.arange(class_labels[i].size)
                    local_class[rows, class_labels[i]-1] = 1

                    train_inds_no_dup = []

                    # combine call pos that are in the same window and merge their labels
                    for pos_ind, win_ind  in enumerate(train_inds):
                        # if the pos to add is in a new window then add it
                        if pos_ind==0 or train_inds_no_dup[-1]!=win_ind:
                            train_inds_no_dup.append(win_ind)
                            if pos_ind==0 and labels.shape[0]==0: labels = np.array([local_class[pos_ind]])
                            else: labels = np.concatenate((labels,np.array([local_class[pos_ind]])), axis=0)
                        else:
                            index_one = np.where(local_class[pos_ind]==1)[0][0]
                            # if the pos to add is in the same window but it is a new class then combine the labels
                            # with all entries of the same window
                            if labels[-1][index_one]!=1:
                                same_win_ind = np.where(train_inds_no_dup==win_ind)[0] + nb_inds_no_dup
                                labels[same_win_ind] = np.logical_or(labels[same_win_ind],local_class[pos_ind]).astype('int')
                            # if the pos to add is in the same window and it is not a new class then add it
                            # only if it is the first class that was observed for that window (to generate duplicates)
                            elif labels[-1].sum() == 1:
                                train_inds_no_dup.append(win_ind)
                                labels = np.concatenate((labels,np.array([local_class[pos_ind]])), axis=0)
                                
                    feats.append(local_feats[train_inds_no_dup, :, :, :])
                    if labels_not_merged.shape[0] == 0: labels_not_merged = local_class
                    else: labels_not_merged = np.vstack((labels_not_merged, local_class))
                    nb_inds_no_dup += len(train_inds_no_dup)
        
        if goal=="detection": labels = labels.astype(np.uint8)
        features = np.vstack(feats)
        return features, labels, labels_not_merged
    
    def test(self, goal, file_name=None, file_duration=None, audio_samples=None, sampling_rate=None):
        """
        Makes a prediction on the position, probability and class of the calls present in an audio file.
        
        Parameters
        -----------
        goal : String
            Indicates whether the features are used for detection or classification
            or more specifically for validation.
            Can be either "detection", "classification" or "validation".
        file_name : String
            Name of the wav file used to make a prediction.
        file_duration : float
            Duration of the wav file used to make a prediction.
        audio_samples : numpy array
            Data read from wav file.
        sampling_rate : int
            Sample rate of wav file.

        Returns
        --------
        nms_pos : ndarray
            Predicted positions of calls for every test file.
        nms_prob : ndarray
            Confidence level of each prediction for every test file.
        pred_classes : ndarray
            Predicted class of each prediction for every test file.
        nb_windows : ndarray
            Number of windows for every test file.
        """

        # compute features and perform detection
        tic = time.time()
        features = self.create_or_load_features(goal, file_name, audio_samples, sampling_rate)
        toc=time.time()
        self.params.features_computation_time += toc-tic
        features = features.reshape(features.shape[0], features.shape[2], features.shape[3], 1)
        nb_windows = features.shape[0]
        tic = time.time()
        y_predictions_detect = self.network_detect.predict(features)
        toc=time.time()
        self.params.detect_time += toc - tic

        # smooth the output prediction per column so smooth each class prediction over time
        tic = time.time()
        if self.params.smooth_op_prediction:
            y_predictions_detect = gaussian_filter1d(y_predictions_detect, self.params.smooth_op_prediction_sigma, axis=0)
        
        # trying to get rid of rows with 0 highest
        call_predictions_bat = y_predictions_detect[:,1:]
        call_predictions_not_bat = y_predictions_detect[:,0]
        high_preds = np.array([np.max(x) for x in call_predictions_bat])[:, np.newaxis] 
        pred_classes = np.array([np.argmax(x)+1 for x in call_predictions_bat])[:, np.newaxis]
        
        # perform non max suppression
        pos, prob, pred_classes, call_predictions_not_bat, features = nms.nms_1d(high_preds[:,0].astype(np.float), pred_classes, 
                                                                    call_predictions_not_bat, features, self.params.nms_win_size, file_duration)
        
        # remove pred that have a higher probability of not being a bat
        pos_bat = []
        prob_bat = []
        pred_classes_bat = []
        features_bat = []
        for i in range(len(pos)):
            if prob[i][0]>call_predictions_not_bat[i]:
                pos_bat.append(pos[i])
                prob_bat.append(prob[i])
                pred_classes_bat.append(pred_classes[i])
                features_bat.append(features[i])
        toc=time.time()
        self.params.nms_computation_time += toc-tic

        # perform classification
        tic = time.time()
        pred_proba = np.array([])
        pred_classes = np.array([])
        if np.array(features_bat).shape[0] != 0:
            y_predictions_classif = self.network_classif.predict(np.array(features_bat))
            pred_proba = y_predictions_classif.flatten('F')[..., np.newaxis]
            pred_classes = np.repeat(np.arange(1,8,1),len(pos_bat))
        toc=time.time()
        self.params.classif_time += toc - tic

        nms_pos = np.array(pos_bat*7)
        nms_prob = pred_proba
        return nms_pos, nms_prob, pred_classes, nb_windows



    def create_or_load_features(self, goal, file_name=None, audio_samples=None, sampling_rate=None):
        """
        Does 1 of 3 possible things
        1) computes feature from audio samples directly
        2) loads feature from disk OR
        3) computes features from file name

        Parameters
        -----------
        goal : String
            Indicates whether the features are used for detection or classification
            or more specifically for validation.
            Can be either "detection", "classification" or "validation".
        file_name : String
            Name of the wav file used to make a prediction.
        audio_samples : numpy array
            Data read from wav file.
        sampling_rate : int
            Sample rate of wav file.

        Returns
        --------
        features : ndarray
            Array containing the spectrogram features for each window of the audio file.
        """

        if goal == "detection":
            audio_dir = self.params.audio_dir_detect
            data_set = self.params.data_set_classif if "multilabel" in file_name else self.params.data_set_detect
        elif goal =="classification":
            audio_dir = self.params.audio_dir_classif
            data_set = self.params.data_set_classif
        elif goal =="validation":
            audio_dir = self.params.audio_dir_valid
            data_set = self.params.data_set_valid

        # 1) computes feature from audio samples directly
        if file_name is None:
            features = compute_features_spectrogram(audio_samples, sampling_rate, self.params)
        else:
            # 2) loads feature from disk
            if self.params.load_features_from_file and path.exists(self.params.feature_dir + data_set + '_' + file_name.split("/")[-1] + '_spectrogram' + '.npy'):
                features = np.load(self.params.feature_dir + data_set + '_' + file_name.split("/")[-1] + '_spectrogram' + '.npy')
            # 3) computes features from file name
            else:
                if self.params.load_features_from_file: print("missing features have to be computed")
                sampling_rate, audio_samples = wavfile.read(audio_dir + file_name.split("/")[-1]  + '.wav')
                features = compute_features_spectrogram(audio_samples, sampling_rate, self.params)
                if self.params.save_features_to_file or self.params.load_features_from_file:
                    np.save(self.params.feature_dir + data_set + '_' + file_name.split("/")[-1] + '_spectrogram', features)
        return features

    def save_features(self, goal, files):
        """
        Computes and saves features to disk.

        Parameters
        ----------
        goal : String
            Indicates whether the features are used for detection or classification
            or more specifically for validation.
            Can be either "detection", "classification" or "validation".
        files : String
            Name of the wav file used to make a prediction.
        """
        
        if goal == "detection":
            audio_dir = self.params.audio_dir_detect
            data_set = self.params.data_set_detect
        elif goal =="classification":
            audio_dir = self.params.audio_dir_classif
            data_set = self.params.data_set_classif
        elif goal =="validation":
            audio_dir = self.params.audio_dir_valid
            data_set = self.params.data_set_valid

        for file_name in files:
            sampling_rate, audio_samples = wavfile.read(audio_dir + file_name.split("/")[-1] + '.wav')
            features = compute_features_spectrogram(audio_samples, sampling_rate, self.params)
            np.save(self.params.feature_dir + data_set + '_' + file_name.split("/")[-1] + '_spectrogram', features)
