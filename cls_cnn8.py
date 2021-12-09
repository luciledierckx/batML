import numpy as np
from scipy.ndimage.filters import gaussian_filter1d
from scipy.io import wavfile
import pyximport; pyximport.install()
from os import path
import time

from spectrogram import compute_features_spectrogram
import nms as nms
from cnn_helper import network_fit, tune_network


class NeuralNet:

    def __init__(self, params_):
        """
        Creates a new CNN with 8 classes to detect and classify.

        Parameters
        -----------
        params_ : DataSetParams
            Parameters of the model.
        """
        self.params = params_
        self.network_classif = None
        self.nb_error = 0

    def train(self, positions, class_labels, files, durations):
        """
        Takes the file names and ground truth call positions and trains model.

        Parameters
        -----------
        positions : ndarray
            Training positions for each training file.
        class_labels : numpy array
            Class label for each training position. 1 if positive example and 0 otherwise.
        files : numpy array
            Names of the wav files used to train the model.        
        durations : numpy array
            Durations of the wav files used to train the model.
        """

        # compute or load the features of the training files
        print("Compute or load features")
        tic = time.time()
        features, labels, labels_not_merged = self.features_labels_from_file(positions, class_labels, files, durations)
        toc = time.time()
        self.params.features_computation_time += toc-tic
        
        # tuning of the hyperparameters of the CNN
        if self.params.tune_cnn_8:
            print("Tune cls_cnn8")
            tic_cnn_8 = time.time()
            tune_network(self.params, features, labels, labels_not_merged, self.params.trials_filename_1)
            toc_cnn_8 = time.time()
            while toc_cnn_8-tic_cnn_8 < self.params.tune_time:
                tune_network(self.params, features, labels, labels_not_merged, self.params.trials_filename_1)
                toc_cnn_8 = time.time()
            print('total tuning time', round(toc_cnn_8-tic_cnn_8, 3), '(secs) =', round((toc_cnn_8-tic_cnn_8)/60,2), r"min \\")
        
        # fit the CNN
        print("Fit cls_cnn8")
        self.network_classif, _ = network_fit(self.params, features, labels, labels_not_merged, 8)
    
    def features_labels_from_file(self, positions, class_labels, files, durations):
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

        Returns
        --------
        features : ndarray
            Array containing the spectrogram features for each training position of the audio files.
        labels : ndarray
            Class labels in one-hot encoding for each training position of the audio files.
        labels_not_merged : ndarray
            Array containing one class label per call instead of per position in one-hot encoding.
            (Used to compute the class weights.)
        """
        
        feats = []
        labels = np.array([])
        labels_not_merged = np.array([], dtype=int)
        nb_inds_no_dup = 0
        for i, file_name in enumerate(files):
            if positions[i].shape[0] > 0:
                local_feats = self.create_or_load_features("classification", file_name)

                # convert time in file to integer
                positions_ratio = positions[i] / durations[i]
                train_inds = (positions_ratio*float(local_feats.shape[0])).astype('int')

                # one-hot encoding of the class labels
                local_class = np.zeros((class_labels[i].size, 8), dtype=int)
                rows = np.arange(class_labels[i].size)
                local_class[rows, class_labels[i]] = 1
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

        # flatten list of lists
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

        # compute features and perform classification
        tic = time.time()
        features = self.create_or_load_features(goal, file_name, audio_samples, sampling_rate)
        toc=time.time()
        self.params.features_computation_time += toc-tic
        features = features.reshape(features.shape[0], features.shape[2], features.shape[3], 1)
        tic = time.time()
        y_predictions = self.network_classif.predict(features)
        toc=time.time()
        self.params.classif_time += toc - tic

        # smooth the output prediction per column so smooth each class prediction over time
        tic = time.time()
        if self.params.smooth_op_prediction:
            y_predictions = gaussian_filter1d(y_predictions, self.params.smooth_op_prediction_sigma, axis=0)

        call_predictions_not_bat = y_predictions[:,0]
        pos_bat = []
        prob_bat = []
        pred_classes_bat = []

        # perform non max suppression for each class
        for i in range(1,8):
            call_predictions_bat = y_predictions[:,i]
            classes = np.repeat(i, call_predictions_bat.shape[0])
            pos, prob, classes, call_predictions_not_bat_nms = nms.nms_1d(call_predictions_bat.astype(np.float),
                                    classes, call_predictions_not_bat, self.params.nms_win_size, file_duration)

            # remove pred that have a higher probability of not being a bat
            for i in range(len(pos)):
                if prob[i][0]>call_predictions_not_bat_nms[i]:
                    pos_bat.append(pos[i])
                    prob_bat.append(prob[i])
                    pred_classes_bat.append(classes[i])
        
        # sort according to position in file
        sorted_inds = np.argsort(pos_bat)
        pos_bat = np.array(pos_bat)[sorted_inds]
        prob_bat = np.array(prob_bat)[sorted_inds]
        pred_classes_bat = np.array(pred_classes_bat)[sorted_inds]
        
        toc=time.time()
        self.params.nms_computation_time += toc-tic

        nms_pos = np.array(pos_bat)
        nms_prob = np.array(prob_bat)
        pred_classes = np.array(pred_classes_bat)
        nb_windows = features.shape[0]
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
            data_set = self.params.data_set_detect
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
