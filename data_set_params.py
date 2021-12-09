import numpy as np
import csv
from models_params_helper import cnn_params, cnn_params_1, cnn_params_2, xgboost_params, resnet_params, resnet_params_1, resnet_params_2


class DataSetParams:

    def __init__(self, model_name=None):
        """
        Create an object containing all the needed parameters.

        Parameters
        -----------
        model_name : String
            Can be one of: 'cnn8', 'cnn2', 'hybrid_cnn_xgboost',
            'resnet8', 'resnet2', 'hybrid_resnet_xgboost'
        """

        self.spectrogram_params()
        self.detection()

        self.classification_model = model_name

        # path to the directories and datasets
        self.audio_dir_detect = '' # path to the directory containing the wav files for detection
        self.audio_dir_classif = '' # path to the directory containing the wav files for classification
        self.data_set_detect = '' # path to the npz file for detection
        self.data_set_classif = '' # path to the npz file for classification
        self.model_dir = 'data/models/' # path to the saved models
        self.feature_dir = '' # path to the saved features

        # gain of time by computng and saving the features only once and then loading them
        self.save_features_to_file = False # True to compute and save the features
        self.load_features_from_file = True # True to load the features instead of computing them

        self.num_hard_negative_mining = 0

        # non max suppression - smoothing and window
        self.smooth_op_prediction = True  # smooth the op parameters before nms
        self.smooth_op_prediction_sigma = 0.006 / self.time_per_slice
        self.nms_win_size = int(np.round(0.12 / self.time_per_slice))  #ie 21 samples at 0.02322 fft win size, 0.75 overlap

        # tuning hyperopt
        self.tune_cnn_8 = False
        self.tune_cnn_2 = False
        self.tune_cnn_7 = False
        self.tune_xgboost_spectrogram = False
        self.tune_resnet_8 = False
        self.tune_resnet_2 = False
        self.tune_resnet_7 = False
        self.tune_time = 17*60*60 # maximum time of tuning in seconds
        self.filename_tuning()
        
        # CNN
        self.num_epochs = 200
        self.restore_best_weights = True
        self.validation_split = 0.1
        self.net_type = 'params'

        # XGBoost
        self.objective = 'multi:softprob'
        self.eval_metric = 'mlogloss'

        # the time needed for nms, features computation, detection and classification
        # will be determinded during the program
        self.nms_computation_time = 0
        self.features_computation_time = 0
        self.detect_time = 0
        self.classif_time = 0
        
        # misc
        self.add_extra_calls = True  # sample some other positive calls near the GT (= ground truth)
        self.aug_shift = 0.015  # unit seconds, add extra call either side of GT if augmenting


    def filename_tuning(self):
        """
        Sets the name of the tuning file corresponding to the chosen model.
        """

        self.trials_filename_1 = "results/"
        self.trials_filename_2 = "results/"
        if self.classification_model == "cnn8":
            self.trials_filename_1 += "trials_"+self.classification_model
        elif self.classification_model == "cnn2":
            self.trials_filename_1 += "trials_"+self.classification_model+"_1"
            self.trials_filename_2 += "trials_"+self.classification_model+"_2"
        elif self.classification_model == "hybrid_cnn_xgboost":
            self.trials_filename_1 += "trials_hybrid_cnn_spectrogram"
            self.trials_filename_2 += "trials_hybrid_xgboost_spectrogram"
        elif self.classification_model == "resnet8":
            self.trials_filename_1 += "trials_"+self.classification_model
        elif self.classification_model == "resnet2":
            self.trials_filename_1 += "trials_"+self.classification_model+"_1"
            self.trials_filename_2 += "trials_"+self.classification_model+"_2"
        elif self.classification_model == "hybrid_resnet_xgboost":
            self.trials_filename_1 += "trials_hybrid_resnet_spectrogram"
            self.trials_filename_2 += "trials_hybrid_xgboost_spectrogram"

    
    def load_params_from_csv(self, model_name):
        """
        Reads the csv file corresponding to the chosen model and sets the parameters.

        Parameters
        -----------
        model_name : String
            Can be one of: 'cnn8', 'cnn2', 'hybrid_cnn_xgboost',
            'resnet8', 'resnet2', 'hybrid_resnet_xgboost'
        """
        
        filename_cnn = "data/cnn_params.csv"
        filename_xgboost = "data/xgboost_params.csv"
        filename_resnet = "data/resnet_params.csv"
        dict_cnn = self.load_params(filename_cnn)
        dict_xgboost = self.load_params(filename_xgboost)
        dict_resnet = self.load_params(filename_resnet)
        if model_name=="cnn8":
            cnn_params(self, dict_cnn['cnn_8'])
        elif model_name == "cnn2":
            cnn_params_1(self, dict_cnn['cnn_2'])
            cnn_params_2(self, dict_cnn['cnn_7'])
        elif model_name == "hybrid_cnn_xgboost":
            cnn_params(self, dict_cnn['cnn_8'])
            xgboost_params(self, dict_xgboost['xgboost_spectrogram'])
        elif model_name == "resnet8":
            resnet_params(self, dict_resnet['resnet_8'])
        elif model_name == "resnet2":
            resnet_params_1(self, dict_resnet['resnet_2'])
            resnet_params_2(self, dict_resnet['resnet_7'])
        elif model_name == "hybrid_resnet_xgboost":
            resnet_params(self, dict_resnet['resnet_8'])
            xgboost_params(self, dict_xgboost['xgboost_spectrogram'])
        else:
            print("Error while loading csv for model parameters")
            
    def load_params(self, filename):
        """
        Puts the information of the file in a dictionary. Each key is a model and
        its associated value is a dictionary of the corresponding parameter names and values.

        Parameters
        -----------
        filename : String
            Name of the file to read.
        
        Returns
        --------
        dic : dict
            Each key is a model and its associated value is a dictionary
            of the corresponding parameter names and values.
        """
        dic = {}
        with open(filename, 'r', encoding='utf-8-sig') as data: 
            for line in csv.DictReader(data):
                dic[line['model']] = {k: line[k] for k in line.keys() - {'model'}}
        return dic
    
    def spectrogram_params(self):
        """
            Sets the parameters related to the spectrograms.
        """

        self.valid_file_length = 169345  # some files are longer than they should be

        # spectrogram generation
        self.fft_win_length = 0.02322  # ie 1024/44100.0 about 23 msecs.
        self.fft_overlap = 0.75  # this is a percent - previously was 768/1024
        self.time_per_slice = ((1-self.fft_overlap)*self.fft_win_length)

        self.denoise = True
        self.mean_log_mag = 0.5  # sensitive to the spectrogram scaling used
        self.smooth_spec = True  # gaussian filter

        # throw away unnecessary frequencies, keep from bottom
        # TODO this only makes sense as a frequency when you know the sampling rate
        # better to think of these as indices
        self.crop_spec = True
        self.max_freq = 270
        self.min_freq = 10

        # if doing 192K files for training
        #self.fft_win_length = 0.02667  # i.e. 512/19200
        #self.max_freq = 240
        #self.min_freq = 10

    def detection(self):
        """
        Sets the parameters related to the detection.
        """
        self.window_size = 0.230  # 230 milliseconds (in time expanded, so 23 ms for not)
        # represent window size in terms of the number of time bins
        self.window_width = np.rint(self.window_size / ((1-self.fft_overlap)*self.fft_win_length))
        self.detection_overlap = 0.1  # needs to be within x seconds of GT to be considered correct
        self.detection_prob = 0.5  # everything under this is considered background - used in HNM

    