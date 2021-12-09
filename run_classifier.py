from scipy.io import wavfile
import numpy as np
import os
import glob
import time
import tensorflow as tf
import json
from tensorflow.keras.models import Model, load_model
import joblib

import write_op as wo
import classifier as clss
from data_set_params import DataSetParams

def read_audio(file_name, do_time_expansion, chunk_size, win_size):
    """
    Reads the audio file and apply time expansion if needed.

    Parameters
    -----------
    file_name : String
        Name of the audio file.
    do_time_expansion : bool
        True if time expansion need to be applied on the audio file and False otherwise.
    chunk_size : float
        Size of an audio chunk.
    win_size : float
        Size of a window.

    Returns
    --------
    read_fail : bool
        True if an error occurred while reading the file and False otherwised.
    audiopad : numpy array
        Audio samples padded with zeroes so the calls are not too close to the end of the file.
    file_dur : float
        Duration of the file.
    samp_rate : float
        Sampling rate of the file after a potential time expansion.
    samp_rate_orig : float
        Original sampling rate of the file.
    """

    # try to read in audio file
    try:
        samp_rate_orig, audio = wavfile.read(file_name)
    except:
        print('  Error reading file: ', file_name)
        return True, None, None, None, None

    # convert to mono if stereo
    if len(audio.shape) == 2:
        print('  Warning: stereo file. Just taking right channel.')
        audio = audio[:, 1]
    file_dur = audio.shape[0] / float(samp_rate_orig)

    # original model is trained on time expanded data
    samp_rate = samp_rate_orig
    if do_time_expansion:
        samp_rate = int(samp_rate_orig/10.0)
        file_dur *= 10

    # pad with zeros so we can go right to the end
    multiplier = np.ceil(file_dur/float(chunk_size-win_size))
    diff = multiplier*(chunk_size-win_size) - file_dur + win_size
    audio_pad = np.hstack((audio, np.zeros(int(diff*samp_rate))))

    read_fail = False
    return read_fail, audio_pad, file_dur, samp_rate, samp_rate_orig


def run_classifier(model, audio, file_dur, samp_rate, threshold_classes, chunk_size):
    """
    Uses the model to predict the time, class and confidence level of bat calls in the file.

    Parameters
    -----------
    model : Classifier
        Model used to detect and classify.
    audio : numpy array
        Audio samples of the file.
    file_dur : float
        Duration of the file.
    samp_rate : float
        Sampling rate of the file.
    threshold_classes : numpy array
        Thresholds above which the confidence level needs to be to consider the prediction as a call.
        There is one threshold per class.
    chunk_size : float
        Size of an audio chunk.
    
    Returns
    --------
    call_time : numpy array
        Positions where calls are predicted in the file.
    call_prob : numpy array
        Confidence level of the predicted calls.
    call_class : list
        Classes of the predicted calls.
    """

    call_time = []
    call_prob = []
    call_class = []
    test_time = []

    # files can be long so we split each up into separate (overlapping) chunks
    st_positions = np.arange(0, file_dur, chunk_size-model.params.window_size)
    for chunk_id, st_position in enumerate(st_positions):

        # take a chunk of the audio
        st_pos = int(st_position*samp_rate)
        en_pos = int(st_pos + chunk_size*samp_rate)
        audio_chunk = audio[st_pos:en_pos]

        # make predictions
        tic = time.time()
        pos, prob, classes = model.test_single(audio_chunk, samp_rate)
        toc = time.time()
        test_time.append(round(toc-tic, 3))

        if pos.shape[0] > 0:
            prob = prob[:, 0]

            # remove predictions near the end (if not last chunk) and ones that are
            # below the detection threshold
            if chunk_id == (len(st_positions)-1):
                inds = (prob >= threshold_classes[classes])
            else:
                inds = (prob >= threshold_classes[classes]) & (pos < (chunk_size-(model.params.window_size/2.0)))

            # keep valid detections and convert detection time back into global time
            if pos.shape[0] > 0:
                call_time.append(pos[inds] + st_position)
                call_prob.append(prob[inds])
                call_class.append(classes[inds])

    if len(call_time) > 0:
        call_time = np.hstack(call_time)
        call_prob = np.hstack(call_prob)

        # undo the effects of times expansion
        if do_time_expansion:
            call_time /= 10.0
    
    return call_time, call_prob, call_class


if __name__ == "__main__":
    """
    This code takes a directory of audio files and runs a model to perform bat call detection and classification.
    It returns in a csv file the time of the detection, the species of the calls
    and the confidence level of the predicted species.
    """
    
    ####################################
    # Parameters to be set by the user #
    ####################################
    on_GPU = True   # True if tensorflow runs on GPU, False otherwise
    do_time_expansion = True  # set to True if audio is not already time expanded
    save_res = True    # True to save the results in a csv file and False otherwise
    load_features_from_file = False 
    data_dir = '' # path of the directory containing the audio files
    result_dir = 'results/'    # path to the directory where the results are saved
    model_dir = 'data/models/'  # path to the saved models
    model_name = "hybrid_cnn_xgboost" # one of: 'cnn8', 'cnn2', 'hybrid_cnn_xgboost'
    #'resnet8', 'resnet2', "hybrid_resnet_xgboost"

    chunk_size = 4.0    # The size of an audio chunk

    # name of the result file
    classification_result_file = result_dir + 'classification_result.csv'
    if not os.path.isdir(result_dir):
        os.makedirs(result_dir)

    if on_GPU:
        # needed to run tensorflow on GPU
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        session = tf.compat.v1.InteractiveSession(config=config)
    else:
        # needed to run tensorflow on CPU
        config = tf.compat.v1.ConfigProto(device_count = {'GPU': 0})
        tf.config.set_visible_devices([], 'GPU')
        session = tf.compat.v1.InteractiveSession(config=config)

    # model name and load models
    if model_name == "cnn8":
        date = "25_05_21_12_12_25_"
        hnm = "1"
        model_file_classif = model_dir + date + "classif_" + model_name + "_hnm" + hnm
        network_classif = load_model(model_file_classif + '_model')
    elif model_name == "cnn2":
        date = "25_05_21_15_09_28_"
        hnm = "0"
        model_file_detect = model_dir + date + "detect_" + model_name + "_hnm" + hnm
        network_detect = load_model(model_file_detect + '_model')
        model_file_classif = model_dir + date + "classif_" + model_name + "_hnm" + hnm
        network_classif = load_model(model_file_classif + '_model')
    elif model_name == "hybrid_cnn_xgboost":
        date = "02_06_21_09_32_04_"
        hnm = "2"
        model_file_features = model_dir + date + "features_" + model_name + "_hnm" + hnm
        network_features = load_model(model_file_features + '_model')
        network_feat = Model(inputs=network_features.input, outputs=network_features.layers[-3].output)
        model_file_classif = model_dir + date + "classif_" + model_name + "_hnm" + hnm
        network_classif = joblib.load(model_file_classif + '_model.pkl')
    elif model_name == "resnet8":
        date = "04_11_21_13_53_01_"
        hnm = "1"
        model_file_classif = model_dir + date + "classif_" + model_name + "_hnm" + hnm
        network_classif = load_model(model_file_classif + '_model')
    elif model_name == "resnet2":
        date = "02_11_21_11_44_38_"
        hnm = "0"
        model_file_detect = model_dir + date + "detect_" + model_name + "_hnm" + hnm
        network_detect = load_model(model_file_detect + '_model')
        model_file_classif = model_dir + date + "classif_" + model_name + "_hnm" + hnm
        network_classif = load_model(model_file_classif + '_model')
    elif model_name == "hybrid_resnet_xgboost":
        date = "03_11_21_12_01_42_"
        hnm = "0"
        model_file_features = model_dir + date + "features_" + model_name + "_hnm" + hnm
        network_features = load_model(model_file_features + '_model')
        network_feat = Model(inputs=network_features.input, outputs=network_features.layers[-2].output)
        model_file_classif = model_dir + date + "classif_" + model_name + "_hnm" + hnm
        network_classif = joblib.load(model_file_classif + '_model.pkl')
    
    # load params
    with open(model_file_classif + '_params.p') as f:
        parameters = json.load(f)
    print("params=", parameters)

    # array with group name according to class number
    group_names = ['not call', 'Barbarg', 'Envsp', 'Myosp', 'Pip35','Pip50', 'Plesp', 'Rhisp']

    # model classifier
    params = DataSetParams(model_name)
    params.window_size = parameters['win_size']
    params.max_freq = parameters['max_freq']
    params.min_freq = parameters['min_freq']
    params.mean_log_mag = parameters['mean_log_mag']
    params.fft_win_length = parameters['slice_scale']
    params.fft_overlap = parameters['overlap']
    params.crop_spec = parameters['crop_spec']
    params.denoise = parameters['denoise']
    params.smooth_spec = parameters['smooth_spec']
    params.nms_win_size = parameters['nms_win_size']
    params.smooth_op_prediction_sigma = parameters['smooth_op_prediction_sigma']
    if model_name in ["hybrid_cnn_xgboost"]: params.n_estimators = parameters["n_estimators"]
    params.load_features_from_file = load_features_from_file
    params.detect_time = 0
    params.classif_time = 0
    model_cls = clss.Classifier(params)
    if model_name in  ["cnn8", "cnn2", "hybrid_cnn_xgboost","resnet8", "resnet2", "hybrid_resnet_xgboost"]:
        model_cls.model.network_classif = network_classif
    if model_name in ["cnn2", "resnet2"]:
        model_cls.model.network_detect = network_detect
    if model_name in ["hybrid_cnn_xgboost", "hybrid_resnet_xgboost"]:
        model_cls.model.network_features = network_features
        model_cls.model.model_feat = network_feat
    
    # load thresholds
    threshold_classes = np.load(model_file_classif + '_thresholds.npy')
    threshold_classes = threshold_classes / 100

    print("model name =", model_name)
    results = []
    # load audio file names and loop through them
    audio_files = glob.glob(data_dir + '*.wav')
    for file_cnt, file_name in enumerate(audio_files):
        print("------------",file_name,"--------------")
        file_name_root = file_name[len(data_dir):]

        # read audio file - skip file if cannot read
        read_fail, audio, file_dur, samp_rate, samp_rate_orig = read_audio(file_name,
                                do_time_expansion, chunk_size, model_cls.params.window_size)
        if read_fail:
            continue
        if file_dur>4:
            # run classifier
            tic = time.time()
            call_time, call_prob, call_classes = run_classifier(model_cls, audio, file_dur, samp_rate, threshold_classes, chunk_size)
            toc = time.time()
            print("total time = ",toc-tic)
            num_calls = len(call_time)
            if num_calls>0:
                call_classes = np.concatenate(np.array(call_classes)).ravel()
                call_species = [group_names[i] for i in call_classes]
                print("call pos=",call_time)
                print("call species=", call_species)
                print("call proba=",call_prob)
            print('  ' + str(num_calls) + ' calls found')

            # save results
            if save_res:
                # save to AudioTagger format
                op_file_name = result_dir + file_name_root[:-4] + '-sceneRect.csv'
                wo.create_audio_tagger_op(file_name_root, op_file_name, call_time,
                                        call_classes, call_prob,
                                        samp_rate_orig, group_names)

                # save as dictionary
                if num_calls > 0:
                    res = {'filename':file_name_root, 'time':call_time,
                        'prob':call_prob, 'pred_classes':call_species}
                    results.append(res)

    # save to large csv
    if save_res and (len(results) > 0):
        print('\nsaving results to', classification_result_file)
        wo.save_to_txt(classification_result_file, results)
    else:
        print('no detections to save')
