import numpy as np
import time

import cls_cnn8 as cls_cnn8
import cls_cnn2 as cls_cnn2
import cls_hybrid_cnn as cls_hybrid_cnn
import cls_resnet8 as cls_resnet8
import cls_resnet2 as cls_resnet2
import cls_hybrid_resnet as cls_hybrid_resnet
from run_training import save_model
import evaluate as evl

class Classifier:

    def __init__(self, params_):
        """
        Creates a new classifier.

        Parameters
        -----------
        params_ : DataSetParams
            Parameters of the model.
        """

        self.params = params_
        if self.params.classification_model == 'cnn8':
            self.model = cls_cnn8.NeuralNet(self.params)
        elif self.params.classification_model == 'cnn2':
            self.model = cls_cnn2.NeuralNet(self.params)
        elif 'hybrid_cnn' in self.params.classification_model:
            self.model = cls_hybrid_cnn.NeuralNet(self.params)
        elif self.params.classification_model == 'resnet8':
            self.model = cls_resnet8.NeuralNet(self.params)
        elif self.params.classification_model == 'resnet2':
            self.model = cls_resnet2.NeuralNet(self.params)
        elif self.params.classification_model == 'hybrid_resnet_xgboost':
            self.model = cls_hybrid_resnet.NeuralNet(self.params)
        else:
            print('Invalid model specified')

    def save_features(self, goal, files):
        """
        Computes and saves features to disk.

        Parameters
        ----------
        goal : String
            Indicates whether the features are computed for detection or classification.
            Can be "detection", "classification" or "validation".
        files : String
            Name of the wav file used to make a prediction.
        """
        self.model.save_features(goal, files)

    def train(self, files, gt_pos, durations, gt_classes, files_valid, pos_valid, durations_valid, classes_valid,
                test_files, test_pos, test_durations, test_classes):
        '''
        Takes the file names and ground truth call positions and trains model.

        Parameters
        -----------
        files : numpy array
            Names of the wav files used to train the model.        
        gt_pos : ndarray
            Ground truth positions of the calls for each training file.
        durations : numpy array
            Durations of the wav files used to train the model. 
        gt_classes : numpy array
            Ground truth class for each training file.
        files_valid : numpy array
            Names of the wav files used to validate the model.  
        pos_valid : ndarray
            Ground truth positions of the calls for each validation file.
        durations_valid : numpy array
            Durations of the wav files used to validate the model. 
        classes_valid : numpy array
            Ground truth class of the calls for each validation file.
        test_files : numpy array
            Names of the wav files used to test the model.  
        test_pos : ndarray
            Ground truth positions of the calls for each test file.
        test_durations : numpy array
            Durations of the wav files used to test the model. 
        test_classes : numpy array
            Ground truth class of the calls for each test file.
        '''

        tic_global = time.time()
        # generate the training positions (positive and/or negative)
        # for detection and classification together or separately depending on the model
        print("Generate training positions")
        if self.params.classification_model in ['cnn8', 'hybrid_cnn_xgboost', 'resnet8', 'hybrid_resnet_xgboost']:
            positions, class_labels = generate_training_positions(files, gt_pos, gt_classes, durations, self.params, True)
        elif self.params.classification_model in ['cnn2', 'resnet2']:
            positions = {"detect":[], "classif":[]}
            class_labels = {"detect":[], "classif":[]}
            positions["detect"], class_labels["detect"] = generate_training_positions(files["detect"], gt_pos["detect"], gt_classes["detect"],
                                                                                        durations["detect"], self.params, True, goal="detection")
            positions["classif"], class_labels["classif"] = generate_training_positions(files["classif"], gt_pos["classif"], gt_classes["classif"],
                                                                                        durations["classif"], self.params, False)
        # train the model
        print("Train classifier")
        self.model.train(positions, class_labels, files, durations)

        # hard negative mining
        if self.params.num_hard_negative_mining > 0:
            print('\nhard negative mining')
            time_eval = 0
            for hn in range(self.params.num_hard_negative_mining):
                print('\thmn round', hn)
                self.params.model_identifier_detect = self.params.time + "detect_" + self.params.classification_model + "_hnm"+ str(hn)
                self.params.model_identifier_features = self.params.time + "features_" + self.params.classification_model + "_hnm"+ str(hn)
                self.params.model_identifier_classif = self.params.time + "classif_" + self.params.classification_model + "_hnm"+ str(hn)
                
                tic_eval = time.time()
                # evaluate on the validation files to compute the thresholds for each class
                nms_pos_valid, nms_prob_valid, pred_classes_valid, nb_windows_valid = self.test_batch("classification",
                                                            files_valid, durations_valid)
                threshold_classes = evl.prec_recall_1d( nms_pos_valid, nms_prob_valid, pos_valid,
                                    pred_classes_valid, classes_valid, durations_valid,
                                    self.params.detection_overlap, self.params.window_size, nb_windows_valid, True)                
                
                # evaluate on the test files with the obtained thresholds
                nms_pos, nms_prob, pred_classes, nb_windows = self.test_batch("classification", test_files, test_durations)
                evl.prec_recall_1d( nms_pos, nms_prob, test_pos, pred_classes, test_classes, 
                                    test_durations, self.params.detection_overlap, self.params.window_size, 
                                    nb_windows, False, threshold_classes=threshold_classes)
                toc_eval = time.time()
                toc_global = time.time()
                save_model(self.params.classification_model, self, self.params.model_dir, threshold_classes)

                time_eval += (toc_eval-tic_eval)
                # add training examples through hnm
                if self.params.classification_model in ['cnn8', 'hybrid_cnn_xgboost', 'resnet8', 'hybrid_resnet_xgboost']:
                    positions, class_labels = self.do_hnm_classif(files, gt_pos, gt_classes, durations, positions, class_labels, True)
                elif self.params.classification_model in ['cnn2', 'resnet2']:
                    positions["classif"], class_labels["classif"] = self.do_hnm_classif(files["classif"], gt_pos["classif"], gt_classes["classif"],
                                                                    durations["classif"], positions["classif"], class_labels["classif"], False)
                    positions["detect"], class_labels["detect"] = self.do_hnm_detect(files["detect"], gt_pos["detect"], 
                                                                    durations["detect"], positions["detect"], class_labels["detect"])
                # train the model with the new training set
                self.model.train(positions, class_labels, files, durations)
            
            # set correct model names
            self.params.model_identifier_detect = self.params.time + "detect_" + self.params.classification_model + "_hnm"+ str(self.params.num_hard_negative_mining)
            self.params.model_identifier_features = self.params.time + "features_" + self.params.classification_model + "_hnm"+ str(self.params.num_hard_negative_mining)
            self.params.model_identifier_classif = self.params.time + "classif_" + self.params.classification_model + "_hnm"+ str(self.params.num_hard_negative_mining)

        
    def test_single(self, audio_samples, sampling_rate):
        '''
        Makes a prediction on the position, probability and class of the calls present in the raw audio samples.

        Parameters
        -----------
        audio_samples : numpy array
            Data read from a wav file.
        sampling_rate : int
            Sample rate of a wav file.
        
        Returns
        --------
        nms_pos : numpy array
            Predicted positions of calls.
        nms_prob : numpy array
            Confidence level of each prediction.
        pred_classes : numpy array
            Predicted class of each prediction.
        '''
        duration = audio_samples.shape[0]/float(sampling_rate)
        nms_pos, nms_prob, pred_classes, _ = self.model.test("classification", file_duration=duration, audio_samples=audio_samples, sampling_rate=sampling_rate) # modif: renvoit aussi matches=classes
        return nms_pos, nms_prob, pred_classes

    def test_batch(self, goal, files, durations):
        """
        Makes a prediction on the position, probability and class of the calls present in the list of audio files.

        Parameters
        -----------
        goal : String
            Indicates whether the files need to be tested for detection or classification.
            Can be "detection", "classification" or "validation".
        files : numpy array
            Names of the wav files used to test the model.
        durations : numpy array
            Durations of the wav files used to test the model. 
        
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

        nms_pos = [None]*len(files)
        nms_prob = [None]*len(files)
        pred_classes = [None]*len(files)
        nb_windows = [None]*len(files)
        for ii, file_name in enumerate(files):
            nms_pos[ii], nms_prob[ii], pred_classes[ii], nb_windows[ii] = self.model.test(goal, file_name=file_name,
                                                                            file_duration=durations[ii])
        return nms_pos, nms_prob, pred_classes, nb_windows

    def do_hnm_detect(self, files, gt_pos, durations, positions, class_labels):
        """
        Hard negative mining for detection, adds high confidence false positives to the training set.

        Parameters
        -----------
        files : numpy array
            Names of the wav files used to train the model.
        gt_pos : ndarray
            Ground truth positions of the calls for each training file.
        durations : numpy array
            Durations of the wav files used to train the model.
        positions : ndarray
            Training positions containing both positive and negative examples for each training file.
        class_labels : numpy array
            Class label for each training position. 1 if positive example and 0 otherwise.

        Returns
        --------
        positions_new : ndarray
            New training positions containing both positive and negative examples for each training file.
        class_labels_new : numpy array
            New class label for each training position. 1 if positive example and 0 otherwise.
        """

        print("hnm detection")
        nms_pos, nms_prob, _, _ = self.test_batch("detection", files, durations)
        
        positions_new = [None]*len(nms_pos)
        class_labels_new = [None]*len(nms_pos)
        for ii in range(len(files)):

            # add the false positives that are above the detection threshold
            # and not too close to the GT
            new_negs = np.array([])
            if nms_pos[ii].shape[0] > 0:
                poss_negs = nms_pos[ii][nms_prob[ii][:,0] > self.params.detection_prob]
                if gt_pos[ii].shape[0] > 0:
                    # have the extra newaxis in case gt_pos[ii] shape changes in the future
                    pw_distance = np.abs(poss_negs[np.newaxis, ...]-gt_pos[ii][:,0][..., np.newaxis])
                    dis_check = (pw_distance >= (self.params.window_size / 3)).mean(0)
                    new_negs = poss_negs[dis_check==1]
                else:
                    new_negs = poss_negs
                new_negs = new_negs[new_negs < (durations[ii]-self.params.window_size)]

            # add the new examples to the training set
            positions_new[ii] = np.hstack((positions[ii], new_negs))
            if new_negs.shape[0] != 0:
                class_labels_new[ii] = np.hstack((class_labels[ii], np.zeros((new_negs.shape[0],))))
            else:
                class_labels_new[ii] = class_labels[ii]

            # sort
            sorted_inds = np.argsort(positions_new[ii])
            positions_new[ii] = positions_new[ii][sorted_inds]
            class_labels_new[ii] = class_labels_new[ii][sorted_inds]

        return positions_new, class_labels_new


    def do_hnm_classif(self, files, gt_pos, gt_classes, durations, positions, class_labels, add_neg):
        """
        Hard negative mining for classification, adds high confidence:
        - false positives of class 0
        - false negatives of class 0
        - positions of ground truth calls for which the wrong class has been predicted

        Parameters
        -----------
        files : numpy array
            Names of the wav files used to train the model.
        gt_pos : ndarray
            Ground truth positions of the calls for each training file.
        gt_classes : numpy array
            Ground truth class for each training file.
        durations : numpy array
            Durations of the wav files used to train the model.
        positions : ndarray
            Training positions containing both positive and negative examples for each training file.
        class_labels : numpy array
            Class label for each training position.
        add_neg : bool
            True if false negatives of class 0 need to be added, False otherwise

        Returns
        --------
        positions_new : ndarray
            New training positions containing both positive and negative examples for each training file.
        class_labels_new : numpy array
            New class label for each training position.
        """
        
        print("hnm classification")
        nms_pos, nms_prob, pred_classes, _ = self.test_batch("classification", files, durations)
        
        positions_new = [None]*len(nms_pos)
        class_labels_new = [None]*len(nms_pos)
        detection_overlap = self.params.window_size / 3
        shift = self.params.aug_shift / 2.5
        cnt_examples = 0

        for ii in range(len(files)):
            if nms_pos[ii].shape[0] > 0:
                poss_negs = nms_pos[ii][nms_prob[ii][:,0] > self.params.detection_prob]
                if gt_pos[ii].shape[0] > 0:
                    pos_class = gt_classes[ii][:,0]
                    # positions not matching with any gt pos = FN of class 0
                    # have the extra newaxis in case gt_pos[ii] shape changes in the future
                    pw_distance = np.abs(poss_negs[np.newaxis, ...]-gt_pos[ii][:,0][..., np.newaxis]) # lines=gt pos, col=pred pos, inside = distance btw pred and gt pos
                    dis_check = (pw_distance > detection_overlap).mean(0)
                    new_negs = poss_negs[dis_check==1] # if a predicted pos is far from all the gt pos then mean=1 because only True in dis_check

                    new_augmented_bats = []
                    new_augmented_classes = []
                    within_overlap = (pw_distance <= detection_overlap)
                    for jj in range(gt_pos[ii].shape[0]):
                        inds = np.where(within_overlap[jj,:])[0]  # get the indices of all nms pos that overlap with gt pos jj
                        if inds.shape[0] > 0: # some preds overlap with the gt
                            if (pos_class[jj] == pred_classes[ii][inds]).sum() == 0: # no prediction is of the correct species
                                new_augmented_bats = np.hstack((new_augmented_bats, gt_pos[ii][jj] - shift, gt_pos[ii][jj] + shift))
                                new_augmented_classes = np.hstack((new_augmented_classes,pos_class[jj],pos_class[jj]))
                        else: # no pred overlaps with this gt_pos
                            new_augmented_bats = np.hstack((new_augmented_bats, gt_pos[ii][jj] - shift, gt_pos[ii][jj] + shift))
                            new_augmented_classes = np.hstack((new_augmented_classes,pos_class[jj],pos_class[jj]))
                else:
                    new_negs = poss_negs
                new_negs = new_negs[new_negs < (durations[ii]-self.params.window_size)]

                # add them to the training set
                if add_neg:
                    positions_new[ii] = np.hstack((positions[ii], new_negs, new_augmented_bats))
                    if np.array(new_augmented_bats).shape[0] > 0 and new_negs.shape[0] > 0:
                        class_labels_new[ii] = np.hstack((class_labels[ii], np.zeros((new_negs.shape[0],)), new_augmented_classes))
                    elif np.array(new_augmented_bats).shape[0] == 0 and new_negs.shape[0] > 0:
                        class_labels_new[ii] = np.hstack((class_labels[ii], np.zeros((new_negs.shape[0],))))
                    elif np.array(new_augmented_bats).shape[0] > 0 and new_negs.shape[0] == 0:
                        class_labels_new[ii] = np.hstack((class_labels[ii], new_augmented_classes))
                    else:
                        class_labels_new[ii] = class_labels[ii]
                else:
                    positions_new[ii] = np.hstack((positions[ii], new_augmented_bats))
                    if np.array(new_augmented_bats).shape[0] > 0:
                        class_labels_new[ii] = np.hstack((class_labels[ii], new_augmented_classes))
                    else:
                        class_labels_new[ii] = class_labels[ii]
                
                # remove negative and above duration
                keep_inds_max = np.where(positions_new[ii]<durations[ii])
                positions_new[ii] = positions_new[ii][keep_inds_max]
                class_labels_new[ii] = class_labels_new[ii][keep_inds_max]
                keep_inds_min = np.where(positions_new[ii]>=0)
                positions_new[ii] = positions_new[ii][keep_inds_min]
                class_labels_new[ii] = class_labels_new[ii][keep_inds_min]

                # sort
                sorted_inds = np.argsort(positions_new[ii])
                positions_new[ii] = positions_new[ii][sorted_inds]
                class_labels_new[ii] = class_labels_new[ii][sorted_inds]

            else: # no nms prediction => predicted only class 0
                new_augmented_bats = []
                new_augmented_classes = []
                if gt_pos[ii].shape[0] > 0: # the gt calls were not found => augmentation
                    pos_class = gt_classes[ii][:,0]
                    for jj,gt_p in enumerate(gt_pos[ii]):
                        new_augmented_bats = np.hstack((new_augmented_bats, gt_p - shift, gt_p + shift))
                        new_augmented_classes = np.hstack((new_augmented_classes,pos_class[jj],pos_class[jj]))
                
                # add them to the training set
                positions_new[ii] = np.hstack((positions[ii], new_augmented_bats))
                class_labels_new[ii] = np.hstack((class_labels[ii], new_augmented_classes))

                # sort
                sorted_inds = np.argsort(positions_new[ii])
                positions_new[ii] = positions_new[ii][sorted_inds]
                class_labels_new[ii] = class_labels_new[ii][sorted_inds]

            class_labels_new[ii] = class_labels_new[ii].astype('int')
            cnt_examples += new_negs.shape[0] + len(new_augmented_bats)
        return positions_new, class_labels_new


def generate_training_positions(files, gt_pos, gt_classes, durations, params, add_neg, goal="classification"):
    """
    Generates the training positions based on the ground truth positions of the training files.

    Parameters
    -----------
    files : numpy array
        Names of the wav files used to train the model.
    gt_pos : ndarray
        Ground truth positions of the calls for each training file.
    durations : numpy array
        Durations of the wav files used to train the model.
    params : DataSetParams
        Parameters of the model.
    add_neg : bool
        True if negative examples need to be added, False otherwise
    goal : String
        Indicates whether the features are computed for detection or classification.
        Can be either "detection", "classification" or "valdation".

    Returns
    ---------
    positions : ndarray
        Training positions containing positive and/or negative examples for each file.
    class_labels : numpy array
        Class label for each training position.
    """

    positions = [None]*len(files)
    class_labels = [None]*len(files)
    for ii, ff in enumerate(files):
        if goal=="detection": gt_classes_ii = np.ones((gt_pos[ii].shape[0], 1))
        else: gt_classes_ii = gt_classes[ii]
        if add_neg and "multilabel" not in ff:
            positions[ii], class_labels[ii] = extract_train_position_from_file(gt_pos[ii], gt_classes_ii, durations[ii], params)
        else:
            positions[ii], class_labels[ii] = extract_train_position_from_file_without_neg_positions(gt_pos[ii], gt_classes_ii, durations[ii], params)
    return positions, class_labels


def extract_train_position_from_file(gt_pos, gt_classes, duration, params):
    """
    Data augmentation of groud truth calls and sampling of random negative locations,
    making sure not to overlap with ground truth calls.

    Parameters
    -----------
    gt_pos : ndarray
        Ground truth positions of the calls.
    gt_classes : ndarray
        Ground truth class for the call positions of each training file.
    duration : numpy array
        Duration of the wav file.
    params : DataSetParams
        Parameters of the model.
    
    Returns
    --------
    positions : ndarray
        Training positions containing positive and negative examples.
    class_labels : numpy array
        Class label for each training position.
    """
    
    if gt_pos.shape[0] == 0:
        # dont extract any values if the file does not contain anything
        # we will use these ones for HNM later
        positions = np.zeros(0)
        class_labels = np.zeros((0,1))
        print("No call in this file")
    else:
        shift = 0  # if there is augmentation this is how much we will add
        num_neg_calls = gt_pos.shape[0]
        pos_window = params.window_size / 2  # window around GT that is not sampled from
        pos = gt_pos[:, 0]
        pos_class = gt_classes[:,0]

        # augmentation
        if params.add_extra_calls:
            shift = params.aug_shift
            num_neg_calls *= 3
            pos = np.hstack((gt_pos[:, 0] - shift, gt_pos[:, 0], gt_pos[:, 0] + shift))
            pos_class = np.hstack((gt_classes[:,0],gt_classes[:,0],gt_classes[:,0]))

        # sample a set of negative locations - need to be sufficiently far away from GT
        pos_pad = np.hstack((0-params.window_size, gt_pos[:, 0], duration-params.window_size))
        neg = []
        cnt = 0
        while cnt < num_neg_calls:
            rand_pos = np.random.random()*pos_pad.max()
            if (np.abs(pos_pad - rand_pos) > (pos_window+shift)).mean() == 1:
                neg.append(rand_pos)
                cnt += 1
        neg = np.asarray(neg)

        # remove negative and above duration
        keep_inds_max = np.where(pos<duration)
        pos = pos[keep_inds_max]
        pos_class = pos_class[keep_inds_max]
        keep_inds_min = np.where(pos>=0)
        pos = pos[keep_inds_min]
        pos_class = pos_class[keep_inds_min]

        # sort them
        positions = np.hstack((pos, neg))
        sorted_inds = np.argsort(positions)
        positions = positions[sorted_inds]

        # create labels
        class_labels = np.hstack((pos_class, np.zeros((neg.shape[0],), dtype=int)))
        class_labels = class_labels[sorted_inds]

    return positions, class_labels

def extract_train_position_from_file_without_neg_positions(gt_pos, gt_classes, duration, params):
    """
    Data augmentation of ground truth calls.

    Parameters
    -----------
    gt_pos : ndarray
        Ground truth positions of the calls.
    gt_classes : ndarray
        Ground truth class for the call positions of each training file.
    params : DataSetParams
        Parameters of the model.
    
    Returns
    --------
    positions : ndarray
        Training positions containing positive examples.
    class_labels : numpy array
        Class label for each training position.
    """

    if gt_pos.shape[0] == 0:
        # dont extract any values if the file does not contain anything
        # we will use these ones for HNM later
        positions = np.zeros(0)
        class_labels = np.zeros((0,1))
    else:
        shift = 0  # if there is augmentation this is how much we will add
        pos = gt_pos[:, 0]
        pos_class = gt_classes[:,0]

        # augmentation
        if params.add_extra_calls:
            shift = params.aug_shift
            pos = np.hstack((gt_pos[:, 0] - shift, gt_pos[:, 0], gt_pos[:, 0] + shift))
            pos_class = np.hstack((gt_classes[:,0],gt_classes[:,0],gt_classes[:,0]))

        # remove negative and above duration
        keep_inds_max = np.where(pos<duration)
        pos = pos[keep_inds_max]
        pos_class = pos_class[keep_inds_max]
        keep_inds_min = np.where(pos>=0)
        pos = pos[keep_inds_min]
        pos_class = pos_class[keep_inds_min]

        # sort them
        positions = pos
        sorted_inds = np.argsort(positions)
        positions = positions[sorted_inds]

        # create labels
        class_labels = pos_class[sorted_inds]

    return positions, class_labels
