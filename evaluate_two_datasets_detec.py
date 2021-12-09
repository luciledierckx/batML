import numpy as np

def remove_end_preds(nms_pos_o, nms_prob_o, gt_pos_o, pred_classes_o, durations, win_size):
    """
    Filters out predictions and ground truth calls that are close to the end.

    Parameters
    -----------
    nms_pos_o : ndarray
        Predicted positions of calls for every file.
    nms_prob_o : ndarray
        Confidence level of each prediction for every file.
    gt_pos_o : ndarray
        Ground truth positions of the calls for every file.
    gt_classes_o : ndarray
        Ground truth class for each file.
    pred_classes_o : ndarray
        Predicted class of each prediction for every file.
    durations : numpy array
        Durations of the wav files. 
    win_size : float
        Size of a window.

    Returns
    --------
    nms_pos : ndarray
        Predicted positions of calls for every file without the ones to close to the end of the file.
    nms_prob : ndarray
        Confidence level of each prediction for every file without the ones to close to the end of the file.
    gt_pos : ndarray
        Ground truth positions of the calls for every file without the ones to close to the end of the file.
    gt_classes : ndarray
        Ground truth class for each file without the ones to close to the end of the file.
    pred_classes : ndarray
        Predicted class of each prediction for every file without the ones to close to the end of the file.        
    """
    
    nms_pos = []
    nms_prob = []
    gt_pos = []
    pred_classes = []
    for ii in range(len(nms_pos_o)):
        valid_time = durations[ii] - win_size
        gt_cur = gt_pos_o[ii]
        if gt_cur.shape[0] > 0:
            gt_pos.append(gt_cur[:, 0][gt_cur[:, 0] < valid_time][..., np.newaxis])
        else:
            gt_pos.append(gt_cur)

        if len(nms_pos_o[ii]) > 0:
            valid_preds = nms_pos_o[ii] < valid_time
            nms_pos.append(nms_pos_o[ii][valid_preds])
            nms_prob.append(nms_prob_o[ii][valid_preds, 0][..., np.newaxis])
            pred_classes.append(pred_classes_o[ii][valid_preds])
        else:
            nms_pos.append(nms_pos_o[ii])
            nms_prob.append(nms_prob_o[ii][..., np.newaxis])
            pred_classes.append(pred_classes_o[ii])
    return nms_pos, nms_prob, gt_pos, pred_classes


def prec_recall_1d(nms_pos_o_detect, nms_prob_o_detect, gt_pos_o_detect, pred_classes_o_detect, gt_classes_detect, 
                    durations_detect, nb_windows_detect, nms_pos_o_Nata, nms_prob_o_Nata, gt_pos_o_Nata, pred_classes_o_Nata, 
                    gt_classes_Nata, durations_Nata, nb_windows_Nata, detection_overlap, win_size, filename, 
                    tuning, threshold_classes=np.zeros(8), remove_eof=True):
    """
    Computes the best thresholds based on the two datasets
    or saves the performance for detection, classification using the given thresholds.

    Parameters
    -----------
    nms_pos_o_detect : ndarray
        Predicted positions of calls for every file of the detection dataset.
    nms_prob_o_detect : ndarray
        Confidence level of each prediction for every file of the detection dataset.
    gt_pos_o_detect : ndarray
        Ground truth positions of the calls for every file of the detection dataset.
    pred_classes_o_detect : ndarray
        Predicted class of each prediction for every file of the detection dataset.
    gt_classes_detect : ndarray
        Ground truth class for each file of the detection dataset.
    durations_detect : numpy array
        Durations of the wav files of the detection dataset.
    nb_windows_detect : ndarray
        Number of windows for every test file of the detection dataset.
    nms_pos_o_Nata : ndarray
        Predicted positions of calls for every file of the classification dataset.
    nms_prob_o_Nata : ndarray
        Confidence level of each prediction for every file of the classification dataset.
    gt_pos_o_Nata : ndarray
        Ground truth positions of the calls for every file of the classification dataset.
    pred_classes_o_Nata : ndarray
        Predicted class of each prediction for every file of the classification dataset.
    gt_classes_Nata : ndarray
        Ground truth class for each file of the classification dataset.
    durations_Nata : numpy array
        Durations of the wav files of the classification dataset.
    nb_windows_Nata : ndarray
        Number of windows for every test file of the classification dataset.
    detection_overlap : float
        Maximum distance between a prediction and a ground truth to be considered as overlapping. 
    win_size : float
        Size of a window.
    filename : String
        Name of the file in which the performance will be saved.
    tuning : bool
        True if the thresholds need to be tuned, False otherwise.
    threshold_classes : numpy array
        Thresholds used to evaluate the performance of the model if tuning is set to False,
        not used otherwise.
    remove_eof : bool
        True if the predictions and ground truth calls that are close to the end should be filtered out.
    
    Returns
    --------
    best_threshold_classes : numpy array
        Thresholds of each class giving the best F1 score.
    """

    if remove_eof:
        # filter out the detections in both ground truth and predictions that are too
        # close to the end of the file - dont count them during eval
        nms_pos_detect, nms_prob_detect, gt_pos_detect, pred_classes_detect = remove_end_preds(nms_pos_o_detect, nms_prob_o_detect, gt_pos_o_detect, pred_classes_o_detect, durations_detect, win_size)
        nms_pos_Nata, nms_prob_Nata, gt_pos_Nata, pred_classes_Nata = remove_end_preds(nms_pos_o_Nata, nms_prob_o_Nata, gt_pos_o_Nata, pred_classes_o_Nata, durations_Nata, win_size)

    else:
        nms_pos_detect = nms_pos_o_detect
        nms_prob_detect = nms_prob_o_detect
        gt_pos_detect = gt_pos_o_detect
        pred_classes_detect = pred_classes_o_detect
        nms_pos_Nata = nms_pos_o_Nata
        nms_prob_Nata = nms_prob_o_Nata
        gt_pos_Nata = gt_pos_o_Nata
        pred_classes_Nata = pred_classes_o_Nata
    
    # compute the performance using the given thresholds
    if not tuning:
        F1_global_classes = 0
        with open(filename,'a')as f: f.write("-------- DETECTION performance on DETECTION dataset --------")
        print("-------- DETECTION performance on DETECTION dataset --------")
        compute_conf_matrices(nms_pos_detect, nms_prob_detect, gt_pos_detect, pred_classes_detect, gt_classes_detect, durations_detect,
                                    detection_overlap, nb_windows_detect, filename, tuning, threshold_classes, "detect")
        with open(filename,'a')as f: f.write("-------- DETECTION AND CLASSIFICATION performance on MULTI-LABEL dataset --------")
        print("-------- DETECTION AND CLASSIFICATION performance on MULTI-LABEL dataset --------")
        compute_conf_matrices(nms_pos_Nata, nms_prob_Nata, gt_pos_Nata, pred_classes_Nata, gt_classes_Nata, durations_Nata,
                                    detection_overlap, nb_windows_Nata, filename, tuning, threshold_classes, "all")
        best_F1 = F1_global_classes
        best_threshold_classes = threshold_classes
    
    # tune the thresholds by incrementing them from 0 to 100% until no improvement is made 
    # and choosing the ones giving the best average F1 score between the two datasets
    else:
        best_F1 = 0
        best_threshold_classes = np.zeros(8)
        change = True
        while change:
            print("while change")
            change = False
            for clss in range(1,8):
                for i in range(0,101):
                    threshold_classes = best_threshold_classes.copy()
                    threshold_classes[clss] = i
                    current_F1_detect = compute_conf_matrices(nms_pos_detect, nms_prob_detect, gt_pos_detect, pred_classes_detect, gt_classes_detect, durations_detect,
                                            detection_overlap, nb_windows_detect, filename, tuning, threshold_classes, "detect")
                    current_F1_Nata = compute_conf_matrices(nms_pos_Nata, nms_prob_Nata, gt_pos_Nata, pred_classes_Nata, gt_classes_Nata, durations_Nata,
                                            detection_overlap, nb_windows_Nata, filename, tuning, threshold_classes, "all")
                    if (current_F1_detect+current_F1_Nata)/2>best_F1:
                        change = True
                        best_threshold_classes[clss] = threshold_classes[clss]
                        best_F1 = (current_F1_detect+current_F1_Nata)/2
                    
        
    with open(filename,'a') as f:
        f.write("Best F1 = "+ str(best_F1))
        f.write("Best thresholds = "+ str(best_threshold_classes))
    print("Best F1=",best_F1)
    print("Best thresholds =",best_threshold_classes)

    return best_threshold_classes


def compute_conf_matrices(nms_pos, nms_prob, gt_pos, pred_classes, gt_classes, durations, detection_overlap,
                        nb_windows, filename, tuning, threshold_classes, type):
    """
    Computes and saves the performance for detection and classification.

    Parameters
    -----------
    nms_pos : ndarray
        Predicted positions of calls for every file.
    nms_prob : ndarray
        Confidence level of each prediction for every file.
    gt_pos : ndarray
        Ground truth positions of the calls for every file.
    pred_classes : ndarray
        Predicted class of each prediction for every file.
    gt_classes : ndarray
        Ground truth class for each file.
    durations : numpy array
        Durations of the wav files.
    detection_overlap : float
        Maximum distance between a prediction and a ground truth to be considered as overlapping. 
    nb_windows : ndarray
        Number of windows for every test file.
    filename : String
        Name of the file in which the performance will be saved.
    tuning : bool
        True if the thresholds need to be tuned, False otherwise.
    threshold_classes : numpy array
        Thresholds used to evaluate the performance of the model.
    type : String
        can take the value "all" or "detect", respectively when the F1 score is computed on detection and classification, 
        and when it is only computed for the detection.
    
    Returns
    --------
    F1_threshold : numpy array
        Array containing the global F1 score and the F1 scores of each class.
    """
    
    conf_matrix = np.zeros((8,2,2), dtype=int)
    conf_matrix_detect = np.zeros((2,2), dtype=int)
    conf_matrix_classif = np.zeros((7,2,2), dtype=int)
    cl_num = np.arange(0,8,1)

    # loop through each file
    for ii in range(len(nms_pos)):

        # check to make sure the file contains some predictions
        num_preds = nms_pos[ii].shape[0]
        if num_preds > 0:
            distance_to_gt = np.abs(gt_pos[ii].ravel()-nms_pos[ii].ravel()[:, np.newaxis])
            within_overlap = (distance_to_gt <= detection_overlap)
            # lines=pred pos, col=gt pos, inside=true if distance btw pred and gt pos is <= detection overlap

            # True if the gt_pos overlaps with a predicted call having the correct class
            gt_found_correct = np.array([False] * gt_pos[ii].shape[0], dtype=bool)
            # True if the gt_pos overlaps with a predicted call but not of the correct class
            gt_found_incorrect = np.array([False] * gt_pos[ii].shape[0], dtype=bool)
            # (p,c) is True if the gt_pos p overlaps with an nms_pos of class c and c is not the correct class for p
            gt_incorrect_classes = np.zeros((gt_pos[ii].shape[0],8),dtype=bool)

            # loop on the predictions
            for jj in range(num_preds):
                if nms_prob[ii][jj] > (threshold_classes[pred_classes[ii][jj]]/100):
                    # get the indices of all gt pos that overlap with pred pos jj
                    inds = np.where(within_overlap[jj,:])[0]
                    # some gt overlap with the preds
                    if inds.shape[0] > 0:
                        # correct timing but not correct species
                        if (gt_classes[ii][inds] == pred_classes[ii][jj]).sum() == 0: # correct timing but not correct species
                            conf_matrix[pred_classes[ii][jj]][1][0] += inds.shape[0]
                            conf_matrix_classif[pred_classes[ii][jj]-1][1][0] += inds.shape[0]
                            gt_found_incorrect[inds] = True
                            gt_incorrect_classes[inds,pred_classes[ii][jj]] = True
                        # correct timing and correct species
                        else: 
                            for i_overlap in inds: # one pred can overlap with several gt pos
                                # do not add to conf matrix if the gt pos was already overlapped by another pred pos
                                if gt_classes[ii][i_overlap][0]==pred_classes[ii][jj] and not gt_found_correct[i_overlap]:
                                    conf_matrix[gt_classes[ii][i_overlap],0,0] += 1
                                    conf_matrix_detect[0][0] += 1
                                    conf_matrix_classif[gt_classes[ii][i_overlap]-1,0,0] += 1
                                    gt_found_correct[i_overlap] = True

                    # a bat call is predicted while there is no call
                    else:
                        conf_matrix[0][0][1] += 1
                        conf_matrix[pred_classes[ii][jj]][1][0] += 1
                        conf_matrix[np.delete(cl_num, pred_classes[ii][jj])[1:], 1,1] += 1
                        conf_matrix_detect[1][0] += 1

            for i_gt in range(len(gt_found_correct)):
                # gt pos that were not overlapped by any pred
                if (not gt_found_correct[i_gt]) and (not gt_found_incorrect[i_gt]):
                    conf_matrix[gt_classes[ii][i_gt],0,1] += 1
                    conf_matrix[np.delete(cl_num, gt_classes[ii][i_gt])[1:], 1,1] += 1
                    conf_matrix[0][1][0] += 1
                    conf_matrix_detect[0][1] += 1
                # gt pos that was overlapped but never with the correct species
                # misclassification is counted only once even when multiple incorrect species were predicted
                if (not gt_found_correct[i_gt]) and gt_found_incorrect[i_gt]:
                    conf_matrix[gt_classes[ii][i_gt], 0,1] += 1
                    conf_matrix_classif[gt_classes[ii][i_gt]-1,0,1] += 1
                    gt_incorrect_classes[i_gt][gt_classes[ii][i_gt][0]] = True
                    not_tn_classes = np.where(gt_incorrect_classes[i_gt])
                    conf_matrix[np.delete(cl_num,not_tn_classes), 1, 1] += 1
                    conf_matrix_detect[0][0] += 1
                    conf_matrix_classif[np.delete(cl_num, not_tn_classes)[1:]-1, 1, 1] += 1
                # when a call overlaps with several predictions and among those predictions one is of the correct class
                # then add a TN to all classes that are not part of this set
                if gt_found_correct[i_gt]:
                    gt_incorrect_classes[i_gt][gt_classes[ii][i_gt][0]] = True
                    not_tn_classes = np.where(gt_incorrect_classes[i_gt])
                    conf_matrix[np.delete(cl_num, not_tn_classes), 1, 1] += 1
                    conf_matrix_classif[np.delete(cl_num, not_tn_classes)[1:]-1, 1,1] += 1

        # no calls predicted so for all gt pos we wrongly predicted that there is no call
        else:
            unique, frequency = np.unique(gt_classes[ii], return_counts=True)
            unique = unique.astype('int')
            frequency = frequency.astype('int')
            conf_matrix[unique,0,1] += frequency
            conf_matrix[np.delete(cl_num, unique)[1:], 1,1] += len(gt_pos[ii])
            conf_matrix[0][1][0] += len(gt_pos[ii])
            conf_matrix_detect[0][1] += len(gt_pos[ii])

    # add to the conf matrix the TP of class 0 for the current file
    nms_pos_ratio = np.divide(nms_pos,durations)
    nms_pos_inds = np.multiply(nms_pos_ratio,np.array(nb_windows))
    nms_pos_inds = np.array([np.floor(nms_pos_inds[i]) for i in range(nms_pos_inds.shape[0])])
    gt_pos_ratio = np.divide(gt_pos,durations)
    gt_pos_inds = np.multiply(gt_pos_ratio,np.array(nb_windows))
    gt_pos_inds = np.array([np.floor(gt_pos_inds[i]) for i in range(gt_pos_inds.shape[0])])
    # nb_tp_0 is an approximation because for the rest of the evaluation we consider the calls and not the windows
    nb_tp_0 = sum(nb_windows)
    for i in range(gt_pos_inds.shape[0]):
        all_inds = set(np.concatenate((nms_pos_inds[i],gt_pos_inds[i].ravel())))
        nb_tp_0 -= len(all_inds)
    conf_matrix[0][0][0] = nb_tp_0
    conf_matrix[cl_num[1:], 1,1] += nb_tp_0
    conf_matrix_detect[1][1] = nb_tp_0

    F1_threshold = compute_perf('detect', conf_matrix_detect, filename, tuning, threshold_classes)
    if type == "all": 
        F1_threshold += compute_perf('classif', conf_matrix_classif, filename, tuning, threshold_classes)
        F1_threshold /= 2
    return F1_threshold

def compute_perf(perf_type, conf_matrix, filename, tuning, threshold_classes):
    """
    Computes the performance based on the confusion matrix.
    Saves and displays them only if tuning is False.

    Parameters
    -----------
    perf_type : String
        Can be 'detect' or 'classif' in function of the given confusion matrix.
    conf_matrix : numpy array
        Confusion matrix of the model.
    filename : String
        Name of the file in which the performance will be saved.
    tuning : bool
        True if the thresholds need to be tuned, False otherwise.
    threshold_classes : numpy array
        Thresholds used to evaluate the performance of the model.
    
    Returns
    --------
    F1_threshold : numpy array
        Array containing the global F1 score.
    """

    with open(filename,'a') as f:
        # print and save the thresholds and the confusion matrix
        if not tuning:
            f.write('\n=====================================================\n')
            f.write('Threshold = '+str(threshold_classes)+'\n')
            f.write('\n=====================================================\n')
            f.write('\n--------------\n')
            f.write('Confusion matrix '+perf_type+'\n')
            f.write('--------------\n')
            f.write(''+str(conf_matrix)+'\n')
            print('\n=====================================================\n')
            print('Threshold = '+str(threshold_classes)+'\n')
            print('\n=====================================================\n')

        # compute the metrics
        if perf_type=="detect":
            TP = np.array([conf_matrix[0][0]])
            FP = np.array([conf_matrix[1][0]])
            FN = np.array([conf_matrix[0][1]])
            TN = np.array([conf_matrix[1][1]])
            nb_iter = 1
        else:
            TP = conf_matrix[:,0,0]
            FP = conf_matrix[:,1,0]
            FN = conf_matrix[:,0,1]
            TN = conf_matrix[:,1,1]
            nb_iter = conf_matrix.shape[0]
        PRE = TP/(TP+FP).astype(float)
        REC = TP/(TP+FN).astype(float)
        F1 = 2*(PRE*REC)/(PRE+REC)

        # print and save the metrics
        if not tuning:
            for i in range(nb_iter):
                print('--------------')
                if perf_type == 'classif': print('Class', i+1, perf_type)
                elif perf_type == 'detect': print("GLOBAL", perf_type)
                else: print('Class', i, perf_type)
                print('--------------')
                print('Precision', PRE[i])
                print('Recall', REC[i])
                print('F1',F1[i])
                print( )

                f.write('--------------\n')
                if perf_type == 'classif': f.write('Class '+ str((i+1))+' '+ str(perf_type)+'\n')
                elif perf_type == 'detect': f.write("GLOBAL "+ str(perf_type)+'\n')
                else: f.write('Class '+ str(i)+' ' + str(perf_type)+'\n')
                f.write('--------------\n')
                f.write('True Positive '+ str(TP[i])+'\n')
                f.write('False Positive '+ str(FP[i])+'\n')
                f.write('False Negative '+ str(FN[i])+'\n')
                f.write('True Negative '+ str(TN[i])+'\n')
                f.write('Precision '+str(PRE[i])+'\n')
                f.write('Recall '+str(REC[i])+'\n')
                f.write('F1 '+str(F1[i])+'\n')
            if perf_type != "detect":
                print('--------------')
                print('GLOBAL', perf_type)
                print('--------------')
                print('Average Precision', np.mean(PRE))
                print('Average Recall', np.mean(REC))
                print('Average F1', np.mean(F1))
            f.write('--------------\n')
            f.write('GLOBAL '+perf_type+'\n')
            f.write('--------------\n')
            f.write('Average Precision '+str(np.mean(PRE))+'\n')
            f.write('Average Recall '+str(np.mean(REC))+'\n')
            f.write('Average F1 '+str(np.mean(F1))+'\n')
        
    
    return np.mean(F1)
