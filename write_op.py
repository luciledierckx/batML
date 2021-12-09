import pandas as pd
import numpy as np
import datetime as dt


def save_to_txt(op_file, results):
    """
    Takes a list of dictionaries of results and saves them to file.

    Parameters
    -----------
    op_file : String
        Path to the file in which the results will be saved.
    results : list
        Contains dictionaries with each the four following fields: filename, time, prob, pred_classes.
    """

    with open(op_file, 'w') as file:
        head_str = 'file_name,predicted_time,predicted_species,predicted_prob'
        file.write(head_str + '\n')
        for ii in range(len(results)):
            for jj in range(len(results[ii]['prob'])):
                row_str = results[ii]['filename'] + ','
                tm = round(results[ii]['time'][jj],3)
                pr = round(results[ii]['prob'][jj],3)
                sp = results[ii]['pred_classes'][jj]
                row_str += str(tm) + ',' +str(sp) + ',' + str(pr)
                file.write(row_str + '\n')


def create_audio_tagger_op(ip_file_name, op_file_name, st_times,
                           class_pred, class_prob, samp_rate, class_names):
    """
    Saves the detections in an audiotagger friendly format.

    Parameters
    -----------
    ip_file_name : String
        Name of the wav file.
    op_file_name : String
        Name of the csv in which the results are saved.
    st_times : numpy array
        Starting times of the predicted calls.
    class_pred : list
        Classes of the predicted calls.
    class_prob : numpy array
        Confidence levels of the predicted calls.
    samp_rate : float
        Sampling rate of the file.
    class_names : list
        Maps the class number to the class name.
    
    """

    col_names = ['Filename', 'Label', 'LabelTimeStamp', 'Spec_NStep',
                 'Spec_NWin', 'Spec_x1', 'Spec_y1', 'Spec_x2', 'Spec_y2',
                 'LabelStartTime_Seconds', 'LabelEndTime_Seconds',
                 'LabelArea_DataPoints', 'ClassifierConfidence']

    nstep = 0.001
    nwin = 0.003
    call_width = 0.001  # code does not output call width so just put in dummy value (batdetective)
    y_max = (samp_rate*nwin)/2.0
    num_calls = len(st_times)

    if num_calls == 0:
        da_at = pd.DataFrame(index=np.arange(0), columns=col_names)
    else:
        da_at = pd.DataFrame(index=np.arange(0, num_calls), columns=col_names)
        da_at['Spec_NStep'] = nstep
        da_at['Spec_NWin'] = nwin
        da_at['Label'] = 'bat'
        da_at['LabelTimeStamp'] = dt.datetime.now().isoformat()
        da_at['Spec_y1'] = 0
        da_at['Spec_y2'] = y_max
        da_at['Filename'] = ip_file_name

        for ii in np.arange(0, num_calls):
            st_time = st_times[ii]
            da_at.loc[ii, 'LabelStartTime_Seconds'] = st_time
            da_at.loc[ii, 'LabelEndTime_Seconds'] = st_time + call_width
            da_at.loc[ii, 'Label'] = class_names[class_pred[ii]]
            da_at.loc[ii, 'Spec_x1'] = st_time/nstep
            da_at.loc[ii, 'Spec_x2'] = (st_time + call_width)/nstep
            da_at.loc[ii, 'ClassifierConfidence'] = round(class_prob[ii], 3)

    # save to disk
    da_at.to_csv(op_file_name, index=False)
