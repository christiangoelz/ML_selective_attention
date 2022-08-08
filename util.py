"""

Author: Christian Goelz <c.goelz@gmx.de>

Created: 24th February, 2022
"""

from pathlib import Path 
import glob 
import numpy as np

import mne 
from mne.decoding import Vectorizer
from mne.preprocessing.xdawn import _XdawnTransformer
mne.set_log_level('critical')
from mne.preprocessing import ICA
from autoreject import AutoReject

from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import RandomUnderSampler

import pyreadstat

import warnings
warnings.filterwarnings("ignore")
# GET FILE LIST FROM LOCATION
#if str(Path.cwd()).split('/')[-2] == 'lifespan':
#    input = Path.cwd().parent / 'input' / '*.*df'
#elif str(Path.cwd()).split('/')[-1] == 'lifespan': 
#    input = Path.cwd() / 'input' / '*.*df'

#FILES = glob.glob(str(input))
FILES = glob.glob('/home/christian/Schreibtisch/lifespan/input/EEG_Daten_Flanker/*.*df')
METADATA = glob.glob('/home/christian/Schreibtisch/lifespan/input/EEG_Daten_Flanker/stats/*.sav')[0]
METADATA, _ = pyreadstat.read_sav(METADATA)
METADATA['part'] = METADATA["VPN_file"].str.lower()
METADATA.set_index('part', inplace = True)
METADATA = METADATA[['GRUPPE','Alter','Geschlecht','eegusedNEW']]

OUTPATH = str(Path.cwd().parent / 'results')


def read_file(f, preload = True):
    """Reads in EEG files in edf or bdf format. Returns MNE raw object and name of file
    Parameters
    ----------
    f : string
        string with file path to edf/ bdf file 

    Returns
    -------
    name: string
        name of file
    raw: object
        MNE raw object
    """
    name = f.split('/')[-1][:-4]
    if f[-3:] == 'edf':
        raw = mne.io.read_raw_edf(f, eog=['EXG1', 'EXG2', 'EXG3', 'EXG4'], 
                                  exclude = ['EXG7', 'EXG8'], misc=['EXG5', 'EXG6'],
                                  preload=preload)
    else:
        raw = mne.io.read_raw_bdf(f, eog=['EXG1', 'EXG2', 'EXG3', 'EXG4'], 
                                  exclude = ['EXG7', 'EXG8'],misc=['EXG5', 'EXG6'],
                                  preload=preload)
    return name.lower(), raw 

def correct_events(raw):
    """ Function to correct the event coding of the Events given by Presentation:
    Ignores the evaluation done online during the experiment and reavaluates it. 
    The experimental setup together with the event coding is visualized in resources/setup.pdf
    Correct: correct response between 100ms and 1200ms and response button pressed 
            only once

    Parameters
    ----------
    raw : mne raw object
        raw EEG data structure of MNE python

    Returns
    -------
    events: array, shape = (n_events,3)
        corrected events 
    mapping: dict, 
        mapping between event id and label
    response_time: list of dicts, 
        a list containing a dict for each correct trial 
        - dict with key representing the task label and value the response time-
    accuracy: dict 
        the accuracy per stimulus 
    """
    response_time = []
    labels = [16,32,64]
    buttons = [1,2]
    mapping = {201: 'start_C_correct',
               202: 'start_N_correct',
               203: 'start_IC_correct',
               204: 'start_C_wrong',
               205: 'start_N_wrong',
               206: 'start_IC_wrong',
               101: 're_C_correct',
               102: 're_N_correct',
               103: 're_IC_correct',
               104: 're_C_wrong',
               105: 're_N_wrong',
               106: 're_IC_wrong'}
               
    events = mne.find_events(raw, shortest_event=1)
    trial_start_ids = np.where(events[:,2] == 8)[0]

    for start_id in trial_start_ids: 
    
        # get corresponding label
        stim_id = start_id-1
        while events[stim_id,2] not in range(41,47):
            stim_id -= 1
            if stim_id < 0:
                stim_id = None 
                break
            elif events[stim_id,2] >= 201: # reach last event
                stim_id = None 
                break

        # get button press 
        button_id = start_id+1  
        while events[button_id,2] not in buttons: 
            # end of experiment
            button_id += 1
            if button_id >= len(events)-1:
                button_id = None
                break
            elif events[button_id,2] == 8:
                button_id = None
                break

        # re-evaluate
        # case button pressed
        if button_id is not None and stim_id is not None: 
            stim = events[stim_id,2]
            button = events[button_id,2]
            rt = ((events[button_id,0] - events[start_id,0]) / raw.info['sfreq']) * 1000 # response time in ms

            # evaluate
            if (stim in [41,44,46] and button == 2) or (stim in [42,43,45] and button == 1): 
                correct = True
                
                # evaluate if there is a second press: 
                id = button_id
                while events[id,2] != 8:
                    id += 1
                    if id >= len(events):
                        break
                    elif events[id,2] in buttons and events[id,2] != events[button_id,2]:
                        correct = False
                        #print("double pressed other")
            else: 
                correct = False 
            if rt < 100 or rt > 1200:
                correct = False
                #print("to slow")

            # rewrite coding
            if correct: 
                # Congruent correct
                if stim in [41,42]: 
                    events[start_id, 2] = 201
                    events[button_id, 2] = 101
                    response_time.append({'task':'C','rt':rt})
                # Neutral correct
                elif stim in [43,44]:
                    events[start_id, 2] = 202
                    events[button_id, 2] = 102
                    response_time.append({'task':'N','rt':rt})
                # Incongruent correct
                else: 
                    events[start_id, 2] = 203
                    events[button_id, 2] = 103
                    response_time.append({'task':'IC','rt':rt})
            else:
                # Congruent wrong
                if stim in [41,42]: 
                    events[start_id, 2] = 204
                    events[button_id, 2] = 104
                # Neutral wrong
                elif stim in [43,44]:
                    events[start_id, 2] = 205
                    events[button_id, 2] = 105
                # Incongruent wrong
                else: 
                    events[start_id, 2] = 206
                    events[button_id, 2] = 106 

        # case no button pressed
        elif stim_id is not None: 
            stim = events[stim_id,2]
            # Congruent wrong
            if stim in [41,42]: 
                events[start_id, 2] = 204
            # Neutral wrong
            elif stim in [43,44]:
                events[start_id, 2] = 205
            # Incongruent wrong
            else: 
                events[start_id, 2] = 206
        
    # calculate accuracy 
    accuracy = []
    c = 201 # code correct (see mapping)
    w = 204 # code wrong answer (see mapping)
    for task in ['C','N','IC']:
        corrects = len(np.where(events[:,2] == c)[0])
        all = len(np.where((events[:,2] == c) | (events[:,2] == w))[0])
        acc = (corrects / all) * 100 
        accuracy.append({'task':task, '#corrects': corrects, 'all': all, 'acc':acc})
        c += 1
        w += 1

    return events, mapping, response_time, accuracy

def correct_events_per_color(raw):
    """ Function to correct the event coding of the Events given by Presentation:
    Ignores the evaluation done online during the experiment and reavaluates it. 
    The experimental setup together with the event coding is visualized in resources/setup.pdf
    Correct: correct response between 100ms and 1200ms and response button pressed 
            only once

    Parameters
    ----------
    raw : mne raw object
        raw EEG data structure of MNE python

    Returns
    -------
    events: array, shape = (n_events,3)
        corrected events 
    mapping: dict, 
        mapping between event id and label
    response_time: list of dicts, 
        a list containing a dict for each correct trial 
        - dict with key representing the task label and value the response time-
    accuracy: dict 
        the accuracy per stimulus 
    """
    response_time = []
    labels = [16,32,64]
    buttons = [1,2]
    mapping = {201: 'start_C_correct_red',
               202: 'start_N_correct_red',
               203: 'start_IC_correct_red',
               301: 'start_C_correct_green',
               302: 'start_N_correct_green',
               303: 'start_IC_correct_green',
               204: 'start_C_wrong',
               205: 'start_N_wrong',
               206: 'start_IC_wrong',
               101: 're_C_correct',
               102: 're_N_correct',
               103: 're_IC_correct',
               104: 're_C_wrong',
               105: 're_N_wrong',
               106: 're_IC_wrong'}
               
    events = mne.find_events(raw, shortest_event=1)
    trial_start_ids = np.where(events[:,2] == 8)[0]

    for start_id in trial_start_ids: 
    
        # get corresponding label
        stim_id = start_id-1
        while events[stim_id,2] not in range(41,47):
            stim_id -= 1
            if stim_id < 0:
                stim_id = None 
                break
            elif events[stim_id,2] >= 201: # reach last event
                stim_id = None 
                break

        # get button press 
        button_id = start_id+1  
        while events[button_id,2] not in buttons: 
            # end of experiment
            button_id += 1
            if button_id >= len(events)-1:
                button_id = None
                break
            elif events[button_id,2] == 8:
                button_id = None
                break

        # re-evaluate
        # case button pressed
        if button_id is not None and stim_id is not None: 
            stim = events[stim_id,2]
            button = events[button_id,2]
            rt = ((events[button_id,0] - events[start_id,0]) / raw.info['sfreq']) * 1000 # response time in ms

            # evaluate
            if (stim in [41,44,46] and button == 2) or (stim in [42,43,45] and button == 1): 
                correct = True
                
                # evaluate if there is a second press: 
                id = button_id
                while events[id,2] != 8:
                    id += 1
                    if id >= len(events):
                        break
                    elif events[id,2] in buttons and events[id,2] != events[button_id,2]:
                        correct = False
                        #print("double pressed other")
            else: 
                correct = False 
            if rt < 100 or rt > 1200:
                correct = False
                #print("to slow")

            # rewrite coding
            if correct: 
                # Congruent correct red
                if stim == 41: 
                    events[start_id, 2] = 201
                    events[button_id, 2] = 101
                    response_time.append({'task':'C','rt':rt})
                # Neutral correct red
                elif stim == 44:
                    events[start_id, 2] = 202
                    events[button_id, 2] = 102
                    response_time.append({'task':'N','rt':rt})
                # Incongruent correct red
                elif stim == 46: 
                    events[start_id, 2] = 203
                    events[button_id, 2] = 103
                    response_time.append({'task':'IC','rt':rt})
                # Congruent correct green
                elif stim == 42: 
                    events[start_id, 2] = 301
                    events[button_id, 2] = 101
                    response_time.append({'task':'C','rt':rt})
                # Neutral correct green
                elif stim == 43:
                    events[start_id, 2] = 202
                    events[button_id, 2] = 102
                    response_time.append({'task':'N','rt':rt})
                # Incongruent correct green
                elif stim == 45: 
                    events[start_id, 2] = 203
                    events[button_id, 2] = 103
                    response_time.append({'task':'IC','rt':rt})
            else:
                # Congruent wrong
                if stim in [41,42]: 
                    events[start_id, 2] = 204
                    events[button_id, 2] = 104
                # Neutral wrong
                elif stim in [43,44]:
                    events[start_id, 2] = 205
                    events[button_id, 2] = 105
                # Incongruent wrong
                else: 
                    events[start_id, 2] = 206
                    events[button_id, 2] = 106 

        # case no button pressed
        elif stim_id is not None: 
            stim = events[stim_id,2]
            # Congruent wrong
            if stim in [41,42]: 
                events[start_id, 2] = 204
            # Neutral wrong
            elif stim in [43,44]:
                events[start_id, 2] = 205
            # Incongruent wrong
            else: 
                events[start_id, 2] = 206
        
    # calculate accuracy 
    accuracy = []
    c = 201 # code correct (see mapping)
    w = 204 # code wrong answer (see mapping)
    for task in ['C','N','IC']:
        corrects = len(np.where(events[:,2] == c)[0])
        all = len(np.where((events[:,2] == c) | (events[:,2] == w))[0])
        acc = (corrects / all) * 100 
        accuracy.append({'task':task, '#corrects': corrects, 'all': all, 'acc':acc})
        c += 1
        w += 1

    return events, mapping, response_time, accuracy

def prepare_data(f, param):
    try:
        subj, raw = read_file(f)
        print(f'Preprocessing participant: {subj}')
        try:
            raw = raw.drop_channels(['GSR1', 'GSR2', 'Erg1', 'Erg2', 'Resp', 'Plet', 'Temp'])
        except:
            pass

        event_id = {'C_correct':201, 'IC_correct':203}  
        ref_ch = ['EXG5','EXG6']
        raw.set_eeg_reference(ref_ch)
        events, _, _, _ = correct_events(raw)
        raw.filter(param['filt_low'], param['filt_high'], fir_design="firwin")

        picks = mne.pick_types(raw.info, eeg=True, stim=False, eog=False, exclude="bads")
        raw, events = raw.resample(param['sfreq'], events=events)
        raw.set_montage('standard_1020')

        if param['ica']: 
            ica = ICA(method='fastica', random_state=param['random_state'])
            events_ica = mne.make_fixed_length_events(raw, duration=1.0)
            epochs_ica = mne.Epochs(raw, events_ica, tmin=0.0, tmax=1.0, baseline=None, preload = True)
            ica.fit(epochs_ica)
            eog_indices, eog_scores = ica.find_bads_eog(raw)
            ica.exclude = eog_indices
            ica.apply(raw)            

        epochs = mne.Epochs(
            raw,
            events,
            event_id,
            param['tmin'],
            param['tmax'],
            proj=False,
            picks=picks,
            baseline=None,
            preload=True,
            verbose=False,
            detrend=None
        )

        if param['ar']:
            ar = AutoReject(random_state = param['random_state'], verbose=False, n_interpolate = np.array([1, 2, 4, 8]))
            epochs  = ar.fit_transform(epochs) 

        if len(epochs['IC_correct']) < param['exclude_thresh'] or len(epochs['C_correct']) < param['exclude_thresh']:
            return (subj, None)  
        return(subj, epochs)
    except:
        return(subj, None)

def task_classification(epochs, param, subj, time_resovled = True, permutation = False, perm_seed = None):
    xd = _XdawnTransformer(n_components=param['n_filter'])
    steps = [('vec',Vectorizer()),
             ('scale',StandardScaler()),
             ('svc',SVC())]

    # defining parameter range
    pipe = Pipeline(steps)
    scores_all = []
    labels = epochs.events[:, -1]
    
    # Permute labels for test score
    if permutation == True:
        seed = np.random.RandomState(perm_seed)
        labels = seed.permutation(labels)

    vec = Vectorizer()
    epochs_data = vec.fit_transform(epochs)
    epochs_data = vec.inverse_transform(epochs_data)
    w_start = np.arange(0, epochs_data.shape[2] - param['w_length'], param['w_step'])
    
    # split and cv
    cv = StratifiedShuffleSplit(param['k'], test_size=param['test_size'], random_state=123)#i)
    cv_split = cv.split(epochs_data, labels)
    for train_idx, test_idx in cv_split:
        y_train, y_test = labels[train_idx], labels[test_idx]
        X_train = xd.fit_transform(epochs_data[train_idx])        
        X_test = xd.transform(epochs_data[test_idx])
        score_this = []

        if time_resovled == True:
            score_this = []

            for n in w_start:
                pipe.fit(X_train[:,:,n:n+param['w_length']], y_train)
            
                # score 
                if param['score'] == 'balanced_accuracy':
                    y_pred = pipe.predict(X_test[:,:,n:n+param['w_length']])
                    score_this.append(balanced_accuracy_score(y_test, y_pred))
                elif param['score'] == 'accuracy':
                    score_this.append(pipe.score(X_test[:,:,n:n+param['w_length']], y_test))
                elif param['score'] == 'roc_auc':
                    y_score = pipe.decision_function(X_test[:,:,n:n+param['w_length']])
                    score_this.append(roc_auc_score(y_test, y_score))
        else: 
            pipe.fit(X_train, y_train)
            if param['score'] == 'balanced_accuracy':
                y_pred = pipe.predict(X_test)
                score_this.append(balanced_accuracy_score(y_test, y_pred))
            elif param['score'] == 'accuracy':
                score_this.append(pipe.score(X_test, y_test))
            elif param['score'] == 'roc_auc':
                y_score = pipe.decision_function(X_test)
                score_this.append(roc_auc_score(y_test, y_score))

        scores_all.append(score_this)
    return(scores_all)