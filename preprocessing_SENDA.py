import mne
import pickle
import glob
from pathlib import Path
from joblib import Parallel, delayed


def _annotations_from_event(raw):
    '''
    Correct annotation description

    Parameters
    ----------
    raw : mne raw object,
        eeg data as mne object.

    Returns
    -------
    raw with corrected annotation.

    '''
    # Renaming events
    events, events_id = mne.events_from_annotations(raw)
    events_id = {201: 'C_correct', 2: 'C_correct_response',
                 3: 'C_false', 4: 'C_false_response',
                 5: 'N_correct', 6: 'N_correct_response',
                 7: 'N_false', 8: 'N_false_response',
                 203: 'IC_correct', 10: 'IC_correct_response',
                 11: 'IC_false', 12: 'IC_false_response'}

    #1001 - congruent
    #1002 - neutral
    #1003 - incongruent
    #1004 - correct
    #1005 - false

    for e in range(len(events)-1):
        if events[e, 2] == 10001 and events[e+1, 2] == 10004:
            events[e, 2] = 201
            events[e+1, 2] = 2
        elif events[e, 2] == 10002 and events[e+1, 2] == 10004:
            events[e, 2] = 5
            events[e+1, 2] = 6
        elif events[e, 2] == 10003 and events[e+1, 2] == 10004:
            events[e, 2] = 203
            events[e+1, 2] = 10
        elif events[e, 2] == 10001 and events[e+1, 2] == 10005:
            events[e, 2] = 3
            events[e+1, 2] = 4
        elif events[e, 2] == 10002 and events[e+1, 2] == 10005:
            events[e, 2] = 7
            events[e+1, 2] = 8
        elif events[e, 2] == 10003 and events[e+1, 2] == 10005:
            events[e, 2] = 11
            events[e+1, 2] = 12

    return (events, events_id)


def prepare_SENDA(file, param):
    try:
        subj = file.split('/')[-1][:6]
        print(subj)
        raw = mne.io.read_raw_brainvision(file, preload=True)
        events, events_id = _annotations_from_event(raw)
        annot = mne.annotations_from_events(events=events,
                                            event_desc=events_id,
                                            sfreq=raw.info['sfreq'],
                                            orig_time=raw.info['meas_date'])
        raw.set_annotations(annot)

        event_id = {'C_correct': 201, 'IC_correct': 203}
        raw.filter(param['filt_low'], param['filt_high'], fir_design="firwin")

        picks = mne.pick_types(
            raw.info, eeg=True, stim=False, eog=False, exclude="bads")
        raw, events = raw.resample(param['sfreq'], events=events)
        raw.set_montage('standard_1020')

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
        epochs = epochs[['C_correct', 'IC_correct']]

        if len(epochs['IC_correct']) <= 35:
            return (subj, None)

        elif len(epochs['C_correct']) <= 35:
            return (subj, None)
        return (subj, epochs)
    except:
        return (subj, None)


if __name__ == '__main__':
    param = {'tmin': -.1, 'tmax': .8,
             'filt_low': 1, 'filt_high': 40,
             'sfreq': 256,
             'score': 'accuracy',
             'exclude_thresh': 36,
             'ica': False,
             'ar': False,
             'random_state': 777}

    path = "/media/christian/SEAGATE_BAC/Hard_drive/data/SENDA/Flanker_T1"
    FILES = glob.glob(str(Path(path) / '*.vhdr'))
    epochs = Parallel(n_jobs=-1)(delayed(prepare_SENDA)(f, param)
                                 for f in FILES)
    outfolder = '/home/christian/Schreibtisch/lifespan/results'
    with open(outfolder + '/preprocessed_epochs_SENDA.pkl', 'wb') as f:
        pickle.dump(epochs, f, pickle.HIGHEST_PROTOCOL)
