from util import FILES, METADATA, prepare_data
import pickle
from joblib import Parallel, delayed
import numpy as np
from sklearn.model_selection import StratifiedKFold
#from sklearn.svm import SVC
#from sklearn.ensemble import RandomForestClassifier
import xgboost

from sklearn.preprocessing import StandardScaler
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler 

import mne 
from mne.preprocessing.xdawn import Xdawn
from mne.decoding import Vectorizer

def run_window(pipe, X_train, X_test, y_train, y_test, n, param):
    pipe.fit(X_train[:,:,n:n+param['w_length']], y_train)
    y_pred = pipe.predict(X_test[:,:,n:n+param['w_length']])
    return(n, y_pred, y_test)

if __name__ == '__main__':
    overtime = False
    scores = {}
    scores_perm = {}
    faults = []
    store_name = 'age_prediction'
    param = {'tmin':-.1, 'tmax':.8,
            'filt_low':1,'filt_high':40,
            'sfreq':256,
            'w_length':20, 'w_step':1,
            'n_filter':5,
            'k':10,
            'test_size':0.2,
            'random_state':123,
            'score': 'accuracy',
            'exclude_thresh': 36,
            'classifier': 'sv-stratfied shuffle',
            'Under Sampling': None
            }
            
    #epochs = Parallel(n_jobs=-1)(delayed(prepare_data)(f,param) for f in FILES) ### done in preprocessing.py
    outfolder = '/home/christian/Schreibtisch/lifespan/results'
    file_epochs =  '/home/christian/Schreibtisch/lifespan/results/preprocessed_epochs.pkl'
    with open(file_epochs,'rb') as f:
        epochs = pickle.load(f)

    # get list of participants and map with metadata
    part = [data[0].lower() for data in epochs if data[1] != None]
    metadata = METADATA.loc[METADATA.index.isin(part),:]
    metadata = metadata[metadata.eegusedNEW == 1]
    subjects = metadata.index.values
    groups = metadata.GRUPPE.values

    # split into train and test subjects account for aging structure in dataset 
    cv = StratifiedKFold(n_splits=10)
    steps = [('vec',Vectorizer()),
            ('rus',RandomUnderSampler(random_state=param['random_state'])),
            ('scale',StandardScaler()),
            #('xgb', xgboost.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss'))
            #('rf', RandomForestClassifier())
            ('svc',SVC(probability=True))
            ]

    y_preds = []
    y_trues = []
    parts_tested = []
    fold = 0
    for train_idx, test_idx in cv.split(subjects,groups):
        print(f'running fold: {fold}')
        fold += 1
        subj_train = subjects[train_idx]
        subj_test = subjects[test_idx]

        # create global epochs object 
        epochs_train = []
        y_train = []
        epochs_test = []
        y_test = []
        for data in epochs:
            if data[1] != None:
                ntrials = len(data[1])
                label_vec = np.ravel(ntrials*[metadata[metadata.index.isin([data[0]])].GRUPPE.values])-1
            if data[0] in subj_train:
                epochs_train.append(data[1])
                y_train.append(label_vec)
            elif data[0] in subj_test:
                epochs_test.append(data[1])
                y_test.append(label_vec)


        epochs_train = mne.concatenate_epochs(epochs_train)
        y_train = np.concatenate(y_train)
        epochs_test = mne.concatenate_epochs(epochs_test)
        y_test = np.concatenate(y_test)

        xd = Xdawn(n_components=5)
        
        X_train = xd.fit_transform(epochs_train)
        X_test = xd.transform(epochs_test)
        pipe = Pipeline(steps)
        
        if overtime == False:
            pipe.fit(X_train,y_train)
            y_preds.append(pipe.predict(X_test))
            y_trues.append(y_test)
            with open(outfolder + '/xgb_fold' + str(fold) + '_pipe.pkl', 'wb') as f:
                pickle.dump(pipe, f, pickle.HIGHEST_PROTOCOL)

        else:
            w_start = np.arange(0, X_train.shape[2] - param['w_length'], param['w_step'])
            w_results = Parallel(n_jobs=-1)(delayed(run_window)(pipe, X_train, X_test, y_train, y_test, n, param) 
                        for n in w_start)
            y_preds.append([w_r[1] for w_r in w_results])
            y_trues.append([w_r[2] for w_r in w_results])

    with open(outfolder + '/xgb_age_group_y_preds.pkl', 'wb') as f:
            pickle.dump(y_preds, f, pickle.HIGHEST_PROTOCOL)

    with open(outfolder + '/xgb_age_group_y_trues.pkl', 'wb') as f:
            pickle.dump(y_trues, f, pickle.HIGHEST_PROTOCOL)




