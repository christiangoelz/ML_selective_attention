import numpy as np
from util import FILES, prepare_data
import pickle 
import json
from joblib import Parallel, delayed

param = {'tmin':-.1, 'tmax':.8,
        'filt_low':1,'filt_high':40,
        'sfreq':256,
        'score': 'accuracy',
        'exclude_thresh': 36,
        'ica':False,
        'ar':False,
        'random_state': 777}

epochs = Parallel(n_jobs=-1)(delayed(prepare_data)(f,param) for f in FILES) 

outfolder = '/home/christian/Schreibtisch/lifespan/results'
with open(outfolder + '/preprocessed_epochs.pkl', 'wb') as f:
            pickle.dump(epochs, f, pickle.HIGHEST_PROTOCOL)