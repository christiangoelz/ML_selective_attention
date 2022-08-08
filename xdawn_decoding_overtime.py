import numpy as np
from util import FILES,METADATA, prepare_data, task_classification
import pickle 
import json
from joblib import Parallel, delayed

def run_classification(data):
    subj = data[0]
    print(subj)
    if data[1] == None:
        return (subj, None)
    epochs = data[1]
    time_res_scores = task_classification(epochs, param, subj, time_resovled = True, permutation = False)
    #perm_scores = permuation_score(epochs, param, subj, n_iterations=1000)
    return(subj, time_res_scores)#, perm_scores)

def permuation_score(epochs, param, subj, n_iterations):
    perm_scores = []
    for i in range(n_iterations):
        this_scores = task_classification(epochs, param, subj, time_resovled = False, permutation = True, perm_seed = i) 
        perm_scores.append(np.mean(this_scores))
    return(perm_scores)    

if __name__ == '__main__':
    scores = {}
    scores_perm = {}
    faults = []
    store_name = 'task_classification'
    param = {'w_length':20, 'w_step':1,
            'n_filter':5,
            'k':10,
            'test_size':0.2,
            'random_state':123,
            'score': 'roc_auc',
            'exclude_thresh': 36,
            'classifier': 'svc-stratfied shuffle',
            }
    file_epochs =  '/home/christian/Schreibtisch/lifespan/results/preprocessed_epochs.pkl'
    with open(file_epochs,'rb') as f:
        epochs = pickle.load(f)        
    results = Parallel(n_jobs=-1)(delayed(run_classification)(epoch) for epoch in epochs)
    
    #extract results
    for r in results:
        if r[1] != None:
            scores[r[0]] = r[1]
            # scores_perm[r[0]] = r[2]
        else: 
            faults.append(r[0])

    # Save everything
    with open(store_name + '_parameter.json', 'a',encoding="utf-8") as file:
        json.dump(param, file)

    with open(store_name + '_results.pkl', 'wb') as f:
        pickle.dump(scores, f)

    # with open(store_name + '_indiv_permutation_results.pkl', 'wb') as f:
    #     pickle.dump(scores_perm, f)

    with open("faults.txt", "w") as output:
        output.write(str(faults))
    
