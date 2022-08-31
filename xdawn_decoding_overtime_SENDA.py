from xdawn_decoding_overtime import run_classification, permuation_score
import pickle
import json
from joblib import Parallel, delayed

if __name__ == '__main__':
    scores = {}
    scores_perm = {}
    faults = []
    store_name = 'task_classification'
    param = {'w_length': 20, 'w_step': 1,
             'n_filter': 5,
             'k': 10,
             'test_size': 0.2,
             'random_state': 123,
             'score': 'roc_auc',
             'exclude_thresh': 36,
             'classifier': 'svc-stratfied shuffle',
             }
    file_epochs = '/home/christian/Schreibtisch/lifespan/results/preprocessed_epochs_SENDA.pkl'
    with open(file_epochs, 'rb') as f:
        epochs = pickle.load(f)
    results = Parallel(n_jobs=1)(delayed(run_classification)(epoch, param)
                                 for epoch in epochs)

    # extract results
    for r in results:
        if r[1] != None:
            scores[r[0]] = r[1]
            scores_perm[r[0]] = r[2]
        else:
            faults.append(r[0])

    # Save everything
    with open(store_name + '_parameter_SENDA.json', 'a', encoding="utf-8") as file:
        json.dump(param, file)

    with open(store_name + '_results_SENDA.pkl', 'wb') as f:
        pickle.dump(scores, f)

    with open(store_name + '_indiv_permutation_results_SENDA.pkl', 'wb') as f:
        pickle.dump(scores_perm, f)

    with open("faults_SENDA.txt", "w") as output:
        output.write(str(faults))
