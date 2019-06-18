#!/usr/bin/env python3

import pickle
import numpy as np
import sys

if __name__ == "__main__":
    pickle_in = open(sys.argv[1],"rb")
    example_dict = pickle.load(pickle_in)
    
    i = int(sys.argv[2])
    
    #skilled = example_dict[i]['predictions']['skilledPreds']
    #random = example_dict[i]['predictions']['randomPreds']
    #genuine = example_dict[i]['predictions']['randomPreds']
    #print('skilled: {:.2f} (+- {:.2f})'.format(np.mean(skilled) * 100, np.std(skilled) * 100))
    #print('random: {:.2f} (+- {:.2f})'.format(np.mean(random) * 100, np.std(random) * 100))
    #print('genuine: {:.2f} (+- {:.2f})'.format(np.mean(genuine) * 100, np.std(genuine) * 100))

    metrics = ['FRR', 'FAR_random', 'FAR_skilled', 'mean_AUC', 'EER', 'EER_userthresholds', 'auc_list', 'global_threshold']
    for m in metrics:
        print(f"{m}: {example_dict[i]['all_metrics'][m]}")
    
    #all_metrics = example_dict[i]['all_metrics']
    #predictions = example_dict[i]['predictions']
    #
    #print(predictions[i]['genuinePreds'])
    #print(predictions[i]['randomPreds'])
    #print(predictions[i]['skilledPreds'])