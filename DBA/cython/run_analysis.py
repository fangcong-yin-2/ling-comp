from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from DBA import performDBA
import json
import os
from scipy import stats

def read_data(data_path,seq_key):
    full_data = json.load(open(data_path,'r'))
    data = []
    dmax = 0
    dmin = 1000
    for row in full_data:
        if len(row) == 0 or seq_key not in row or len(row[seq_key]) == 0:
            continue
        if type(row[seq_key])==list and type(row[seq_key][0]) == list:
            data.extend(row[seq_key])
        else:
            ## Standardize the input feature
            data.append(stats.zscore(row[seq_key]))
    return data

def group_data(data):
    ## Bin data by length
    lens = [len(seq) for seq in data]
    quantiles = [np.quantile(lens,i) for i in np.linspace(0.05,0.95,20)]
    quantiles = [int(q) for q in set(quantiles)]
    ## Define a minimum sentence length
    min_sent_len = 3
    quantiles = range(max(int(np.quantile(lens,0.05)),min_sent_len),int(np.quantile(lens,0.95)))
    groups = [[] for i in range(len(quantiles)-1)]
    for i in range(len(data)):
        if True in np.isnan(data[i]):
            continue
        if len(data[i]) == 1:
            continue
        for j in range(0,len(quantiles)-1):
            if len(data[i]) >= quantiles[j] and len(data[i]) < quantiles[j+1]:
                ## Interpolate features in the same length group to the same length
                padded_data = interpolate(data[i],end_steps=100)
                groups[j].append(padded_data)
                break
                
    return groups
    
def sample_data(data,sample_size,group_size):
    dlen = len(data)
    groups = []
    for i in range(group_size):
        group = []
        sample_idx = np.random.choice(range(dlen),sample_size,replace=False)
        for j in sample_idx:
            if len(data[j]) <= 1:
                continue
            group.append(interpolate(data[j],end_steps=100))
        groups.append(group)
    return groups        


def interpolate(stream,end_steps=100):
     # interpolate to end_steps (100) steps
     real_len = len(stream)
     full_steps = int((end_steps-real_len) / (real_len-1))
     partial_steps = (end_steps-real_len) % (real_len-1)
     output = []
     for step in range(real_len-1):
         splits = full_steps
         if step < partial_steps:
             splits += 1
         start = stream[step]
         end = stream[step+1]
         slope = (end - start) / (splits+1)
         curr = start
         output.append(curr)
         for i in range(splits):
             curr += slope
             output.append(curr)

     output.append(stream[-1])
     return(np.array(output))
def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'same') / w

def main(w):
    ## Provide the path to your feature data file
    ## Feature data file should contain a list of dictionary where each dictionary key is a feature of a source-summary pair
    data_path = '../../agg_analysis_test_data/scitldr_test_dlt.json'
    ## Define the name of the feature in the feature data file
    ## The name should be a key of the dictionary
    ## Each feature should be a list of list of numerical features
    key = 'clean_dlt_pr'
    data = read_data(data_path,key)
    ## Group data by length of the feature
    groups = group_data(data)
    ## Or you can simply sample
    # groups = sample_data(data,sample_size=100,group_size=10)
    group_avgs = []
    total = 0
    for group in groups:
        ## Compute DBA contour for each group
        np_group = np.array(group)
        n = np_group.shape[0]
        total+=n
        group_avg = performDBA(np_group[:],w=np_group.shape[1]//2,n_iterations=10)
        group_avgs.append(np.asarray(group_avg))
    max_len = np.max([np.array(group).shape[1] for group in groups])
    max_len = 100+2*w
    int_groupavgs = []
    for group_avg in group_avgs:
        int_gavg = stats.zscore(interpolate(group_avg,end_steps = max_len))
        
        int_groupavgs.append(int_gavg)
    int_groupavgs = np.array(int_groupavgs)
    return int_groupavgs

if __name__== "__main__":
    avgs = []
    w = 0
    ## Obtain DBA averages with n_iter initializations
    n_iter = 5
    for i in range(n_iter):
        grand_avg = main(w=w)
        avgs.extend(grand_avg)
    avgs = np.array(avgs)
    ## Apply DBA average again on the average contours of all initializations
    mean_avg = performDBA(avgs,w=avgs.shape[1]//2,n_iterations=10)
    mean_avg = np.asarray(mean_avg)
    plt.plot(range(100),mean_avg[:])
    plt.ylabel('Standardized Feature Value')
    plt.xlabel('Relative Position in Sentence')
    plt.show()
