# -*- coding: utf-8 -*-


"""
clustering performance evaluation
"""
import pandas as pd
import math
import time
import numpy as np
from itertools import permutations

def evaluate(K, pred_labels, labels):
    # pred_labels: clustering result
    # labels: true labels
    t0 = time.time()
    n = len(labels)

    # map labels to numbers
    label_to_number = {}
    number_to_label = {}
    labels_str = labels.unique()
    for i in range(len(labels_str)):
        label_to_number[labels_str[i]] = i
        number_to_label[i] = labels_str[i]

    
    # contingency table
    d = pd.DataFrame( {'pred': list(pred_labels), 'true': list(labels) } )
    ct = np.zeros([K,K])
    for k1 in range(K):
        for k2 in range(K):
            cnt = len(d[ (d['true']==number_to_label[k1]) & (d['pred']==k2) ]) # 预测的划分、真实划分，相交的个数
            ct[k1][k2] = cnt

    # print the table
    print("--> Clustering Matrix ( row - true labels ; columns - clusters): ")
    print("==============================")
    show_line = '\t|'
    for k in range(K):
        show_line += ' ' + str(k) + " \t|"
    print(show_line)
    for k1 in range(K):
        show_line = 'True-' + str(number_to_label[k1]) + "|"
        for k2 in range(K):
            show_line = show_line + str(ct[k1][k2]) + "\t| "
        print(show_line)
    print("==============================")
    
    # NMI
    MI = 0
    h_S = 0
    h_real_S = 0
    for k1 in range(K):
        for k2 in range(K):
            p_info = ct[k1][k2] / float(n)
            p_real_S = ct[k1,:].sum() / float(n)
            p_S = ct[:, k2].sum() / float(n)
            if p_info != 0:
                MI = MI + p_info * math.log(p_info / p_S / p_real_S)
            else:
                p_info = 0
    for k in range(0, K):
        p_S = ct[:, k].sum() / float(n)
        p_real_S = ct[k,:].sum() / float(n)
        if p_S != 0:
            h_S = h_S - p_S * math.log(p_S)
        if p_real_S != 0:
            h_real_S = h_real_S - p_real_S * math.log(p_real_S)
    NMI = MI / math.sqrt(h_S*h_real_S)
    print("  -> NMI: %.4f" % NMI)
    
    
    # ARI
    sa2 = .0
    sb2 = .0
    for k in range(K):
        sa2 += cn2(ct[k,:].sum())
        sb2 += cn2(ct[:,k].sum())
    sab2 = .0
    for k1 in range(K):
        for k2 in range(K):
            sab2 += cn2( ct[k1][k2] ) 
    ARI = (sab2 - sa2*sb2/cn2(n))/(0.5*(sa2+sb2)-sa2*sb2/cn2(n))
    print("  -> ARI: %.4f" % ARI)
    
    # ACC
    perm = permutations(list(range(K)))
    acc_K = []
    for _perm in list(perm):
        _acc = .0
        for k in range(K):
            _acc += ct[k][_perm[k]]
        acc_K.append(_acc)
    ACC = max(acc_K) / n
    print("  -> ACC: %.4f" % ACC)
    
    t1 = time.time()
    print('%.2fs' % (t1-t0))
    return [NMI, ARI, ACC]
    
def cn2(x):
    return x*(x-1)/2
    