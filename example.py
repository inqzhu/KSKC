# -*- coding: utf-8 -*-

import pandas as pd 

import ks_multi as ksm
import eva


if __name__=='__main__':
    """
    conduct KSKC clustering
    """
    worker = ksm.KSKC()
    # set the column name of objects
    worker.obj_name = 'Merchant_id'
    # set the column name of observation records
    worker.rec_name = 'Transaction'
    # set the number of clusters
    K = 3
    # set the input dataset
    source = 'test_data.csv'
    # the return is the clustering result
    # containing 2 columns: obj_name, cluster_label
    # each object is assigned to 1 cluster
    cluster_result = worker.work(K, source)
    cluster_result.to_csv('result.csv', index=False)
    


    """
    test clustering performance
    if there are true labels for all objects
    """
    # load clustering result
    pred_labels = pd.read_csv('result.csv')
    # load true labels (only for evaluation; not used in the clustering)
    true_labels = pd.read_csv('test_data_label.csv')
    # merge clustering result and true labels
    data = pd.merge(pred_labels, true_labels, on='Merchant_id')
    # evaluation
    # criterion: NMI, ARI, ACC(based on Hungarian algorithm)   
    eva.evaluate(3, data['cluster_label'], data['label'])


