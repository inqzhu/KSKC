# -*- coding: utf-8 -*-

import pandas as pd
import time
import numpy as np
import random
import scipy.stats as ss
import math
import copy
import multiprocessing

import warnings
warnings.filterwarnings("ignore")


def now_time():
    # report current time
    return time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))

class KSKC(object):
    # whether to adopt degree correction
    is_corrected = False
    # number of available processors (for multi processing)
    NUM_THREAD = 4
    # Fast KSKC
    is_fast = True
    # Fast KSKC - subsample size
    sample_num = 1000

    # column name of object ID
    obj_name = None
    # column name of observation record
    rec_name = None

    # number of iterations
    niter = 30

    # calculate K-S distance
    def ks_dis(self, a, b):
        [d, s] = ss.ks_2samp(a, b)

        if self.is_corrected:
            # degree correction
            n = len(a)
            m = len(b)
            d = math.sqrt(n*m/float(n+m)) * d
        return [d, s]

    # return K NULL lists 
    # for storing K index sets of the clusters
    def blank_samples(self, K):
        c_samples = []
        for i in range(0, K):
            c_samples.append([])
        return c_samples  

    # E-step
    def e_procedure(self, K, data, c_samples):
        c = {}
        for i in range(0, K):
            temp_c = data[ data[self.obj_name].isin(c_samples[i]) ][self.rec_name]

            if self.is_fast:
                if len(temp_c) > self.sample_num:
                    new_temp_c = temp_c
                    temp_c = sorted(new_temp_c)
                    new_temp_c = []
                    block = int(len(temp_c) / self.sample_num)
                    for j in range(0, len(temp_c)):
                        if j % block == 0:
                            new_temp_c.append(temp_c[j])
                    temp_c = new_temp_c

            c[i] = temp_c
        return c

    # M-step(for multi processing)
    def sub_m(self, K, mids, trans_dic, c, order, st, ed, global_c, global_s):
        c_samples = self.blank_samples(K)
        s_samples = self.blank_samples(K)

        for i in range(st, ed):
            min_dis = -1
            opt_c = -1
            for j in range(0, K):
                [temp_ks, temp_s] = self.ks_dis(trans_dic[mids[i]], c[j])
                if temp_ks < min_dis or min_dis < 0:
                    min_dis = temp_ks
                    opt_c = j
            # assign this point to the cluster with the smallest K-S distance
            c_samples[opt_c].append(mids[i])
            s_samples[opt_c].append(min_dis)

        global_c[order] = c_samples
        global_s[order] = s_samples

    # M-step
    def m_procedure(self, K, mids, trans_dic, c):
        NUM_THREAD = self.NUM_THREAD
        manager = multiprocessing.Manager()
        global_c = manager.dict()
        global_s = manager.dict()
        jobs = []
        Ncs = len(mids)
        block = int(Ncs / NUM_THREAD)

     	# conduct M-step using multi processing
        for t in range(0, NUM_THREAD):
            st = t*block
            ed = (t+1)*block
            worker = multiprocessing.Process(target = self.sub_m, args = (K, mids, trans_dic, c, t, st, ed, global_c, global_s, ))
            jobs.append(worker)
            worker.start()

        for j in jobs:
            j.join()

        c_samples = self.blank_samples(K)
        s_samples = self.blank_samples(K)

        for d,x in global_c.items():
            for k in range(0, K):
                c_samples[k] += x[k]

        for d,x in global_s.items():
            for k in range(0, K):
                s_samples[k] += x[k]

        return [c_samples, s_samples]

    # KSKC algorithm
    def work(self, K, source):
        print("\n==> start!", now_time())

        # 对每个类，用一个列表存放属于该类的样本ID
        c_samples = self.blank_samples(K)
      
        # load dataset
        data = pd.read_csv(source)
        
        # obtain object list
        mids = data[self.obj_name].unique().tolist()
        n = len(mids)
        print("--> Amount of objects: ", n)

        # obtain all observation records for each object
        trans_dic = {}
        groups = data.groupby(data[self.obj_name])
        for g in groups:
            temp_data = g[1][self.rec_name]
            trans_dic[g[0]] = temp_data
        print("--> Transformation finished ", now_time())

       
        # initialize cluster centers
        '''
        random_seed = random.sample(range(n), K)
        for i in range(0, K):
            c_samples[i].append(mids[random_seed[i]])
        '''
        time_s0 = time.time()
        random_seed = random.sample(range(n), 1)
        c_samples[0].append(mids[random_seed[0]])
        for i in range(1, K):
            max_dis = -1
            next_center = -1
            for j in range(0, n):
                temp_data = trans_dic[mids[j]]
                min_dis = -1
                for z in range(0, i):
                    [temp_d, temp_s] = self.ks_dis(temp_data, trans_dic[c_samples[z][0]])
                    if temp_d < min_dis or min_dis < 0:
                        min_dis = temp_d
                if min_dis > max_dis:
                    max_dis = min_dis
                    next_center = mids[j]
            c_samples[i].append(next_center)
        time_s1 = time.time()
        print("--> initial cluster center seletion: %.2f seconds" % (time_s1 - time_s0))

        # iterative estimation
        time0 = time.time()
        for t in range(0, self.niter):
            iter_time0 = time.time()
            oc_samples = copy.deepcopy(c_samples)
            # E-step 
            c = self.e_procedure(K, data, c_samples)
            # M-step
            [c_samples, s_samples] = self.m_procedure(K, mids, trans_dic, c)       
            #"""
            # print current cluster sizes
            print("------------", t)
            for i in range(0, K):
                print(i, ' == ',len(c_samples[i]))
            #"""
            if oc_samples == c_samples:
                # achieve convergence
                break
            iter_time1 = time.time()
            print("# time of iteration: ", iter_time1 - iter_time0)

        time1 = time.time()
        cpu_time = time1 - time0

        
        # output result
        assign = []
        for k in range(K):
            for i in range(len(c_samples[k])):
                assign.append( [c_samples[k][i], k ] )
        assign = pd.DataFrame(assign, columns=[self.obj_name, 'cluster_label'])
        return assign



