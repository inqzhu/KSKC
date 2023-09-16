# Kolmogorov-Smirnov K-means clustering (KSKC)

KSKC is an iterative algorithm developped for clustering objects with repeated observations.
It is motivated by the clustering of merchants using online payment platforms. The objective is to segment all merchants to non-overlapping groups according to their transaction records. For each merchant, his or her transactions can be viewed as repeated observations about his or her behavior pattern. Rapid developments in third-party online payment platforms now make it possible to record massive bank card transaction data. Clustering on such transaction data is of great importance for the analysis of merchant behaviors. However, traditional methods based on generated features inevitably lead to much loss of information. To make better
use of bank card transaction data, this study investigates the possibility of using the empirical cumulative distribution of transaction amounts. As the distance between two merchants can be measured using the two-sample Kolmogorov-Smirnov test statistic, we propose the Kolmogorov-Smirnov K-means clustering approach based on the distance measure. 


Besides transaction data, KSKC is applicable for various clustering scenarios that involves repeated observations.
The essence of KSKC is to cluster objects according to their ***distributions***. For each object, we obtain its empirical cumulative distribution function (ECDF) based on the repeated observations. Then, each object is presented by its ECDF. The KSKC algorithm is thus implemented using those ECDFs. Objects with similar ECDFs will be clustered together.

More details of KSKC can be found at: https://doi.org/10.1111/rssc.12471

The citation is: `Yingqiu Zhu, Qiong Deng, Danyang Huang, Bingyi Jing, Bo Zhang, Clustering Based on Kolmogorov–Smirnov Statistic with Application to Bank Card Transaction Data, Journal of the Royal Statistical Society Series C: Applied Statistics, Volume 70, Issue 3, June 2021, Pages 558–578, https://doi.org/10.1111/rssc.12471`


To use KSKC, please import the code `ks_multi.py`. The file `example.py` provides an example.
```Python
    import ks_multi as ksm
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
```

The structure of the input CSV should be, for example:
| Merchant_id | Transaction |
| ----------- | ----------- | 
| mer001 | 100 | 
| mer001 | 105 |
| mer002 | 50 | 
| mer002 | 45 |
| ... | ... | 
| mer100 | 10 |

Here `obj_name` and `rec_name` are `Merchant_id` and `Transaction`, respectively.
The output of KSKC (`cluster_result`) is organized as a DataFrame:
| Merchant_id | cluster_label |
| ----------- | ----------- | 
| mer001 | 0 | 
| mer002 | 1 |
| ... | ... | 
| mer100 | 2 |


In addition, ``ks_multi.KSKC()`` supports further specifications:
| Attribute | Description | Default | 
| ----------- | ----------- | ----------- | 
| is_corrected | whether to adopt degree correction | False |
| NUM_THREAD | number of available processors | 4 |
| is_fast | whether to adopt the fast version of KSKC | True |
| sample_num | subsample size for the fast version | 1000 |
| niter | maximal number of iterations | 30 |