
# -*- coding: utf-8 -*-
from time import sleep

import datetime
import readdata
import numpy as np
import matplotlib.pyplot as plt
import evaluation
#from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler
import ddbscn

name = "20000_300_std"
data, cor_labels = readdata.deal('new_datasets/'+str(name)+'.txt')
core_point = readdata.read_core_point('new_datasets/20000_300_stdcorept.txt')
ar = np.array(data)
br = ar[:,1:3]
print (br)
X = br
# Core_Sam = [20000,20001,20002,20003,20004,20005,20006,20007,20008,20009,20010]
#计算
begin = datetime.datetime.now()
CS= core_point[:]
db= ddbscn.DBSCAN(eps=0.283, min_samples=5).fit(X, Core_samples=core_point)


new_labels = []
# for i in db.labels_:
#     new_labels.append(str(i+1))
print ("labels")#
print (new_labels)
print (cor_labels)

core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

n_clusters_ = len(set(labels))-(1 if -1 in labels else 0)

unique_labels = set(labels)
colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

end2 = datetime.datetime.now()
core_duration = end2-begin

for k,col in zip(unique_labels,colors):
    if k == -1:
        col = 'k'
    class_member_mask = (labels == k)
    xy = X[class_member_mask & core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], '.', markerfacecolor=col,
             markeredgecolor='k', markersize=20)

    xy = X[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], '.', markerfacecolor=col,
             markeredgecolor='k', markersize=10)
for i in CS:# 标记核心点
    print (i)
    plt.plot(X[i][0], X[i][1], '.', markerfacecolor='red',
             markeredgecolor='k', markersize=30)
end = datetime.datetime.now()
ttt = end - begin
print("Total length of time:"+str(ttt))
print(cor_labels)
print(labels)
ARS = evaluation.ARS(cor_labels, labels)
NMI = evaluation.NMI(cor_labels, labels)
plt.title(name+'\n'+'Estimated number of clusters: %d' % n_clusters_+'    '+"Dbscan_duration:"+str(core_duration)+'    '+"Total length of time:"+str(ttt)+'\n'+'Adjusted Rand Score:'+str(ARS)+'    '+'Normalized Mutual Information:'+str(NMI))
print("Dbscan_duration:"+str(core_duration))
print("Total length of time:"+str(ttt))
plt.show()
