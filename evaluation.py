import sklearn.cluster
from sklearn import metrics

def ARS(labels_true,labels_pred):
    '''adjusted_rand_score 兰德调整指数'''
    return metrics.adjusted_rand_score(labels_true,labels_pred)

def NMI(labels_true,labels_pred):
    '''Normalized Mutual Information 标准互信息'''
    return metrics.adjusted_mutual_info_score(labels_true,labels_pred)