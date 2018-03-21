import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.datasets import make_blobs


def make_first():
    X1, y1=datasets.make_circles(n_samples=5000, factor=.6,
                                      noise=.05)
    X2, y2 = datasets.make_blobs(n_samples=1000, n_features=2, centers=[[1.2,1.2]], cluster_std=[[.1]],
               random_state=9)

    X = np.concatenate((X1, X2))
    print(X)
    plt.scatter(X[:, 0], X[:, 1], marker='o')
    plt.show()

def make_20000():
    centers = []
    for i in range(0,90,3):
        for j in range(0,60,3):
            centers.append([i,j])

    X, labels_true = make_blobs(n_samples=100000,
                                centers=centers,
                                cluster_std=0.4,
                                random_state=0)
    X=np.row_stack((X,centers))

    tmp = -np.ones((1,len(centers)))[0]
    tmp_ = list(tmp)
    labels_ = list(labels_true)
    labels_ = labels_+tmp_
    labels_true = np.array(labels_)

    plt.scatter(X[:, 0], X[:, 1], marker='o')
    plt.show()
    save_data(X,labels_true,centers,'100000_500_std')


def save_data(X, label,core_pt, name):
    X1 = np.column_stack((X,label))

    print (X1)
    f=open(r'new_datasets/'+name+'.txt','a')
    for i in X1:
        f.write(str(i[2])+' '+str(i[0])+' '+str(i[1])+'\n')
    f.close()

    f = open(r'new_datasets/'+name+'corept.txt','a')
    for i in core_pt:
        f.write(str(i[0])+' '+str(i[1])+'\n')
    f.close()

if __name__=='__main__':
    make_20000()