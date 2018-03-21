from sklearn import svm
import numpy as np
import readdata
import plotdata
data = readdata.deal('../dataset/spiral.txt')

print (data)
plotdata.plotdata(data)

data = np.asarray(data)
X = data[:, 1:3]
y = data[:, 0]
clf = svm.SVC()
clf.fit(X,y)
clf.predict([[2.,2.]])