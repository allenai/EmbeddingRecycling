import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

X_train = np.array([[[-1, -1], [0, 1]], [[-2, -1], [0, 1]], [[0, 1], [1, 1]], [[0, 1], [2, 1]]])
X_test = np.array([[[2, -1], [1, 0]], [[2, 0], [1, 1]]])

Y_train = np.array([1, 1, 2, 2])
Y_test = np.array([0, 2])

print(X_train.shape)
print(X_test.shape)

X_train=X_train.reshape(len(X_train),-1)
X_test=X_test.reshape(len(X_test),-1)

print(X_train.shape)
print(X_test.shape)



# Always scale the input. The most convenient way is to use a pipeline.
clf = make_pipeline(StandardScaler(),
                    SGDClassifier(max_iter=1000, tol=1e-3))
clf.fit(X_train, Y_train)


print(clf.predict(X_test))




