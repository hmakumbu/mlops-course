from mlops.load_data import load_train
from mlops.model import LogisticRegression, evaluate
from mlops.test_unit import test_unit
from sklearn.datasets import load_iris

dfiris = load_iris()
X = dfiris.data
y = dfiris.target

#PipeLine

#load data
X_train, X_test, y_train, y_test = load_train(X,y)

#train




