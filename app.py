from mlops.load_data import load_train
#from mlops.test_unit import test_unit
from sklearn.datasets import load_iris
from mlops.model import  ModelRegression

dfiris = load_iris()
X = dfiris.data
y = dfiris.target

#PipeLine

if __name__=="__main__":

    #load data
    X_train, X_test, y_train, y_test = load_train(X,y)

    #initialize model
    model = ModelRegression()

    #train model
    model.training(X_train=X_train, y_train=y_train)

    #prediction
    accuracy = model.evaluate(X_test,y_test)

    print(accuracy)
