from  mlops.load_data import load_train
from sklearn.datasets import load_iris

data_iris = load_iris()
X = data_iris.data 
y = data_iris.target
X_train, X_test, y_train, y_test = load_train(X,y)

#initialize model
model = ModelRegression()

#train model
model.training(X_train=X_train, y_train=y_train)

#prediction
accuracy = model.evaluate(X_test,y_test


def test_unit():
    accuracy = evaluate()
    assert accuracy > 0.8, f"Expected accuracy > 0.8, but got {accuracy:.2f}"