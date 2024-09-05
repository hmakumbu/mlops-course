from  mlops.load_data import load_train
from sklearn.datasets import load_iris

data_iris = load_iris()
X = data_iris.data 
y = data_iris.target



def test_unit():
    accuracy = evaluate()
    assert accuracy > 0.8, f"Expected accuracy > 0.8, but got {accuracy:.2f}"