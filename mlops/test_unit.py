from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

data_iris = load_iris()
X = data_iris.data 
y = data_iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model=LogisticRegression()
model.fit(X_train, y_train)
prediction = model.predict(X_test)

def test_unit():
    accuracy = accuracy_score(y_true=y_test, y_pred=prediction)
    assert accuracy > 0.8, f"Expected accuracy > 0.8, but got {accuracy:.2f}"

if __name__=="__name__":

    test_unit()