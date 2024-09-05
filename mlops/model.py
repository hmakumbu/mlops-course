from sklearn.linear_model import LogisticRegression
from mlops.load_data import load_train
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score


X_train, X_test, y_train, y_test = load_train()


model = LogisticRegression(max_iter=200)

model.fit(X_train,y_train)

def evaluate():
    prediction = model.predict(X_test)
    accuracy = accuracy_score(y_true=y_test,y_pred=prediction)
    return accuracy
