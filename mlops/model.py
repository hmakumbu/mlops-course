from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score

class ModelRegression:

    def __init__(self):
       self.model = LogisticRegression(max_iter=200)

    def training(self,X_train, y_train):
        self.model.fit(X_train,y_train)
        return self.model

    def evaluate(self,X_test,y_test):
        prediction = self.model.predict(X_test)
        accuracy = accuracy_score(y_true=y_test,y_pred=prediction)
        return accuracy
