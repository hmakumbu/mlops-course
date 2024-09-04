from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

data_iris = load_iris()
X = data_iris.data 
y = data_iris.target

def load_train():
     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
     return X_train, X_test, y_train, y_test

if __name__==  "__main__":
     
    X_features = pd.DataFrame(X)
    y_target = pd.DataFrame(y)
    df = pd.concat([X,y],axis=1)
    print(df)
