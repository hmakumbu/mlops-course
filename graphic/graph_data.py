import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

iris = load_iris()

plt.figure(figsize=(8, 6))
sns.scatterplot(x=iris.data[:, 0], y=iris.data[:, 2], hue=iris.target, palette='Set1')
plt.title('Scatter Plot: Sepal Length vs Petal Width')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Petal Width (cm)')
plt.legend(title='Class')
plt.show()


def graph_perfomance(y_test,y_pre, data):

    cm = confusion_matrix(y_test,y_pre)
    # Plot the confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=iris.target_names)
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.show()
