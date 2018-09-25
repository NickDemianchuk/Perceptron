import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import normalize

from adaline import Adaline
from task_3_4.data_cleaner import DataCleaner


def build_plot():
    plt.plot(range(1, len(ada.cost_) + 1), ada.cost_, marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Cost')
    plt.tight_layout()
    plt.show()


def get_accuracy(actual, predicted):
    counter = 0
    for i in range(actual.shape[0]):
        if actual[i] == predicted[i]:
            counter += 1

    accuracy = counter / actual.shape[0] * 100
    print('The accuracy of the prediction is ' + str(round(accuracy, 2)) + '%')


def task_four():
    # Getting features from dataframe
    columns = list(df)
    # Adding features and corresponding weights to the list
    weights = []
    for i in range(1, len(ada.w_)):
        weights.append(str(round(abs(ada.w_[i]), 2)) + " " + columns[i])
    # Sorting and printing the features with the highest weight
    print('The most predictive features are:')
    print(sorted(weights, reverse=True)[:3])


dc = DataCleaner()
df = pd.read_csv('train.csv')
# Preprocessing data
df = dc.clean_data(df)

X = df.values[:, 1:]
X = normalize(X, axis=0, norm='max')
y = df.values[:, 0]

# Splitting dataset into 70% train and 30% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0)

# Training Adaline classifier
ada = Adaline(lrn_rate=0.0001, epochs=100)
ada.fit(X_train, y_train)

# Building plot for the cost function
build_plot()
# Evaluating on test data
get_accuracy(y_test, ada.predict(X_test))
# Determining the most predictive feature
task_four()
