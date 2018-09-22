import pandas as pd

from perceptron import Perceptron
from plot_builder import PlotBuilder

df = pd.read_csv('task_two.csv')
X = df.iloc[:, [1, 2]].values
y = df.iloc[:, 3].values
pb = PlotBuilder()
ppn = Perceptron(lrn_rate=0.1, epochs=1000)
ppn.fit(X, y)

pb.plot_data(X, 'Height [cm]', 'Age')
pb.plot_misclassifications(ppn)
