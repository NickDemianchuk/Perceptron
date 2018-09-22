import pandas as pd

from perceptron import Perceptron
from plot_builder import PlotBuilder

df = pd.read_csv('task_one.csv')
X = df.iloc[:, [1, 2]].values
y = df.iloc[:, 3].values
pb = PlotBuilder()
ppn = Perceptron(lrn_rate=0.1, epochs=40)
ppn.fit(X, y)

pb.plot_data(X, 'Height [cm]', 'Weight [kg]')
pb.plot_misclassifications(ppn)
pb.plot_decision_regions(X, y, ppn, 'Height [cm]', 'Weight [kg]')
