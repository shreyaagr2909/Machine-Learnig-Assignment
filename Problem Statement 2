from matplotlib import pyplot
from sklearn.datasets import make_blobs
from sklearn.datasets import make_regression
from pandas import DataFrame

class GenerateData():
  def logisticRegre(self):
    # generate 2d classification dataset
    X2, y2 = make_blobs(n_samples=100, centers=2, n_features=2)
    # scatter plot, dots colored by class value
    df = DataFrame(dict(x=X2[:,0], y=X2[:,1], label=y2))
    colors = {0:'red', 1:'blue'}
    fig, ax = pyplot.subplots()
    grouped = df.groupby('label')
    for key, group in grouped:
        group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key])
    pyplot.show()
    return X2, y2
  def linearRegre(self):
    # generate regression dataset
    X1, y1 = make_regression(n_samples=100, n_features=3, noise=0.3)
    # plot regression dataset
    pyplot.scatter(X1[:,0]+X1[:,1]+X1[:,2],y1)
    pyplot.show()
    return X1, y1
  def kMeans(self):
    # generate 2d classification dataset
    X3, y3 = make_blobs(n_samples=100, centers=3, n_features=2)
    # scatter plot, dots colored by class value
    df = DataFrame(dict(x=X3[:,0], y=X3[:,1], label=y3))
    colors = {0:'red', 1:'blue', 2:'green'}
    fig, ax = pyplot.subplots()
    grouped = df.groupby('label')
    for key, group in grouped:
        group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key])
    pyplot.show()
    return X3, y3
