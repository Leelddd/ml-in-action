import matplotlib.pyplot as plt
import operator
import pickle

decisionNode = dict(boxstyle='round', pad=0.1, fc='0.8')
leafNode = dict(boxstyle='circle', fc='0.8')
arrow_args = dict(arrowstyle="<-")


def plotNode(nodeTxt, start, to, nodeType):
    createPlot.ax1.annotate(nodeTxt, xy=start, xycoords='axes fraction',
                            xytext=to, textcoords='axes fraction',
                            va="center", ha="center", bbox=nodeType, arrowprops=arrow_args)


def plotArr(txt, start, to):
    x_mid = (start[0] + to[0]) / 2
    y_mid = (start[1] + to[1]) / 2
    createPlot.ax1.text(x_mid, y_mid, txt, va='center', ha='center', rotation=30)


def plotTree(parent, tree, f, t, level):
    level = min(level, 5)
    if isinstance(tree, float):
        plotNode(str(tree), f, t, leafNode)
        plotArr(parent['val'], f, t)
    else:
        plotNode(tree['label'], f, t, decisionNode)
        if parent != tree:
            plotArr(parent['val'], f, t)
        plotTree(tree, tree['left'], t, (t[0] - 0.3 / 2 ** level, t[1] - 0.1), level + 1)
        plotTree(tree, tree['right'], t, (t[0] + 0.3 / 2 ** level, t[1] - 0.1), level + 1)


def createPlot(tree):
    fig = plt.figure(figsize=[18, 8], facecolor='white')
    fig.clf()
    createPlot.ax1 = plt.subplot(111, frameon=False)
    plotTree.center = (0, 0)
    plotTree(tree, tree, (0.5, 1.0), (0.5, 1.0), 0)
    plt.show()
