import numpy as np
from scipy.stats import multivariate_normal

def generate_binary(n, dist0, dist1, bias=True):
    X = np.vstack((dist0.rvs(n), dist1.rvs(n)))
    if bias:
        X = np.hstack((np.ones((2*n,1)), X))
    y = np.concatenate([np.zeros(n, dtype=np.int), np.ones(n, dtype=np.int)])
    return X,y

# two-dimensional with bias term, binary classification, separable
np.random.seed(1)
X1, y1 = generate_binary(100,
                         multivariate_normal([3,4], np.identity(2)),
                         multivariate_normal([-2,7], np.identity(2)))

# two-dimensional with bias term, binary classification, not separable
np.random.seed(1)
X2, y2 = generate_binary(100,
                         multivariate_normal([3,4], np.identity(2)),
                         multivariate_normal([-1,5], np.identity(2)))

# one-dimensional, regression
np.random.seed(1)
X3 = np.linspace(0, 4*np.pi, 100).reshape((100,1))
y3 = np.sin(X3) + np.random.normal(0, 0.1, X3.shape[0])[:,np.newaxis]
X3test = np.linspace(0, 4*np.pi, 123).reshape((123,1))
y3test = np.sin(X3test) + np.random.normal(0, 0.1, X3test.shape[0])[:,np.newaxis]

# ## one-dimensional, regression
# f3 <- function(x) sin(x)
# x3 <- matrix(seq(0,4*pi,length.out=100), ncol=1)
# y3 <- f3(x3)+rnorm(length(x3), sd=0.1)
# x3.test <- matrix(seq(0,4*pi,length.out=1000), ncol=1)
# y3.test <- f3(x3.test)+rnorm(length(x3.test), sd=0.1)

# ## 5-dimensional, regression
# f4 <- function(x) 10*sin(pi*x[1]*x[2])+20*(x[3]-0.5)^2 +10*x[4] +5*x[5] + rnorm(1)
# n4 <- 1000
# x4 <- matrix(rnorm(5*n4), n4, 5)
# y4 <- apply(x4[,1:5], 1, f4)
# n4.test <- 100
# x4.test <- matrix(rnorm(5*n4.test), n4.test, 5)
# y4.test <- apply(x4.test[,1:5], 1, f4)

def abline(slope, intercept, color=None, label=None):
    """Plot a line from slope and intercept"""
    axes = plt.gca()
    x = np.array(axes.get_xlim())
    y = intercept + slope*x
    axes.set_autoscale_on(False)
    plt.plot(x, y, color=color, label=label)
    axes.set_autoscale_on(True)

def plot3(X, y):
    plt.scatter(X[y==0,1], X[y==0,2], c='red', label="negative")
    plt.scatter(X[y==1,1], X[y==1,2], c='green', label="positive")
    plt.legend()

def plot3db(w, color=None, label=None):
    intercept = -w[0]/w[2]
    slope = -w[1]/w[2]
    abline(slope, intercept, color, label)
    if label is not None:
        plt.legend()

def plot3dbs(X, y, n=10, maxepochs=100, pocket=False):
    N,D = X.shape
    plot3(X,y)

    mrperceptron = N
    for i in range(n):
        w = pt_train(X, y, maxepochs=maxepochs, pocket=pocket,
                     w0 = np.random.randn(3))
        label = "pocket" if pocket else "perceptron"
        plot3db(w, color="lightgray", label=label if i==0 else None)
        mrperceptron = min(mrperceptron, np.sum(np.abs(pt_classify(X,w)-y)))

    w = svm.LinearSVC(fit_intercept=False).fit(X,y).coef_[0]
    plot3db(w, label="SVM")
    mrsvm = np.sum(np.abs(pt_classify(X,w)-y))

    w = LogisticRegression(fit_intercept=False).fit(X,y).coef_[0]
    plot3db(w, label="LogReg")
    mrlogreg = np.sum(np.abs(pt_classify(X,w)-y))

    print()
    print("Misclassification rates (train)")
    print("Perceptron (best result): {:d}".format(int(mrperceptron)))
    print("Linear SVM (C=1)        : {:d}".format(int(mrsvm)))
    print("Logistic regression     : {:d}".format(int(mrlogreg)))

def plot1(X,y, label=None):
    plt.plot(X,y, linestyle=' ', marker="x", label=label)
    plt.xlabel('x')
    plt.ylabel('y')

def sigma(x):
    return 1. / (1. + np.exp(-x))

def plot1fit(X, model, label="fit", hidden=False, scale=True, alpha=0.3):
    lines = plt.plot(X, model.predict(X), label=label)

    if hidden:
        ax = plt.gca()
        if scale:
            ax2 = plt.twinx()
        else:
            ax2 = plt.twinx()
        plt.sca(ax)

        if type(model) is MLPRegressor:
            W2 = model.coefs_[1]
            b2 = model.intercepts_[1]
            W1 = model.coefs_[0]
            b1 = model.intercepts_[0]
            activation = model.get_params()['activation']
        elif type(model) is Sequential:
            W2 = model.layers[-1].get_weights()[0]
            b2 = model.layers[-1].get_weights()[1]
            W1 = model.layers[-2].get_weights()[0]
            b1 = model.layers[-2].get_weights()[1]
            activation = 'logistic' # TODO: detect others

        if activation == 'logistic':
            h = sigma(b1 + X@W1)
        elif activation == 'relu':
            h = np.fmax(b1 + X@W1, 0)
        # yhat = b2 + h@W2
        for i in range(h.shape[1]):
            label = '$h_' + str(i) + '$'
            if scale:
                lines += ax2.plot(X,h[:,i]*W2[i,0], label=label + " scaled", alpha=alpha)
            else:
                lines += ax2.plot(X,h[:,i], label=label, alpha=alpha)

    plt.legend(lines, [ l.get_label() for l in lines ])
