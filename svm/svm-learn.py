import argparse

from numpy import where
from sklearn import svm
from matplotlib import pyplot as plt
from matplotlib import cm


class Numbers:
    """
    Class to store MNIST data
    """

    def __init__(self, location):
        # You shouldn't have to modify this class, but you can if
        # you'd like.

        import cPickle, gzip

        # Load the dataset
        f = gzip.open(location, 'rb')
        train_set, valid_set, test_set = cPickle.load(f)

        self.train_x, self.train_y = train_set
        self.test_x, self.test_y = valid_set
        f.close()

        # 3 and 8 only
        idx_train = where((self.train_y==3)|(self.train_y==8))
        self.train_x = self.train_x[idx_train]
        self.train_y = self.train_y[idx_train]

        idx_test = where((self.test_y==3)|(self.test_y==8))
        self.test_x = self.test_x[idx_test]
        self.test_y = self.test_y[idx_test]


if __name__ == "__main__":
    """
    All the parameters are same name with params of SVC in scikit-learn
    http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
    """
    parser = argparse.ArgumentParser(description='SVM classifier options')
    parser.add_argument('--C', type=float, default=1.0,
            help="Penalty parameter C of the error term.")
    parser.add_argument('--degree', type=int, default=1,
            help="Degree of the polynomial kernel function (poly)")
    parser.add_argument('--kernel', type=str, default='rbf',
            help="Kernel: linear, poly, rbf, sigmoid, precomputed")
    parser.add_argument('--gamma', type=float, default=0.0,
            help="Kernel coefficient for rbf, poly and sigmoid.")
    parser.add_argument('--coef0', type=float, default=0.0,
            help="Independent term in kernel function.")
    parser.add_argument('--limit', type=int, default=-1,
            help="Restrict training to this many examples")
    parser.add_argument('--examples', type=bool, default=False,
            help="Show examples.")

    # setup
    args = parser.parse_args()
    data = Numbers("../data/mnist.pkl.gz")
    clf = svm.SVC(C=args.C,
                  kernel=args.kernel,
                  degree=args.degree,
                  coef0=args.coef0,
                  gamma=args.gamma)

    # training
    if args.limit > 0:
        print("Data limit: %i" % args.limit)
        clf.fit(data.train_x[:args.limit], data.train_y[:args.limit])
    else:
        clf.fit(data.train_x, data.train_y)

    # testing
    prediction = clf.predict(data.test_x)

    # evaluation
    accuracy = sum(prediction==data.test_y)/float(len(data.test_y)) * 100

    if args.kernel == 'linear':
        print "C: "+ repr(args.C) + ", kernel: " + args.kernel
    else:
        print "C: "+ repr(args.C) + ", kernel: " + args.kernel + ", degree: " + repr(args.degree) + ", gamma: "+ repr(args.gamma) + ", coef0: " + repr(args.coef0)

    print "accuracy: ", accuracy

    if args.examples:
        plt.subplot(221)
        idx_3 = where(data.train_y[clf.support_]==3)
        train_sv = data.train_x[clf.support_]
        im3 = plt.imshow(train_sv[idx_3[0][0]].reshape((28,28)),
                cmap=cm.gray, interpolation='nearest')
        plt.show(im3)
        plt.subplot(222)
        im3_2 = plt.imshow(train_sv[idx_3[0][10]].reshape((28,28)),
                cmap=cm.gray, interpolation='nearest')
        plt.show(im3_2)
        plt.subplot(223)
        idx_8 = where(data.train_y[clf.support_]==8)
        im8 = plt.imshow(train_sv[idx_8[0][-1]].reshape((28,28)),
                cmap=cm.gray, interpolation='nearest')
        plt.show(im8)
        plt.subplot(224)
        idx_8 = where(data.train_y[clf.support_]==8)
        im8_2 = plt.imshow(train_sv[idx_8[0][-10]].reshape((28,28)),
                cmap=cm.gray, interpolation='nearest')
        plt.show(im8_2)
