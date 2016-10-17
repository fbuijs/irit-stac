from __future__ import print_function
from os import path as fp
import os
import sys
from attelo.io import (load_multipack)
from gather import extract_features
from sklearn.externals import joblib
from attelo.io import Torpor
from attelo.learning.interface import AttachClassifier
from sklearn import linear_model
from sklearn import ensemble
import sklearn.metrics as metrics
from sklearn import cross_validation
import scipy as sc
from attelo.table import Graph
import numpy as np
import random
import matplotlib.pyplot as plt
from attelo.decoding.mst import MstDecoder
from attelo.decoding.mst import MstRootStrategy
from attelo.parser.interface import Parser
from ..ilp import ILPDecoder
from collections import OrderedDict
from copy import copy
import time
from scipy import sparse
from copy import deepcopy
from matplotlib.backends.backend_pdf import PdfPages
from attelo.decoding.baseline import (LastBaseline)
from attelo.learning import (SklearnAttachClassifier)
from attelo.parser.attach import (AttachPipeline)
from scipy.sparse import csr_matrix
from attelo.score import score_edges
from attelo.report import CombinedReport
from attelo.report import EdgeReport
from pympler.tracker import SummaryTracker
import gc

tracker = SummaryTracker()

np.seterr(divide='ignore', invalid='ignore')

NAME = 'structuredLearning'

TINY_TRAINING_CORPUS = 'data/tiny'
TRAINING_FEATURES = 'TMP/latest'
TEST_FEATURES = 'TMP/latest'
TRAIN_PREFIX = fp.join(TRAINING_FEATURES, 'training-2015-05-30')
TEST_PREFIX = fp.join(TEST_FEATURES, 'test-2015-05-30')
FIGURES_FOLDER = '../figures/'
begin = time.time()


def config_argparser(psr):
    """
    Subcommand flags.

    You should create and pass in the subparser to which the flags
    are to be added.
    """
    psr.add_argument("--skip-training",
                     default=False, action="store_true",
                     help="only gather test data")
    psr.add_argument('--strip-mode',
                     choices=['head', 'broadcast', 'custom'],
                     default='head',
                     help='CDUs stripping method')
    psr.set_defaults(func=main)


def printWithTime(*args, **kwargs):
    args += ("          ",)
    args += (time.strftime("%Hh %Mmin %Ss", time.gmtime(time.time() - begin)),)
    print(*args, **kwargs)


def printScore(scores, scoringType):
    """
    Print the score on the cross validation

    Parameters
    ----------
    scores : cross validation scores
    scoringType : option given for the cross validation (precision / recall / f1 / ...)
    """

    print(scoringType, ":", round(scores, 2))


def collectPairs(mpack, multipack=True):
    """
    Collect all pairs and associated features and label in mpack

    Parameters
    ----------
    mpack : multipack

    Returns
    -------
    featsArray : numpy array with all the features
    labelArray : numpy array with all label (-1; 1)
    """
    feats = None
    label = []

    # Collect all pairs' label and features
    if multipack:
        for dpack in mpack.values():
            if feats is None:
                feats = dpack.data
            else:
                feats = sc.sparse.vstack((feats, dpack.data), format='csr')
            label.extend(((dpack.target != 2)))
    else:
        feats = mpack.data
        label.extend(((mpack.target != 2)))

    labelArray = np.asarray(label, dtype=np.int16)

    return feats, labelArray


def collectTargetLabels(mpack):
    """
    Collect all pairs and associated features and label in mpack

    Parameters
    ----------
    mpack : multipack

    Returns
    -------
    labelArray : numpy array with all label (-1; 1)
    """
    label = []

    # Collect all pairs' label and features
    for dpack in mpack.values():
        label.extend(((dpack.target != 2)))

    labelArray = np.asarray(label, dtype=np.int16)
    return labelArray


def cleanFeatures(featsTrain, featsTest, maxCorrelation=1, minVariance=0, title='Eigenvalues corresponding to features',
                  fileName=None):
    # type: (object, object, object, object, object, object) -> object
    """
    Delete all features that are correlated or that have low variance

    Parameters
    ----------
    featsArray : list of all features on the examples
    maxCorrelation : maximum correlation between 2 features
    minVariance : minimum variance on every features

    Returns
    -------
    featsArray : cleaned features

    """
    with Torpor("Cleaning features"):
        if fileName is not None:
            XtX = np.dot(featsTrain.transpose(), featsTrain)
            eig = np.log10(np.real(np.linalg.eigvals(XtX)))
            plt.plot(eig, label="No filtering")

        # delete high correlation features
        res = np.corrcoef(featsTrain.transpose())
        res = np.absolute(res)

        index = np.where(res >= maxCorrelation)

        index = [index[0][x] for x in range(len(index[0])) if index[0][x] != index[1][x]]
        index = list(set(index))
        index = np.asarray(index)

        featsTrain = np.delete(featsTrain.transpose(), np.s_[index], 0)
        featsTest = np.delete(featsTest.transpose(), np.s_[index], 0)

        if fileName is not None:
            XtX = np.dot(featsTrain, featsTrain.transpose())
            eig = np.log10(np.real(np.linalg.eigvals(XtX)))
            plt.plot(eig, label="Correlation filtering")

        # delete high correlation features
        res = np.var(featsTrain.transpose(), axis=0)
        res = np.absolute(res)

        index = np.where(res <= minVariance)
        featsTrain = np.delete(featsTrain, np.s_[index[0]], 0)
        featsTest = np.delete(featsTest, np.s_[index[0]], 0)

        if fileName is not None:
            XtX = np.dot(featsTrain, featsTrain.transpose())
            eig = np.log10(np.real(np.linalg.eigvals(XtX)))
            plt.plot(eig, label="Variance filtering")
            plt.title(title)
            plt.xlabel('Eigenvalues')
            plt.ylabel('Values')
            plt.grid(True)
            plt.legend(loc=1, borderaxespad=0.)

            plt.savefig(fileName)
            plt.close()

    return featsTrain.transpose(), featsTest.transpose()


def getAUPR(mpackTrain, mpackTest, model, varMin, varMax, num, title='AUPR en fonction de alpha',
            fileName=None, folder='', param='alpha', proba=False):
    """
    This function calculate and returns AUPRs for alpha in [alphaMin ; alphaMax] with num values in between

    Parameters
    ----------
    featsArray
    labelArray
    alphaMin
    alphaMax
    num
    title
    fileName

    Returns
    -------

    """

    with Torpor("Learning and Scoring Linear Regression"):
        vars = np.logspace(np.log10(varMin), np.log10(varMax), num=num)
        area = np.zeros(num)
        P = np.zeros(num)
        R = np.zeros(num)
        label = collectTargetLabels(mpackTest)
        learner = Learner(model)

        for j in range(len(vars)):
            print("\rLearning and Scoring Linear Regression... [", j + 1, "/", len(vars), "] ", sep='', end='')
            sys.stdout.flush()

            # Create the linear models
            kwargs = {param: vars[j]}
            learner._instantiateLearner(model, **kwargs)

            # Save the linear model
            # joblib.dump(logisticRegression, '/home/fbuijs/filename.pkl')

            # Learn and evaluate scores using cross validation
            learner.fit(mpackTrain)

            scores = learner.predict_score(mpackTest, proba=proba)

            precision, recall, _ = metrics.precision_recall_curve(label, scores)

            precision = [x for (y, x) in sorted(zip(recall, precision))]
            recall.sort()
            # plt.plot(recall, precision)
            area[j] = metrics.auc(recall, precision)

    if fileName is not None:
        plt.semilogx(vars, area)
        plt.title(title)
        plt.xlabel('Alpha')
        plt.ylabel('AUPR')

        plt.savefig(folder + fileName)
        plt.close()

        plt.figure()
        plt.semilogx(vars, P, label="Precision")
        plt.semilogx(vars, R, label="Recall")
        plt.title('Precision and Recall')
        plt.xlabel('Alpha')
        plt.ylabel('PR')
        plt.grid(True)
        plt.legend(loc=1, borderaxespad=0.)

        plt.savefig(folder + 'PR_' + fileName)
        plt.close()

    return area, vars


def message(title, message):
    os.system('notify-send -i /home/fbuijs/logo-irit.png "' + title + '" "' + message + '"')

#######################################################################################
# Functions for structured learning

def loss(y1, y2):
    return np.sum(np.multiply(1*y1, 1*np.logical_not(y2)))


def memory_usage_resource():
    import resource
    rusage_denom = 1024.
    if sys.platform == 'darwin':
        # ... it seems that in OSX the output is different units ...
        rusage_denom = rusage_denom * rusage_denom
    mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / rusage_denom
    return mem

def phi(dpack, pred):
    data = dpack.data
    y = np.asarray(pred == 2)

    j = 0
    i = 0
    l = 0
    while j < len(y):
        while j < len(y) and y[j]:
            j += 1

        if j > 0:
            data = sparse.vstack([data[:(i-l), :], data[(j-l):, :]])
            l += j-i

        while j < len(y) and not y[j]:
            j += 1
        i = j

    return np.asarray(np.sum(data.todense(), axis=0))[0]


def predict(learner, dpack, w):
    data = dpack.data
    score = []
    for j in range(len(np.asarray(data.todense())[:, 0])):
        score.append(np.dot(w, data[j, :].todense().transpose())[0, 0])

    dpack = learner.multiply(dpack, attach=np.asarray(score))

    return learner.decoder_.decode(dpack)


def Hi(learner, dpack, target, w):
    data = dpack.data
    yi = (target == 2)
    score = []
    diff = []

    for j in range(len(np.asarray(data.todense())[:, 0])):
        score.append((np.dot(w, data[j, :].todense().transpose()) + (1 * (yi[j])))[0, 0])
        diff.append((np.dot(w, data[j, :].todense().transpose()) * (1 - (1*yi[j])))[0, 0])

    dpack = learner.multiply(dpack, attach=np.asarray(score))

    resDpack = learner.decoder_.decode(dpack)

    # TODO get the value 'res' from the decoder directly
    res = np.sum(np.multiply(np.asarray(score), 1 * (resDpack.graph[0] != 2)))
    res = res - np.sum(diff)

    return res, resDpack


def F(dpack, w, pred):
    data = dpack.data
    feats = []
    for i in range(len(np.asarray(data.todense())[:, 0])):
        feats.append(np.dot(w, data[i, :].todense().transpose()) * (1*(pred[i] != 2)))

    return np.sum(feats)


def psi(dpack, target, w):
    return F(dpack, w, target) - F(dpack, w, dpack.graph[0])


def shuffled(list):
    random.shuffle(list)
    return list


class Learner(Parser):
    def __init__(self, learner=None, alpha=None, C=None, n_estimators=None, decoder=None, decoder2 = None, Lambda=None):
        # declare attributes
        self.lambda_ = Lambda
        self.learner_ = learner
        self.decoder_ = decoder
	if decoder2 is None:
	    self.decoder2_ = decoder
	else:
	    self.decoder2_ = decoder2
        self.w_ = None
        self.wBar_ = None
        self.mpackTrain_ = None
        self.labelTrain_ = None
        self.featsTrain_ = None
        self.mpackTest_ = None
        self.featsTest_ = None

        # init
        if learner is not None:
            self._instantiateLearner(learner, alpha=alpha, C=C, n_estimators=n_estimators)

    def fit(self, mpackTrain, keyWeight=False, newAlgo = True):

        if self.learner_ is not None:
            if self.mpackTrain_ is not mpackTrain:
                self.mpackTrain_ = mpackTrain
                self.featsTrain_, self.labelTrain_ = collectPairs(mpackTrain)

            if keyWeight:
                nbPos = np.sum(self.labelTrain_ > 0)
                nbNeg = len(self.labelTrain_) - nbPos
                wPos = np.ones(len(self.labelTrain_)) * float(nbNeg / len(self.labelTrain_))
                wNeg = np.ones(len(self.labelTrain_)) * float(nbPos / len(self.labelTrain_))
                wPos *= self.labelTrain_ > 0
                wNeg *= self.labelTrain_ <= 0
                weight = wPos + wNeg + 0.00001
            else:
                weight = None

            self.learner_.fit(self.featsTrain_, self.labelTrain_, sample_weight=weight)

        elif not newAlgo:
            length = len(mpackTrain.values())
            self.w_ = np.zeros(5384)
            # /!\ n must be a float
            n = 1000.0
            l = 0.0
            K = 5000
            k = 0
            length *= K
            s = []
            dualGap = []

            while k < K:
                try:
                    k += 1
                    i = 0
                    ws = np.zeros(5384)
                    ls = 0.0
                    hi = 0.0
                    print("\rLearning [", k, "/", K, "]", end='', sep='')
                    sys.stdout.flush()

                    for dpack in mpackTrain.values():
                        i += 1
                        if i > n:
                            break

                        dpack2 = self.multiply(dpack, attach=1 * (dpack.target != 2))

                        targetpack = self.decoder_.decode(dpack2)

                        target = targetpack.graph[0]

                        resHi, LAD = Hi(self, dpack, target, self.w_)
                        ls += loss((LAD.graph[0] != 2), (target != 2))
                        ws += (phi(LAD, target) - phi(LAD, LAD.graph[0]))

                        hi += resHi

                    ws /= (n * self.lambda_)
                    ls /= n

                    dualGap.append(self.lambda_ * np.dot((self.w_ - ws).transpose(), self.w_) - l + ls)
                    gamma = dualGap[-1] / (self.lambda_ * (np.linalg.norm(self.w_ - ws) ** 2))

                    if gamma > 1:
                        gamma = 1
                    elif gamma < 0:
                        print("error gamma < 0")
                        print("gamma =", gamma)

                    self.w_ = (1 - gamma) * self.w_ + gamma * ws
                    l = (1 - gamma) * l + gamma * ls

                    s.append(self.lambda_ / 2 * np.linalg.norm(self.w_) ** 2 + (hi / n))

                    if dualGap[-1] / s[-1] < 1e-2:
                        break

                except KeyboardInterrupt:

                    return s, self.w_, dualGap

            return s, self.w_, dualGap

        else:
            random.seed(4)
            length = len(mpackTrain.values())
            self.w_ = np.zeros(5384)
            self.wBar_ = np.zeros(5384)
            # /!\ n must be a float
            n = 968.0
            l = 0.0
            lGap = 0.0
            K = 5000
            k = -1
            s = []
            wi = np.zeros([5384, 968])
            li = np.zeros(968)
            dualGap = []
            targets = []

            for dpack in mpackTrain.values():
                dpack2 = self.multiply(dpack, attach=1 * (dpack.target != 2))
                targetpack = self.decoder2_.decode(dpack2)
                targets.append(targetpack.graph[0])

            begin = time.time()

            while k < K:
                try:
                    index = shuffled(range(length))
                    k += 1
                    m = 0
                    hi = 0.0
                    ls = 0.0
                    printWithTime("\rLearning [", k, "/", K, "]", end='', sep='')
                    sys.stdout.flush()

                    for i in index:
                        dpack = mpackTrain.values()[i]
                        # print(dpack)
                        m += 1
                        if m > n:
                            break

                        resHi, LAD = Hi(self, dpack, targets[i], self.w_)
                        ls = loss((LAD.graph[0] != 2), (targets[i] != 2))
                        ws = (phi(LAD, targets[i]) - phi(LAD, LAD.graph[0]))

                        ws /= (n * self.lambda_)
                        ls /= n

                        if (wi[:, i] - ws).any():
                            gamma = self.lambda_ * np.dot((wi[:, i] - ws).transpose(), self.w_) - li[i] + ls
                            gamma /= (self.lambda_ * (np.linalg.norm(wi[:, i] - ws) ** 2))

                            if gamma > 1:
                                gamma = 1
                            elif gamma < 0:
                                gamma = 0

                            wi_old = copy(wi[:, i])
                            li_old = copy(li[i])

                            wi[:, i] = (1 - gamma) * wi[:, i] + gamma * ws
                            li[i] = (1 - gamma) * li[i] + gamma * ls

                            self.w_ = self.w_ + wi[:, i] - wi_old
                            l = l + li[i] - li_old

                            v = 0.0
                            self.wBar_ = v * self.wBar_ + (1-v) * self.w_
                            # print(np.linalg.norm(self.w_))

                    if k % 10 == 0:
                        printWithTime("\rLearning [", k, "/", K, "]  Calculating duality gap", end='', sep='')
                        sys.stdout.flush()
                        j = 0
                        for dpack, target in zip(mpackTrain.values(), targets):
                            j += 1
                            printWithTime("\rLearning [", k, "/", K, "]  Calculating duality gap [", j, "/", length, "]",
                                  end='', sep='')
                            sys.stdout.flush()

                            resHi, LAD = Hi(self, dpack, target, self.wBar_)
                            ls += loss((LAD.graph[0] != 2), (target != 2))
                            ws += (phi(LAD, target) - phi(LAD, LAD.graph[0]))

                            hi += resHi

                        ws /= (968.0 * self.lambda_)
                        ls /= 968.0

                        dualGap.append(self.lambda_ * np.dot((self.wBar_ - ws).transpose(), self.wBar_) - l + ls)

                        s.append(self.lambda_ / 2 * np.linalg.norm(self.wBar_) ** 2 + (hi / 968.0))
                        printWithTime("\rk = ", k+1, "  &  Duality Gap = ", dualGap[-1], "  &  score = ", s[-1],
                              sep='')
                        gc.collect()
                        print("Memory usage =", memory_usage_resource())
                        tracker.print_diff()
                        if abs(dualGap[-1] / s[-1]) < 1e-2:
                            print()
                            break

                except KeyboardInterrupt:

                    return s, self.wBar_, dualGap

            return s, self.wBar_, dualGap

    def predict_score(self, mpackTest, decoder=None, proba=False, dense=False, test=False):

        if test:
            res = []
            for dpack in mpackTest.values():
                dpack2 = self.multiply(dpack, attach=1 * (dpack.target != 2))

                res.append(self.decoder_.decode(dpack2))

            return res

        elif self.decoder_ is None and decoder is None:
            if self.mpackTest_ is not mpackTest:
                self.featsTest_, _ = collectPairs(mpackTest)
                self.mpackTest_ = mpackTest

            if dense and not proba:
                res = self.learner_.predict(self.featsTest_.toarray())
            elif dense and proba:
                res = self.learner_.predict_proba(self.featsTest_.toarray())[:, 1]
                # res = np.log10(np.divide(res, 1-res))
            elif proba and not dense:
                res = self.learner_.predict_proba(self.featsTest_)[:, 1]
                # res = np.log10(np.divide(res, 1 - res))
            elif not proba and not dense:
                res = self.learner_.predict(self.featsTest_)

        elif self.w_ is None:
            if self.decoder_ is None:
                self.decoder_ = decoder
            res = []
            for dpack in mpackTest.values():
                featsTest, _ = collectPairs(dpack, multipack=False)
                if dense and proba:
                    scores = self.learner_.predict_proba(featsTest.toarray())[:, 1]
                elif proba and not dense:
                    scores = self.learner_.predict_proba(featsTest)[:, 1]
                elif dense and not proba:
                    scores = self.learner_.predict(featsTest.toarray())
                elif not proba and not dense:
                    scores = self.learner_.predict(featsTest)

                dpack = self.multiply(dpack, attach=scores)

                res.append(decoder.decode(dpack))

        else:
            res = []
            if self.decoder_ is None:
                self.decoder_ = decoder
            for dpack in mpackTest.values():
                res.append(predict(self, dpack, self.w_))

        return res

    def transform(self, dpack):
        pass

    def _instantiateLearner(self, learner, alpha=None, C=None, n_estimators=None):
        if alpha is not None:
            self.learner_ = learner(alpha=alpha)
        elif C is not None:
            self.learner_ = learner(C=C)
        elif n_estimators is not None:
            self.learner_ = learner(n_estimators=n_estimators)
        else:
            self.learner_ = learner()


def main(args):
    """
    Load tiny corpus, extract features and labels, and learn the presence of semantic relation

    Parameters
    ----------
    args : not used yet (but necessary)
    """

    # load the data into a multipack
    mpackTrain = load_multipack(TRAIN_PREFIX + '.relations.sparse.edu_input',
                                TRAIN_PREFIX + '.relations.sparse.pairings',
                                TRAIN_PREFIX + '.relations.sparse',
                                TRAIN_PREFIX + '.relations.sparse.vocab',
                                verbose=True)

    mpackTest = load_multipack(TEST_PREFIX + '.relations.sparse.edu_input',
                               TEST_PREFIX + '.relations.sparse.pairings',
                               TEST_PREFIX + '.relations.sparse',
                               TEST_PREFIX + '.relations.sparse.vocab',
                               verbose=True)

    modelRidge = linear_model.Ridge
    modelLogistic = linear_model.LogisticRegression
    modelLinear = linear_model.LinearRegression
    modelRandomForestRegressor = ensemble.RandomForestRegressor
    modelGradientBoost = ensemble.GradientBoostingClassifier
    modelGradientBoostRegressor = ensemble.GradientBoostingRegressor

    decoderMST = MstDecoder(MstRootStrategy.fake_root, use_prob=False)
    decoder = ILPDecoder()

    label = collectTargetLabels(mpackTest)

    LAM = np.logspace(np.log10(6e-2), np.log10(.5), num=5)

    for lam in LAM:
        print("------------------------------------------------------")
        print("Lambda =", lam)
	# create learner
        learner = Learner(decoder=decoderMST, decoder2=decoder, Lambda=lam)

        s, w, dualGap = learner.fit(mpackTrain)

        arbre = predict(learner, mpackTrain.values()[5], w)
        print()

        print(arbre.graph[0])

        dpack2 = learner.multiply(mpackTrain.values()[5], attach=1 * (mpackTrain.values()[5].target != 2))

        targetpack = learner.decoder_.decode(dpack2)

        target = targetpack.graph[0]
        print(target)
        print(arbre.target)

        plt.plot(s, label="score")
        plt.plot(dualGap, label="duality gap")
        plt.plot(np.asarray(s) - np.asarray(dualGap), label="difference")

        plt.title("minimization")
        plt.xlabel('k')
        plt.ylabel('score')
        plt.grid(True)
        plt.legend(loc=1, borderaxespad=0.)

        plt.savefig(FIGURES_FOLDER + 'test'+str(round(lam, 5))+'.pdf')
        plt.close()

        np.savetxt('w_'+str(round(lam, 5))+'.out', w, delimiter=',')

        # test = np.loadtxt('test.out', delimiter=',')

        print()
        res = learner.predict_score(mpackTest)
        pred = []
        for graph in res:
            pred.extend(graph.graph[0] != 2)

        vote = []
        vote.append(1 * pred)

        print("f1 score =", round(metrics.f1_score(label, 1 * pred), 2))

        P = metrics.precision_score(label, pred)
        R = metrics.recall_score(label, pred)

        print("Precision =", round(P, 2))
        print("Recall =", round(R, 2))
        print()
        message("IRIT-STAC", "Travail termine")
    exit(1)

    """
    learner = SklearnAttachClassifier(modelLogistic())
    decoder = MstDecoder(MstRootStrategy.fake_root)
    parser1 = AttachPipeline(learner=learner,
                             decoder=decoder)

    # train the parser
    train_dpacks = mpackTrain.values()
    train_targets = [x.target for x in train_dpacks]
    parser1.fit(train_dpacks, train_targets)

    # now run on a test pack
    dpack = []
    for i in range(len(mpackTest.values())):
        dpack.extend(parser1.transform(mpackTest.values()[i]))
    # print_results(dpack)"""

    """
    # Clean features with high correlation or low variance
    featsTrain_filtered_1, featsTest_filtered_1 = cleanFeatures(featsTrain, featsTest,
                                                                fileName=FIGURES_FOLDER + 'minFilter.pdf')
    featsTrain_filtered_2, featsTest_filtered_2 = cleanFeatures(featsTrain, featsTest, maxCorrelation=0.98,
                                                                minVariance=0.005,
                                                                fileName=FIGURES_FOLDER + 'maxFilter.pdf')
    featsTrain_filtered_3, featsTest_filtered_3 = cleanFeatures(featsTrain, featsTest, maxCorrelation=0.97,
                                                                fileName=FIGURES_FOLDER + 'filter.pdf')"""

    # getAUPR(mpackTrain, mpackTest, modelRidge, 1e3, 1e4, 50, fileName='noFilter_ridge_2_50pts.pdf', folder=FIGURES_FOLDER)
    # exit(1)
    # getAUPR(mpackTrain, mpackTest, modelLogistic, 1e-3, 1e-1, 50, fileName='noFilter_3_50pts.pdf', folder=FIGURES_FOLDER, param='C', proba=True)
    # exit(1)


    ########################################################################
    with Torpor("Learning and Scoring Linear Regression Ridge"):
        learner = Learner(learner=modelRidge, alpha=3500)
        learner.fit(mpackTrain)

        scores = learner.predict_score(mpackTest)

        label = collectTargetLabels(mpackTest)

        precision, recall, seuil = metrics.precision_recall_curve(label, scores)

        f1_score = [0]
        for s in seuil:
            f1 = metrics.f1_score(label, 1 * (scores > s))
            f1_score.append(f1)

        precision = [x for (y, x) in sorted(zip(recall, precision))]
        f1_score = [x for (y, x) in sorted(zip(recall, f1_score))]
        recall.sort()
        # plt.plot(recall, precision)
        area = metrics.auc(recall, precision)
        print("Area =", area)

        plt.plot(recall, precision, label="PR")
        plt.plot(recall, f1_score, label="F1 score")
        precision = np.asarray(precision)
        recall = np.asarray(recall)

        print("max f1_score =", max(f1_score))
        print("max precision =", precision[np.argwhere(f1_score == max(f1_score))])
        print("max recall =", recall[np.argwhere(f1_score == max(f1_score))])

        res = learner.predict_score(mpackTest, decoder=decoder)
        pred = []
        for graph in res:
            pred.extend(graph.graph[0] != 2)

        vote = []
        vote.append(1 * pred)

        plt.title("Linear Regression Ridge AUPR=0.41\n F1 score with decoder :" + str(
            round(metrics.f1_score(label, 1 * pred), 2)))
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.grid(True)

        P = metrics.precision_score(label, pred)
        R = metrics.recall_score(label, pred)

        R_not_decoded = min(recall, key=lambda x: abs(x - R))
        P_not_decoded = max(precision[np.argwhere(recall == R_not_decoded)])
        print("Precision =", P)
        print("Recall =", R)
        print("Precision before decoding =", P_not_decoded)

    plt.plot(R, P, 'ro')

    plt.annotate('with decoder', xy=(R + .005, P + .005), xytext=(.8, .7),
                 arrowprops=dict(facecolor='black', shrink=0.02))

    plt.legend(loc=1, borderaxespad=0.)

    plt.savefig(FIGURES_FOLDER + "Ridge.pdf")
    plt.close()

    ##########################################################################################
    with Torpor("Learning and Scoring Linear Regression"):
        learner._instantiateLearner(modelLinear)
        learner.fit(mpackTrain)

        scores = learner.predict_score(mpackTest)

        label = collectTargetLabels(mpackTest)

        precision, recall, seuil = metrics.precision_recall_curve(label, scores)

        f1_score = [0]
        for s in seuil:
            f1_score.extend([metrics.f1_score(label, 1 * (scores > s))])

        precision = [x for (y, x) in sorted(zip(recall, precision))]
        f1_score = [x for (y, x) in sorted(zip(recall, f1_score))]
        recall.sort()
        # plt.plot(recall, precision)
        area = metrics.auc(recall, precision)
        print("Area =", area)

        plt.plot(recall, precision, label="PR")
        plt.plot(recall, f1_score, label="F1 score")
        precision = np.asarray(precision)
        recall = np.asarray(recall)

        print("max f1_score =", max(f1_score))
        print("max precision =", precision[np.argwhere(f1_score == max(f1_score))])
        print("max recall =", recall[np.argwhere(f1_score == max(f1_score))])

        res = learner.predict_score(mpackTest, decoder=decoder)
        pred = []

        for graph in res:
            pred.extend(graph.graph[0] != 2)

        vote.append(1 * pred)

        plt.title(
            "Linear Regression AUPR=0.33\n F1 score with decoder :" + str(round(metrics.f1_score(label, 1 * pred), 2)))
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.grid(True)

    P = metrics.precision_score(label, pred)
    R = metrics.recall_score(label, pred)

    R_not_decoded = min(recall, key=lambda x: abs(x - R))
    P_not_decoded = max(precision[np.argwhere(recall == R_not_decoded)])
    print("Precision =", P)
    print("Recall =", R)
    print("Precision before decoding =", P_not_decoded)
    plt.plot(R, P, 'ro')

    plt.annotate('with decoder', xy=(R + .005, P + .005), xytext=(.8, .7),
                 arrowprops=dict(facecolor='black', shrink=0.02))

    plt.legend(loc=1, borderaxespad=0.)
    plt.savefig(FIGURES_FOLDER + "RegLinear.pdf")
    plt.close()

    #############################################################################################
    with Torpor("Learning and Scoring Logistic Regression"):
        learner = Learner(learner=modelLogistic)
        learner.fit(mpackTrain)

        scores = learner.predict_score(mpackTest, proba=True)

        label = collectTargetLabels(mpackTest)

        precision, recall, seuil = metrics.precision_recall_curve(label, scores)

        f1_score = [0]
        for s in seuil:
            f1_score.extend([metrics.f1_score(label, 1 * (scores > s))])

        precision = [x for (y, x) in sorted(zip(recall, precision))]
        f1_score = [x for (y, x) in sorted(zip(recall, f1_score))]
        recall.sort()
        # plt.plot(recall, precision)
        area = metrics.auc(recall, precision)
        print("Area =", area)

        plt.plot(recall, precision, label="PR")
        plt.plot(recall, f1_score, label="F1 score")
        precision = np.asarray(precision)
        recall = np.asarray(recall)

        print("max f1_score =", max(f1_score))
        print("max precision =", precision[np.argwhere(f1_score == max(f1_score))])
        print("max recall =", recall[np.argwhere(f1_score == max(f1_score))])

        res = learner.predict_score(mpackTest, decoder=decoderMST, proba=True)
        pred = []

        for graph in res:
            pred.extend(graph.graph[0] != 2)

        vote.append(1 * pred)

        plt.title(
            "Logistic Regression AUPR=0.33\n F1 score with decoder :" + str(
                round(metrics.f1_score(label, 1 * pred), 2)))
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.grid(True)

    P = metrics.precision_score(label, pred)
    R = metrics.recall_score(label, pred)

    R_not_decoded = min(recall, key=lambda x: abs(x - R))
    P_not_decoded = max(precision[np.argwhere(recall == R_not_decoded)])
    print("Precision =", P)
    print("Recall =", R)
    print("Precision before decoding =", P_not_decoded)
    plt.plot(R, P, 'ro')

    plt.annotate('with decoder', xy=(R + .005, P + .005), xytext=(.8, .7),
                 arrowprops=dict(facecolor='black', shrink=0.02))

    plt.legend(loc=1, borderaxespad=0.)
    plt.savefig(FIGURES_FOLDER + "RegLogistic.pdf")
    plt.close()

    #########################################################################################
    with Torpor("Learning and Scoring Gradient Boost Regression"):
        learner._instantiateLearner(modelGradientBoostRegressor)
        learner.fit(mpackTrain)

        scores = learner.predict_score(mpackTest, dense=True)

        precision, recall, seuil = metrics.precision_recall_curve(label, scores)

        f1_score = [0]
        for s in seuil:
            f1_score.extend([metrics.f1_score(label, 1 * (scores > s))])

        precision = [x for (y, x) in sorted(zip(recall, precision))]
        f1_score = [x for (y, x) in sorted(zip(recall, f1_score))]
        recall.sort()
        # plt.plot(recall, precision)
        area = metrics.auc(recall, precision)
        print("Area =", area)

        plt.plot(recall, precision, label="PR")
        plt.plot(recall, f1_score, label="F1 score")
        precision = np.asarray(precision)
        recall = np.asarray(recall)

        print("max f1_score =", max(f1_score))
        print("max precision =", precision[np.argwhere(f1_score == max(f1_score))])
        print("max recall =", recall[np.argwhere(f1_score == max(f1_score))])

        res = learner.predict_score(mpackTest, decoder=decoder, dense=True)
        pred = []

        for graph in res:
            pred.extend(graph.graph[0] != 2)

        vote.append(1 * pred)

        plt.title("Gradient Boost Regression AUPR=0.69\n F1 score with decoder :" + str(
            round(metrics.f1_score(label, 1 * pred), 2)))
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.grid(True)

    P = metrics.precision_score(label, pred)
    R = metrics.recall_score(label, pred)

    R_not_decoded = min(recall, key=lambda x: abs(x - R))
    P_not_decoded = max(precision[np.argwhere(recall == R_not_decoded)])
    print("Precision =", P)
    print("Recall =", R)
    print("Precision before decoding =", P_not_decoded)
    plt.plot(R, P, 'ro')

    plt.annotate('with decoder', xy=(R + .005, P + .005), xytext=(.8, .7),
                 arrowprops=dict(facecolor='black', shrink=0.02))

    plt.legend(loc=1, borderaxespad=0.)
    plt.savefig(FIGURES_FOLDER + "GradientBoost.pdf")
    plt.close()

    ####################################################################################################
    with Torpor("Learning and Scoring Random Forest Regression"):
        learner = Learner(learner=modelRandomForestRegressor, n_estimators=20)
        learner.fit(mpackTrain)

        scores = learner.predict_score(mpackTest, dense=True)

        label = collectTargetLabels(mpackTest)

        precision, recall, seuil = metrics.precision_recall_curve(label, scores)

        print("len(seuil) =", len(seuil))
        print("seuil =", seuil)

        f1_score = [0]
        for s in seuil:
            f1_score.append(metrics.f1_score(label, 1 * (scores > s)))

        precision = [x for (y, x) in sorted(zip(recall, precision))]
        f1_score = [x for (y, x) in sorted(zip(recall, f1_score))]
        recall.sort()
        # plt.plot(recall, precision)
        area = metrics.auc(recall, precision)
        print("Area =", area)

        plt.plot(recall, precision, label="PR")
        plt.plot(recall, f1_score, label="F1 score")
        precision = np.asarray(precision)
        recall = np.asarray(recall)

        print("max f1_score =", max(f1_score))
        print("max precision =", precision[np.argwhere(f1_score == max(f1_score))])
        print("max recall =", recall[np.argwhere(f1_score == max(f1_score))])

        res = learner.predict_score(mpackTest, decoder=decoderMST, dense=True)
        pred = []

        for graph in res:
            pred.extend(graph.graph[0] != 2)

        vote.append(1 * pred)

        plt.title("Random Forest Regression AUPR=0.72\n F1 score with decoder :" + str(
            round(metrics.f1_score(label, 1 * pred), 2)))
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.grid(True)

    P = metrics.precision_score(label, pred)
    R = metrics.recall_score(label, pred)

    R_not_decoded = min(recall, key=lambda x: abs(x - R))
    P_not_decoded = max(precision[np.argwhere(recall == R_not_decoded)])
    print("Precision =", P)
    print("Recall =", R)
    print("Precision before decoding =", P_not_decoded)
    plt.plot(R, P, 'ro')

    plt.annotate('with decoder', xy=(R + .005, P + .005), xytext=(.8, .7),
                 arrowprops=dict(facecolor='black', shrink=0.02))

    plt.legend(loc=1, borderaxespad=0.)
    plt.savefig(FIGURES_FOLDER + "RandomForest.pdf")
    plt.close()
    # print("res =", res)

    pred = 1 * (np.mean(vote, axis=0) >= 0.4)

    print("Vote majorite\nf1_score =", round(metrics.f1_score(label, 1 * pred), 2))
    P = metrics.precision_score(label, pred)
    R = metrics.recall_score(label, pred)
    print("Precision =", P)
    print("Recall =", R)

    message("IRIT-STAC", "Travail termine")

    """
    getAUPR(featsTrain_filtered_1, featsTest_filtered_1, labelTrain, labelTest, 1e-5, 1e8, 10, fileName='minFilter.pdf',
            title='AUPR en fonction de alpha filtre minimal', folder=FIGURES_FOLDER)
    getAUPR(featsTrain_filtered_2, featsTest_filtered_2, labelTrain, labelTest, 1e-5, 1e8, 10, fileName='maxFilter.pdf',
            title='AUPR en fonction de alpha filtre \ncorrelation max 0.97 et variance min 0', folder=FIGURES_FOLDER)
    getAUPR(featsTrain_filtered_3, featsTest_filtered_3, labelTrain, labelTest, 1e-5, 1e8, 10, fileName='filter.pdf',
            title='AUPR en fonction de alpha filtre \ncorrelation max 0.98 et variance min 0.005',
            folder=FIGURES_FOLDER)"""
