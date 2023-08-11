# Kenny Schlegel, Peer Neubert, Peter Protzel
#
# HDC-MiniROCKET : Explicit Time Encoding in Time Series Classification with Hyperdimensional Computing
# Copyright (C) 2022 Chair of Automation Technology / TU Chemnitz


import numpy as np
from config import *
from data.dataset_utils import *
from time import time
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import classification_report, f1_score, confusion_matrix
import logging
from models.Minirocket_utils.minirocket_multivariate import MiniRocketMultivariate

# config logger
logger = logging.getLogger('log')

class MINIROCKET_model:
    def __init__(self,config):
        # params
        self.eval_mod = None
        self.config = config

    def train_model(self,X_train,y_train,X_test,y_test,fold=None):
        # transformation
        self.rocket = MiniRocketMultivariate(random_state = self.config.seed)
        t_proc = []
        # time measurement
        for i in range(self.config.n_time_measures):
            self.rocket.fit(X_train)
            t = time()
            self.X_train_transform = self.rocket.transform(X_train)
            t_proc.append((time() - t))
        self.train_preproc = np.median(t_proc)

        # classifying
        self.classifier = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10), normalize=True)
        t_train = []
        # time measurement
        for i in range(self.config.n_time_measures):
            t = time()
            self.classifier.fit(self.X_train_transform, y_train)
            t_train.append(time() - t)
        self.training_time = np.median(t_train)

    def eval_model(self,X_train,y_train,X_test,y_test,fold=None):
        t_proc = []
        # time measurement
        for i in range(self.config.n_time_measures):
            t = time()
            self.X_test_transform = self.rocket.transform(X_test)
            t_proc.append((time() - t))
        self.test_preproc = np.median(t_proc)

        # time measurement
        t_test = []
        for i in range(self.config.n_time_measures):
            t = time()
            pred = self.classifier.predict(self.X_test_transform)
            t_test.append(time() - t)
        self.testing_time = np.median(t_test)

        # evaluate the results
        logger.info('Results on test data: ')
        report = classification_report(y_test.astype(int), pred, output_dict=True)
        logger.info(classification_report(y_test.astype(int), pred))

        scores = self.classifier.decision_function(self.X_test_HDC)

        # print time results
        logger.info("Data preprocessing time: " + str(self.train_preproc) + ' + ' + str(self.test_preproc) + ' = ' + str(self.train_preproc + self.test_preproc))
        logger.info("Training time: " + str(self.training_time))
        logger.info("Evaluation: " + str(self.testing_time))

        return pred, scores


