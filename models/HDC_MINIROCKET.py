# Kenny Schlegel, Peer Neubert, Peter Protzel
#
# HDC-MiniROCKET : Explicit Time Encoding in Time Series Classification with Hyperdimensional Computing
# Copyright (C) 2022 Chair of Automation Technology / TU Chemnitz


import scipy
import logging
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from numpy.fft import fft, ifft
import numpy as np
from time import time

from models.Minirocket_utils.minirocket_multivariate import MiniRocketMultivariate

# config logger
logger = logging.getLogger('log')


class HDC_MINIROCKET_model:
    '''
    model class for HDC MiniROCKET
    '''
    def __init__(self,config):
        # params
        self.eval_mod = None
        self.config = config

    def normalize(self,input):
        '''
        normalize the HDC vector representations (if it is called first time while training, it creates global
        parameters for mean and std, which are used while testing)
        @param input: HDC vectors [#samples, #dims]
        @return: normalized vectors
        '''
        try:
            self.config.hdc_m
            self.config.hdc_s
        except:
            self.config.hdc_m = np.mean(input, 0)
            self.config.hdc_s = np.std(input, 0)
        output = (input - self.config.hdc_m) / (self.config.hdc_s + 0.0000001)

        return output

    def fract_bind(self, inputs, seed_vec, scale):
        '''
        fractional binding for scalar encoding
        @param inputs: scalar input vector
        @param seed_vec: seed vector [#dim]
        @param scale: scaling of similarity
        @return: HDC vector for each scalar value
        '''

        inputs = np.array(inputs, dtype=np.float32)

        # fft on seed vector
        seed_vec = fft(seed_vec)

        exponent = inputs * scale + 1
        output = np.power(seed_vec[None, :], exponent[:, None])
        output = np.real(ifft(output))

        return output

    def create_pose_matrix(self, num_poses, scale):
        """
        function to encode poses (e.g. timestamps, or other spatial positions)
        @param num_poses:  number of scalar values (e.g. poses)
        @param scale: scale for fractional binding (similarity decrease)
        """
        np.random.seed(self.config.seed)
        # use sinc kernel
        init_vector = np.random.uniform(-np.pi, np.pi, self.config.HDC_dim)

        # create linear spaced pose values (in range [0,1] with number of steps)
        time_space = np.linspace(0, 1, num_poses)
        # fractional binding of scalar poses
        poses = self.fract_bind(time_space, init_vector, scale)

        # standardize poses results
        poses = np.transpose(poses, (1, 0))
        poses = (poses - np.mean(poses, axis=0)) / np.std(poses, axis=0)
        # save poses in config struct
        self.poses = np.transpose(poses, (1, 0)).astype(np.float32)

        return

    def hdc_rocket_tf(self,input, scale=0, compute_poses=True):
        '''
        function to encode the time series into HD vector embeddings
        @param input: input values [#samples, #channels, #timesteps]
        @param scale: scale for similarity decrease
        @param compute_poses: if set, compute pose matrix, otherwise use the one in config struct
        @return: vector embeddings of HDC-MiniRocket [#samples, #dim]
        '''
        # check if poses needs to be computed
        if compute_poses:
            # create pose matrix for time encoding
            self.create_pose_matrix(self.config.n_steps, scale)
            # load pose matrix to rocket transformer
            self.rocket.poses = self.poses.transpose((1,0))

        output = self.rocket.transform(input, use_hdc=True)
        if self.config.norm_hdc_output:
            output = self.normalize(output)

        return output

    def find_best_s(self,input, label):
        '''
        automatically select the correct similarity decrease parameter s wth cross validation
        @param input: input values [#samples, #channels, #timesteps]
        @param label: input class labels
        @return: best scaling parameter
        '''

        np.random.seed(self.config.seed)
        scales = self.config.scales

        # select only n samples per class (n is given by the complete amount of data)
        C = np.unique(label)

        idx_train = []
        idx_test = []
        idx_c = []
        for c in range(len(C)):
            idx = np.where(label == C[c])[0]
            idx_c.append((idx))
            n_samples = np.maximum(int(len(idx) * 0.8),1)

            idx_train = np.append(idx_train, idx[:n_samples]).astype(np.int)
            if n_samples == 1:
                idx_test = np.append(idx_test, idx[:]).astype(np.int)
            else:
                idx_test = np.append(idx_test, idx[n_samples:]).astype(np.int)

        # compute min number of class samples
        min_samples_per_class = 1000
        for i in range(len(idx_c)):
            if idx_c[i].shape[0]<min_samples_per_class:
                min_samples_per_class = idx_c[i].shape[0]

        # cross validation
        max_k = np.minimum(min_samples_per_class, 10)

        self.rocket.fit(input)

        classifier = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))

        acc = np.zeros((max_k,len(scales)))

        # iterate over scales
        for s_idx in range(len(scales)):
            s = scales[s_idx]
            self.config.scale = s

            # create HDC embeddings
            HDC = self.hdc_rocket_tf(input, scale=s)
            # delete computed mean and std since it should not use for the actual training
            del self.config.hdc_m
            del self.config.hdc_s

            # iterate over splits
            if max_k==1:
                # use previous defined split
                HDC_train = HDC[idx_train, :]
                HDC_test = HDC[idx_test, :]
                classifier.fit(HDC_train, label[idx_train])
                pred = classifier.predict(HDC_test)
                acc[0, s_idx] = np.mean(pred == label[idx_test])
            else:
                # multiple splits
                for k in range(max_k):
                    idx_train = []
                    idx_test = []
                    for c in range(len(C)):
                        splits = np.array_split(idx_c[c], max_k)
                        idx_test = np.append(idx_test, splits[k])
                        for t in np.delete(np.arange(max_k), k):
                            idx_train = np.append(idx_train, splits[t])

                    idx_train = idx_train.astype(np.int)
                    idx_test = idx_test.astype(np.int)

                    HDC_train = HDC[idx_train, :]
                    HDC_test = HDC[idx_test, :]
                    classifier.fit(HDC_train, label[idx_train])
                    pred = classifier.predict(HDC_test)
                    acc[k,s_idx] = np.mean(pred == label[idx_test])

        best_scales = np.argmax(acc,1)
        votes = np.bincount(best_scales)
        best_s = scales[np.argmax(votes)]

        return best_s

    def train_model(self,X_train,y_train,X_test,y_test):
        '''
        training function of HDC-MiniRocket
        @param X_train: training data
        @param y_train: training label
        @param X_test: testing data
        @param y_test: testing label
        @return: -
        '''
        self.rocket = MiniRocketMultivariate(random_state=self.config.seed)

        # select best s
        if self.config.best_scale:
            self.config.scale = self.find_best_s(X_train, y_train)
            print("best scale = " + str(self.config.scale))

        t_proc = []
        self.rocket.fit(X_train)
        # time measurement for training
        for i in range(self.config.n_time_measures):
            t = time()
            self.X_train_HDC = self.hdc_rocket_tf(X_train, scale=self.config.scale)
            t_proc.append((time() - t))
        processing_time = np.median(t_proc)
        self.train_preproc = processing_time

        # Ridge classifier
        self.classifier = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10), normalize=True)
        t_train = []
        for i in range(self.config.n_time_measures):
            t = time()
            self.classifier.fit(self.X_train_HDC, y_train)
            t_train.append(time() - t)

        self.training_time = np.median(t_train)

        return

    def eval_model(self, X_train, y_train, X_test, y_test):
        '''
        evaluation function
        @param X_train: training data
        @param y_train: training label
        @param X_test: testing data
        @param y_test: testing label
        @return: accuracy, f1 score and confusion matrix
        '''

        t_proc = []
        # time measurement
        for i in range(self.config.n_time_measures):
            t = time()
            # encode without recomputing the pose matrix for timestamps
            self.X_test_HDC = self.hdc_rocket_tf(X_test, scale=self.config.scale, compute_poses=False)
            t_proc.append((time() - t))
        processing_time = np.median(t_proc)
        self.test_preproc = processing_time

        t_test = []
        for i in range(self.config.n_time_measures):
            t = time()
            pred = self.classifier.predict(self.X_test_HDC)
            t_test.append(time() - t)
        self.testing_time = np.median(t_test)

        # evaluate the results
        logger.info('Results on test data: ')
        report = classification_report(y_test.astype(int), pred, output_dict=True)
        logger.info(classification_report(y_test.astype(int), pred))

        acc = report['accuracy']
        f1 = f1_score(y_test.astype(int), pred, average='weighted')
        cm = confusion_matrix(y_test.astype(int), pred)


        logger.info("Confusion matrix:")
        logger.info(cm)

        # print time results
        logger.info("Data preprocessing time: " + str(self.train_preproc) + ' + ' + str(self.test_preproc) + ' = ' + str(self.train_preproc + self.test_preproc))
        logger.info("Training time: " + str(self.training_time))
        logger.info("Evaluation: " + str(self.testing_time))

        return acc, f1, cm


