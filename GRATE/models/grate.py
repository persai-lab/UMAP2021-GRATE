from GRATE.utils import *
import numpy as np
from numpy import linalg as LA
import time
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import roc_auc_score
import warnings
import copy
import pandas as pd
from surprise import SVD
from surprise import Dataset
from surprise import Reader
from surprise import accuracy
from surprise.model_selection import KFold
from scipy.stats import ttest_rel

warnings.filterwarnings("error")


class GRATE(object):
    """
    GRATE with random state fixed on KFold validation on SVD
    """

    def __init__(self, config, **kwargs):
        """
        :param config:
        :var
        """
        self.random_state = 1
        np.random.seed(self.random_state)
        self.data_str = config['data_str']
        self.time_windows = config['time_windows']
        self.sig_level = config['sig_level']
        log_file = config['log_file']
        verbose = config['verbose']
        if log_file:
            self.logger = create_logger(log_file, verbose)
        self.train_data = config['train']
        self.train_data = sorted(self.train_data, key=lambda x: x[1])
        self.markovian_steps = config['markovian_steps']

        self.num_users = config['num_users']
        self.num_skills = config['num_skills']
        self.num_time_index = config['num_time_index']
        self.num_concepts = config['num_concepts']
        self.num_questions = config['num_questions']
        self.lambda_s = config['lambda_s']
        self.lambda_t = config['lambda_t']
        self.lambda_q = config['lambda_q']
        self.lambda_bias = config['lambda_bias']
        self.penalty_weight = config['penalty_weight']

        self.lr = config['lr']
        self.max_iter = config['max_iter']
        self.tol = config['tol']
        self.metrics = config['metrics']
        self.test_users = config['test_users']
        self.users_data = config['users_data']
        self.current_test_time = config['test_start_time_index']

        # used for student performance prediction
        self.test_obs_list = []
        self.test_pred_list = []
        self.use_global_bias = True
        # True if apply sigmoid for final output value
        self.binarized_question = True
        # False if use log-sigmoid on penalty
        self.exact_penalty = False
        # map from raw time index to aggregated time index
        self.agg_time_index_mapping = {}
        self.agg_train_data = {}
        self.agg_train_data_markov = {}

        self._initialize_parameters()
        # used to weight the least square term in the loss
        self.confidence_dict = {}

    def _initialize_parameters(self):
        """
        :return:
        """
        self.confidence_dict = {}
        self.agg_time_index_mapping = {}
        for i in range(self.num_time_index):
            self.agg_time_index_mapping[i] = i
        self.S = np.random.random_sample((self.num_users, self.num_skills))
        self.T = np.random.random_sample((self.num_skills, self.num_time_index, self.num_concepts))
        self.Q = np.random.random_sample((self.num_concepts, self.num_questions))
        self.bias_s = np.zeros(self.num_users)
        self.bias_t = np.zeros(self.num_time_index)
        self.bias_q = np.zeros(self.num_questions)
        self.global_bias = np.mean(self.train_data, axis=0)[3]

        self.logger.debug(strBlue("*" * 40 + "[ Training Outputs ]" + "*" * 40))
        start_time = time.time()
        converge = False
        iter_num = 0
        min_iter = 5
        best_S, best_T, best_Q = [0] * 3
        best_bias_s, best_bias_t, best_bias_q = [0] * 3

        for stud, _time, ques, obs in self.train_data:
            self.agg_train_data[(stud, _time, ques)] = (_time, obs)
            self.confidence_dict[(stud, _time, ques)] = 1

        self.agg_train_data_markov = {}
        for (stud, _time, ques) in self.agg_train_data:
            upper_steps = min(self.num_time_index, _time + self.markovian_steps + 1)
            for i in range(_time + 1, upper_steps):
                if (stud, i, ques) not in self.agg_train_data:
                    if (stud, i, ques) not in self.agg_train_data_markov:
                        self.agg_train_data_markov[(stud, i, ques)] = True
            lower_steps = max(0, _time - self.markovian_steps)
            for i in range(lower_steps, _time):
                if (stud, i, ques) not in self.agg_train_data:
                    if (stud, i, ques) not in self.agg_train_data_markov:
                        self.agg_train_data_markov[(stud, i, ques)] = True

        # self.agg_train_data_markov = {}
        # for stud, _time, ques, obs in self.train_data:
        #     self.agg_train_data[(stud, _time, ques)] = (_time, obs)
        #     self.confidence_dict[(stud, _time, ques)] = 1
        #     for (stud, _time, ques) in self.agg_train_data:
        #         upper_steps = min(self.num_time_index, _time + self.markovian_steps + 1)
        #         for i in range(_time + 1, upper_steps):
        #             if (stud, i, ques) not in self.agg_train_data:
        #                 if (stud, i, ques) not in self.agg_train_data_markov:
        #                     self.agg_train_data_markov[(stud, i, ques)] = True
        #         lower_steps = max(0, _time - self.markovian_steps)
        #         for i in range(lower_steps, _time):
        #             if (stud, i, ques) not in self.agg_train_data:
        #                 if (stud, i, ques) not in self.agg_train_data_markov:
        #                     self.agg_train_data_markov[(stud, i, ques)] = True

        loss, count, rmse, penalty, reg_loss, reg_bias = self._get_loss()
        self.logger.debug(strBlue(
            "initial: lr: {:.4f}, loss: {:.2f}, q_count: {}, q_rmse: {:.5f}, "
            "penalty: {:.5f}, reg_features: {:.2f}, reg_bias: {:.3f}".format(
                self.lr, loss, count, rmse, penalty, reg_loss, reg_bias))
        )
        loss_list = [loss]
        self.logger.debug(strBlue("*" * 40 + "[ Training Outputs ]" + "*" * 40))

        agg_train_data = list(self.agg_train_data.keys())
        agg_train_data_markov = list(self.agg_train_data_markov.keys())
        while not converge:
            np.random.shuffle(agg_train_data)
            np.random.shuffle(agg_train_data_markov)
            best_S = np.copy(self.S)
            best_T = np.copy(self.T)
            best_Q = np.copy(self.Q)
            best_bias_s = np.copy(self.bias_s)
            best_bias_t = np.copy(self.bias_t)
            best_bias_q = np.copy(self.bias_q)

            for (student, time_index, question) in agg_train_data:
                _, obs = self.agg_train_data[(student, time_index, question)]
                self._optimize_sgd(student, time_index, question, obs)
            # for (student, time_index, question) in agg_train_data_markov:
            #     self._optimize_sgd(student, time_index, question)

            loss, count, rmse, penalty, reg_loss, reg_bias = self._get_loss()

            run_time = time.time() - start_time
            self.logger.debug(
                "iter: {}, lr: {:.4f}, total loss: {:.2f}, count: {}, "
                "weighted rmse: {:.5f}".format(iter_num, self.lr, loss, count, rmse))
            self.logger.debug(
                "--- penalty: {:.5f}, reg_features: {:.2f}, reg_bias: {:.3f}, "
                "run time so far: {:.2f}".format(
                    penalty, reg_loss, reg_bias, run_time))

            if iter_num == self.max_iter:
                self.logger.info("*" * 40 + "[ Training Results ]" + "*" * 40)
                self.logger.info(
                    "** converged **, condition: 0, iter: {}".format(iter_num))
                loss_list.append(loss)
                self.logger.info("training loss: {:.5f}".format(loss))
                self.logger.info("q_rmse: {:.5f}".format(rmse))
                self.logger.info("penalty: {:.5f}".format(penalty))
                self.logger.info(
                    "regularization on parameters: {:.5f}".format(reg_loss))
                converge = True
            elif iter_num >= min_iter and loss >= np.mean(loss_list[-3:]):
                self.logger.info("*" * 40 + "[ Training Results ]" + "*" * 40)
                self.logger.info(
                    "** converged **, condition: 1, iter: {}".format(iter_num))
                self.logger.info("training loss: {:.5f}".format(loss))
                self.logger.info("q_rmse: {:.5f}".format(rmse))
                self.logger.info("penalty: {:.5f}".format(penalty))
                self.logger.info(
                    "regularization on parameters: {:.5f}".format(reg_loss))
                converge = True
            elif loss == np.nan:
                self.lr *= 0.1
            elif loss > loss_list[-1]:
                loss_list.append(loss)
                self.lr *= 0.5
                iter_num += 1
            else:
                loss_list.append(loss)
                iter_num += 1

        # reset to previous S, T, Q
        self.S = best_S
        self.T = best_T
        self.Q = best_Q
        self.bias_s = best_bias_s
        self.bias_t = best_bias_t
        self.bias_q = best_bias_q

    def __getstate__(self):
        """
        since the logger cannot be pickled,
        to avoid the pickle error, we should add this
        :return:
        """
        d = dict(self.__dict__)
        del d['logger']
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)

    def _get_question_prediction(self, student, time_index, question):
        """
        :param time_index: time_index
        :param student: student index
        :param question: question index
        :return:
        """
        pred = np.dot(np.dot(self.S[student, :], self.T[:, time_index, :]), self.Q[:, question])
        if self.use_global_bias:
            pred += self.bias_s[student] + self.bias_q[question] + self.global_bias
        else:
            pred += self.bias_s[student] + self.bias_q[question]

        if self.binarized_question:
            pred = sigmoid(pred)
        return pred

    def _get_loss(self):
        """
        compute the loss, which is RMSE of observed records + regularization
        + penalty of temporal non-smoothness
        :return: loss
        """
        loss, square_loss, bias_reg = 0., 0., 0.
        square_loss_q = 0.
        q_count = 0.
        for (student, time_index, question) in self.agg_train_data:
            _, obs = self.agg_train_data[(student, time_index, question)]
            confd = self.confidence_dict[(student, time_index, question)]
            pred = self._get_question_prediction(student, time_index, question)
            square_loss_q += confd * ((obs - pred) ** 2)
            q_count += 1
        square_loss = square_loss_q

        # regularization
        reg_S = LA.norm(self.S) ** 2
        reg_T = LA.norm(self.T) ** 2
        reg_Q = LA.norm(self.Q) ** 2
        reg_loss = self.lambda_s * reg_S + self.lambda_q * reg_Q + self.lambda_t * reg_T

        q_rmse = np.sqrt(square_loss_q / q_count) if q_count != 0 else 0.
        if self.lambda_bias:
            bias_reg = self.lambda_bias * (LA.norm(self.bias_s) ** 2 + LA.norm(self.bias_q) ** 2)

        if self.penalty_weight != 0. and self.agg_train_data_markov is not None:
            penalty = self._get_penalty()
        else:
            penalty = 0.
        loss = square_loss + reg_loss + bias_reg + penalty
        return loss, q_count, q_rmse, penalty, reg_loss, bias_reg

    def _get_penalty(self):
        """
        compute the penalty on the observations, we want all attempts before
        the obs has smaller score, and the score after obs should be greater.
        otherwise, we add the difference as penalty
        :return:
        """
        penalty = 0.
        for student, time_index, index in self.agg_train_data:
            _, obs = self.agg_train_data[(student, time_index, index)]
            if time_index >= 1:
                gap = self.T[:, time_index, :] - self.T[:, time_index - 1, :]
                knowledge_gap = np.dot(self.S[student, :], gap)
                if self.exact_penalty:
                    knowledge_gap[knowledge_gap > 0.] = 0.
                    penalty_val = -np.dot(knowledge_gap, self.Q[:, index])
                else:
                    diff = np.dot(knowledge_gap, self.Q[:, index])
                    penalty_val = -np.log(sigmoid(diff))
                penalty += self.penalty_weight * penalty_val

        for student, time_index, index in self.agg_train_data_markov:
            if time_index >= 1:
                gap = self.T[:, time_index, :] - self.T[:, time_index - 1, :]
                knowledge_gap = np.dot(self.S[student, :], gap)
                if self.exact_penalty:
                    knowledge_gap[knowledge_gap > 0.] = 0.
                    penalty_val = -np.dot(knowledge_gap, self.Q[:, index])
                else:
                    diff = np.dot(knowledge_gap, self.Q[:, index])
                    penalty_val = -np.log(sigmoid(diff))
                penalty += self.penalty_weight * penalty_val
        return penalty

    def _grad_S_k(self, student, time_index, question, obs=None):
        """
        :param student:
        :param time_index:
        :param question:
        :param obs:
        :return:
        """
        grad = np.zeros_like(self.S[student, :])
        if obs is not None:
            pred = self._get_question_prediction(student, time_index, question)
            confd = self.confidence_dict[(student, time_index, question)]
            if self.binarized_question:
                grad = -2. * confd * (obs - pred) * pred * (1. - pred) * \
                       np.dot(self.T[:, time_index, :], self.Q[:, question])
            else:
                grad = -2. * confd * (obs - pred) * np.dot(
                    self.T[:, time_index, :], self.Q[:, question])
        grad += 2. * self.lambda_s * self.S[student, :]

        last_slice_index = max(self.agg_time_index_mapping.values())
        if time_index == 0:
            diff = self.T[:, time_index + 1, :] - self.T[:, time_index, :]
        elif time_index == last_slice_index:
            diff = self.T[:, time_index, :] - self.T[:, time_index - 1, :]
        else:
            diff = self.T[:, time_index, :] - self.T[:, time_index - 1, :]
            diff += self.T[:, time_index + 1, :] - self.T[:, time_index, :]

        if self.exact_penalty:
            # penalty on Q = - min(0, ST[i]Q - ST[i-1]Q)))
            # grad += self.penalty_weight * grad(penalty on Q)
            # grad(penalty on Q) = - min(0, T[i]Q - T[i-1]Q)
            TQ_diff = np.dot(diff, self.Q[:, question])
            TQ_diff[TQ_diff > 0.] = 0.
            grad += self.penalty_weight * (- TQ_diff)
        else:
            TQ_diff = np.dot(diff, self.Q[:, question])
            val = np.dot(self.S[student, :], TQ_diff)

            # penalty on S = -log(sigmoid(ST[i]Q - ST[i-1]Q))
            # grad += self.penalty_weight * grad(penalty on S)
            # grad(penalty on S) = - (1. / sigmoid(val) * sigmoid(val) * (
            # 1. - sigmoid(val)) * TQ_diff)
            grad += - self.penalty_weight * (
                    1. / sigmoid(val) * sigmoid(val) *
                    (1. - sigmoid(val)) * TQ_diff
            )
        return grad

    def _grad_T_ij(self, student, time_index, index, obs=None):
        """
        compute the gradient of loss w.r.t a specific student j's knowledge at
        a specific attempt i: T_{i,j,:},
        :param time_index: index
        :param student: index
        :param obs: observation
        :return:
        """

        grad = np.zeros_like(self.T[:, time_index, :])
        if obs is not None:
            pred = self._get_question_prediction(student, time_index, index)
            confd = self.confidence_dict[(student, time_index, index)]
            if self.binarized_question:
                grad = -2. * confd * (obs - pred) * pred * (
                        1. - pred) * np.outer(
                    self.S[student, :], self.Q[:, index])
            else:
                grad = -2. * confd * (obs - pred) * np.outer(
                    self.S[student, :], self.Q[:, index])
        grad += 2. * self.lambda_t * self.T[:, time_index, :]

        last_slice_index = max(self.agg_time_index_mapping.values())
        if time_index == 0:
            diff = self.T[:, time_index + 1, :] - self.T[:, time_index, :]
            if self.exact_penalty:
                # penalty on T = - min(0, ST[i]Q - ST[i-1]Q)))
                # grad += self.penalty_weight * grad(penalty on Q)
                # grad(penalty on Q) = - min(0, ST[i] - ST[i-1])
                diff[diff > 0.] = 0.
                penalty_val = -np.dot(np.dot(self.S[student, :], diff),
                                      self.Q[:, index])
                grad += self.penalty_weight * penalty_val * (-1.)
            else:
                val = np.dot(np.dot(self.S[student, :], diff), self.Q[:, index])
                # penalty on T = -log(sigmoid(ST[i]Q - ST[i-1]Q))
                # grad += self.penalty_weight * grad(penalty on T)
                # grad(penalty on T) = - (1. / sigmoid(val) * sigmoid(val) *
                # (1. - sigmoid(val)) * (-1.0)* np.outer(
                # self.S[student,:], self.Q[:, index])

                grad += -self.penalty_weight * (
                        1. / sigmoid(val) * sigmoid(val) * (
                        1. - sigmoid(val)) * (-1.) *
                        np.outer(self.S[student, :], self.Q[:, index])
                )
        elif time_index == last_slice_index:
            diff = self.T[:, time_index, :] - self.T[:, time_index - 1, :]
            if self.exact_penalty:
                diff[diff > 0.] = 0.
                penalty_val = -np.dot(np.dot(self.S[student, :], diff),
                                      self.Q[:, index])
                grad += self.penalty_weight * penalty_val
            else:
                val = np.dot(np.dot(self.S[student, :], diff),
                             self.Q[:, index])
                grad += -self.penalty_weight * (
                        1. / sigmoid(val) * sigmoid(val) * (
                        1. - sigmoid(val)) *
                        np.outer(self.S[student, :], self.Q[:, index])
                )
        else:
            if self.exact_penalty:
                diff = self.T[:, time_index, :] - self.T[:, time_index - 1, :]
                diff[diff > 0.] = 0.
                penalty_val = -np.dot(np.dot(self.S[student, :], diff),
                                      self.Q[:, index])
                grad += self.penalty_weight * penalty_val

                diff = self.T[:, time_index + 1, :] - self.T[:, time_index, :]
                diff[diff > 0.] = 0.
                penalty_val = -np.dot(np.dot(self.S[student, :], diff),
                                      self.Q[:, index])
                grad += self.penalty_weight * penalty_val * (-1.)

            else:
                diff = self.T[:, time_index, :] - self.T[:, time_index - 1, :]
                val = np.dot(np.dot(self.S[student, :], diff),
                             self.Q[:, index])
                grad += -self.penalty_weight * (
                        1. / sigmoid(val) * sigmoid(val) * (
                        1. - sigmoid(val)) *
                        np.outer(self.S[student, :], self.Q[:, index])
                )
                diff = self.T[:, time_index + 1, :] - self.T[:, time_index, :]
                val = np.dot(np.dot(self.S[student, :], diff),
                             self.Q[:, index])
                grad += -self.penalty_weight * (
                        1. / sigmoid(val) * sigmoid(val) * (
                        1. - sigmoid(val)) * (-1.) *
                        np.outer(self.S[student, :], self.Q[:, index])
                )
        return grad

    def _grad_Q_k(self, student, time_index, question, obs=None):
        """
        compute the gradient of loss w.r.t a specific concept
        of a question in Q-matrix,
        :param student:
        :param question:
        :param obs:
        :return:
        """
        grad = np.zeros_like(self.Q[:, question])
        if obs is not None:
            pred = self._get_question_prediction(student, time_index, question)
            confd = self.confidence_dict[(student, time_index, question)]
            if self.binarized_question:
                grad = -2. * confd * (obs - pred) * pred * (
                        1. - pred) * np.dot(
                    self.S[student, :], self.T[:, time_index, :])
            else:
                grad = -2. * confd * (obs - pred) * np.dot(
                    self.S[student, :], self.T[:, time_index, :])
        grad += 2. * self.lambda_q * self.Q[:, question]

        last_slice_index = max(self.agg_time_index_mapping.values())
        if time_index == 0:
            diff = self.T[:, time_index + 1, :] - self.T[:, time_index, :]
        elif time_index == last_slice_index:
            diff = self.T[:, time_index, :] - self.T[:, time_index - 1, :]
        else:
            diff = self.T[:, time_index, :] - self.T[:, time_index - 1, :]
            diff += self.T[:, time_index + 1, :] - self.T[:, time_index, :]
        knowledge_gap = np.dot(self.S[student, :], diff)

        if self.exact_penalty:
            # penalty on Q = - min(0, ST[i]Q - ST[i-1]Q)))
            # grad += self.penalty_weight * grad(penalty on Q)
            # grad(penalty on Q) = - min(0, ST[i] - ST[i-1])
            knowledge_gap[knowledge_gap > 0] = 0.
            grad += self.penalty_weight * (- knowledge_gap)
        else:
            val = np.dot(knowledge_gap, self.Q[:, question])
            # penalty on Q = -log(sigmoid(ST[i]Q - ST[i-1]Q))
            # grad += self.penalty_weight * grad(penalty on Q)
            # grad(penalty on Q) = - (1. / sigmoid(val) * sigmoid(val) *
            # (1. - sigmoid(val)) * knowledge_diff)
            grad += - self.penalty_weight * (
                    1. / sigmoid(val) * sigmoid(val) * (
                    1. - sigmoid(val)) * knowledge_gap
            )
        return grad

    def _grad_bias_s(self, student, time_index, question, obs=None):
        """
        compute the gradient of loss w.r.t a specific bias_s
        :param time_index:
        :param student:
        :param question:
        :param obs:
        :return:
        """
        grad = 0.
        if obs is not None:
            pred = self._get_question_prediction(student, time_index, question)
            confd = self.confidence_dict[(student, time_index, question)]
            if self.binarized_question:
                grad -= 2. * confd * (obs - pred) * pred * (1. - pred)
            else:
                grad -= 2. * confd * (obs - pred)
        grad += 2.0 * self.lambda_bias * self.bias_s[student]
        return grad

    def _grad_bias_t(self, student, time_index, question, obs=None):
        """
        compute the gradient of loss w.r.t a specific bias_a
        :param time_index:
        :param student:
        :param question:
        :return:
        """
        grad = 0.
        if obs is not None:
            pred = self._get_question_prediction(student, time_index, question)
            confd = self.confidence_dict[(student, time_index, question)]
            if self.binarized_question:
                grad -= 2. * confd * (obs - pred) * pred * (1. - pred)
            else:
                grad -= 2. * confd * (obs - pred)
        grad += 2.0 * self.lambda_bias * self.bias_t[time_index]
        return grad

    def _grad_bias_q(self, student, time_index, question, obs=None):
        """
        compute the gradient of loss w.r.t a specific bias_q
        :param time_index:
        :param student:
        :param question:
        :param obs:
        :return:
        """
        grad = 0.
        if obs is not None:
            pred = self._get_question_prediction(student, time_index, question)
            confd = self.confidence_dict[(student, time_index, question)]
            if self.binarized_question:
                grad -= 2. * confd * (obs - pred) * pred * (1. - pred)
            else:
                grad -= 2. * confd * (obs - pred)
        grad += 2. * self.lambda_bias * self.bias_q[question]
        return grad

    def _optimize_sgd(self, student, time_index, question, obs=None):
        """
        train the S, T and Q with stochastic gradient descent
        :param student:
        :param time_index:
        :param question:
        :return:
        """
        # we have 0.1 * self.lr learning rate for updating Q
        if self.current_test_time > 1:
            learning_rate = self.lr * 0.1
        else:
            learning_rate = self.lr

        # optimize S
        # if self.current_test_attempt < (self.num_attempts / 2):
        grad_s = self._grad_S_k(student, time_index, question, obs)
        self.S[student, :] -= self.lr * grad_s
        # when self.exact_penalty == True, S should be always positive
        self.S[student, :][self.S[student, :] < 0.] = 0.
        if self.lambda_s == 0.:
            sum_val = np.sum(self.S[student, :])
            if sum_val != 0:
                self.S[student, :] /= sum_val
        self.bias_s[student] -= self.lr * self._grad_bias_s(
            student, time_index, question, obs)

        # optimize T
        grad_t = self._grad_T_ij(student, time_index, question, obs)
        self.T[:, time_index, :] -= learning_rate * grad_t

        grad_q = self._grad_Q_k(student, time_index, question, obs)
        self.Q[:, question] -= learning_rate * grad_q
        self.Q[:, question][self.Q[:, question] < 0.] = 0.
        if self.lambda_q == 0.:
            sum_val = np.sum(self.Q[:, question])
            if sum_val != 0:
                self.Q[:, question] /= sum_val  # normalization
        self.bias_q[question] -= learning_rate * self._grad_bias_q(
            student, time_index, question, obs)

    def training(self):
        """
        minimize the loss until converged or reach the maximum iterations
        with stochastic gradient descent
        :return:
        """
        self.logger.info(strBlue("*" * 40 + "[ Start   Training ]" + "*" * 40))
        self.logger.info("curr. test time: {}".format(self.current_test_time))
        self.logger.info(strBlue("Ice Breaking ..."))
        self._ice_breaker()

        train_perf = []
        start_time = time.time()
        converge = False
        iter_num = 0
        min_iter = 10
        best_S, best_T, best_Q = [0] * 3
        best_bias_s, best_bias_t, best_bias_q = [0] * 3

        agg_train_data = list(self.agg_train_data.keys())
        agg_train_data_markov = list(self.agg_train_data_markov.keys())

        loss, count, rmse, penalty, reg_loss, reg_bias = self._get_loss()
        self.logger.debug(strBlue(
            "initial: lr: {:.4f}, loss: {:.2f}, q_count: {}, q_rmse: {:.5f}, "
            "penalty: {:.5f}, reg_features: {:.2f}, reg_bias: {:.3f}".format(
                self.lr, loss, count, rmse, penalty, reg_loss, reg_bias))
        )
        loss_list = [loss]
        self.logger.debug(strBlue("*" * 40 + "[ Training Outputs ]" + "*" * 40))

        while not converge:
            np.random.shuffle(agg_train_data)
            np.random.shuffle(agg_train_data_markov)
            best_S = np.copy(self.S)
            best_T = np.copy(self.T)
            best_Q = np.copy(self.Q)
            best_bias_s = np.copy(self.bias_s)
            best_bias_t = np.copy(self.bias_t)
            best_bias_q = np.copy(self.bias_q)

            new_current_test_time = self.agg_time_index_mapping[self.current_test_time]
            count = 0
            for (student, time_index, index) in agg_train_data:
                old_time_index, obs = self.agg_train_data[(student, time_index, index)]
                if -self.time_windows <= (new_current_test_time - time_index) <= self.time_windows:
                    self._optimize_sgd(student, time_index, index, obs)
                    count += 1
            for (student, time_index, index) in agg_train_data_markov:
                # if -(self.time_window + 1) <= (new_current_test_time - time_index):
                if -(self.time_windows + 1) <= (new_current_test_time - time_index) <= (self.time_windows + 1):
                    self._optimize_sgd(student, time_index, index)

            loss, count, rmse, penalty, reg_loss, reg_bias = self._get_loss()
            train_perf.append([count, rmse])

            run_time = time.time() - start_time
            self.logger.debug(
                "iter: {}, lr: {:.4f}, total loss: {:.2f}, count: {}, "
                "weighted rmse: {:.5f}".format(iter_num, self.lr, loss, count, rmse))
            self.logger.debug(
                "--- penalty: {:.5f}, reg_features: {:.2f}, reg_bias: {:.3f}, "
                "run time so far: {:.2f}".format(
                    penalty, reg_loss, reg_bias, run_time))

            if iter_num == self.max_iter:
                self.logger.info("*" * 40 + "[ Training Results ]" + "*" * 40)
                self.logger.info(
                    "** converged **, condition: 0, iter: {}".format(iter_num))
                loss_list.append(loss)
                self.logger.info("training loss: {:.5f}".format(loss))
                self.logger.info("q_rmse: {:.5f}".format(rmse))
                self.logger.info("penalty: {:.5f}".format(penalty))
                self.logger.info(
                    "regularization on parameters: {:.5f}".format(reg_loss))
                converge = True
            elif iter_num >= min_iter and loss >= np.mean(loss_list[-5:]):
                self.logger.info("*" * 40 + "[ Training Results ]" + "*" * 40)
                self.logger.info(
                    "** converged **, condition: 1, iter: {}".format(iter_num))
                self.logger.info("training loss: {:.5f}".format(loss))
                self.logger.info("q_rmse: {:.5f}".format(rmse))
                self.logger.info("penalty: {:.5f}".format(penalty))
                self.logger.info(
                    "regularization on parameters: {:.5f}".format(reg_loss))
                converge = True
            elif loss == np.nan:
                self.lr *= 0.1
            elif loss > loss_list[-1]:
                loss_list.append(loss)
                self.lr *= 0.5
                iter_num += 1
            else:
                loss_list.append(loss)
                iter_num += 1

        # reset to previous S, T, Q
        self.S = best_S
        self.T = best_T
        self.Q = best_Q
        self.bias_s = best_bias_s
        self.bias_t = best_bias_t
        self.bias_q = best_bias_q

        return train_perf[-1]

    def testing(self, test_data, validation=False):
        """
        student performance prediction
        """
        if not validation:
            self.logger.info(
                strGreen("*" * 40 + "[ Testing Results ]" + "*" * 40))
            self.logger.info(
                strGreen("Current testing time index: {}, Test size: {}".format(
                    self.current_test_time, len(test_data))))

        curr_pred_list = []
        curr_obs_list = []

        for (student, old_time_index, question, obs) in test_data:
            curr_obs_list.append(obs)
            new_time_index = self.agg_time_index_mapping[old_time_index]
            pred = self._get_question_prediction(student, new_time_index, question)
            curr_pred_list.append(pred)
            self.test_obs_list.append(obs)
            self.test_pred_list.append(pred)
        return self.eval(curr_obs_list, curr_pred_list)

    def eval(self, obs_list, pred_list):
        """
        evaluate the prediction performance on different metrics
        :param obs_list:
        :param pred_list:
        :return:
        """
        assert len(pred_list) == len(obs_list)

        count = len(obs_list)
        perf_dict = {}
        self.logger.info(strGreen("Test Attempt: {}".format(self.current_test_time)))
        if len(pred_list) == 0:
            return perf_dict
        else:
            self.logger.info(strGreen("Test Size: {}".format(count)))
            perf_dict["count"] = count

        for metric in self.metrics:
            if metric == "rmse":
                rmse = mean_squared_error(obs_list, pred_list, squared=False)
                perf_dict[metric] = rmse
                self.logger.info(strGreen("RMSE: {:.5f}".format(rmse)))
            elif metric == 'mae':
                mae = mean_absolute_error(obs_list, pred_list)
                perf_dict[metric] = mae
                self.logger.info(strGreen("MAE: {:.5f}".format(mae)))
            elif metric == "auc":
                if np.sum(obs_list) == count or np.sum(obs_list) == 0:
                    self.logger.info(strGreen("AUC: None (all ones or all zeros in true y)"))
                    perf_dict[metric] = None
                else:
                    auc = roc_auc_score(obs_list, pred_list)
                    perf_dict[metric] = auc
                    self.logger.info(strGreen("AUC: {:.5f}".format(auc)))
        self.logger.debug("\n")
        return perf_dict

    def _compare_utilities(self, last_slice_dict, candidate_slice_dict, n_splits=5):
        """
        :param n_splits:
        :param last_slice_dict:
        :param candidate_slice_dict:
        :return: True if aggregation is necessary
        """
        last_slice_size = len(last_slice_dict.keys())
        candidate_slice_size = len(candidate_slice_dict.keys())
        if last_slice_size / n_splits < 1.:
            return True
        else:
            # generate compatible dataset for surprise package
            last_slice_ratings_dict = {"itemID": [],
                                       "userID": [],
                                       "rating": []}
            for (student, question) in last_slice_dict:
                _, obs = last_slice_dict[(student, question)]
                last_slice_ratings_dict["itemID"].append(question)
                last_slice_ratings_dict["userID"].append(student)
                last_slice_ratings_dict["rating"].append(obs)

            # compute last slice utility
            df_last_slice = pd.DataFrame(last_slice_ratings_dict)
            reader = Reader(rating_scale=(0, 1))
            data = Dataset.load_from_df(
                df_last_slice[["userID", "itemID", "rating"]], reader)
            kf = KFold(n_splits=n_splits, random_state=self.random_state)
            # this SVD implementation is same as PMF
            algo = SVD(n_factors=self.num_concepts, random_state=self.random_state)
            last_slice_rmse_list = []
            testset_list = []
            for trainset, testset in kf.split(data):
                algo.fit(trainset)
                testset_list.append(testset)
                predictions = algo.test(testset)
                rmse = accuracy.rmse(predictions, verbose=False)
                last_slice_rmse_list.append(rmse)
            last_slice_utility = np.mean(last_slice_rmse_list)
            self.logger.debug(f"Last Slice Mean RMSE over {n_splits} "
                              f"Splits: {last_slice_utility}")

            # compute first candidate slice utility
            candidate_slice_ratings_dict = {"itemID": [],
                                            "userID": [],
                                            "rating": []}
            for (student, question) in candidate_slice_dict:
                _, obs = candidate_slice_dict[(student, question)]
                candidate_slice_ratings_dict["itemID"].append(question)
                candidate_slice_ratings_dict["userID"].append(student)
                candidate_slice_ratings_dict["rating"].append(obs)

            df_candidate_slice = pd.DataFrame(candidate_slice_ratings_dict)
            reader = Reader(rating_scale=(0, 1))
            data = Dataset.load_from_df(
                df_candidate_slice[["userID", "itemID", "rating"]], reader)
            algo = SVD(n_factors=self.num_concepts, random_state=self.random_state)
            candidate_slice_rmse_list = []
            # build testset and trainset based on testset
            # from last_slice_rating data
            for testset in testset_list:
                testset_dict = dict(((user, item), rating) for (user, item, rating) in testset)
                raw_trainset = []
                for (user, item) in candidate_slice_dict.keys():
                    _, rating = candidate_slice_dict[(user, item)]
                    if (user, item) not in testset_dict:
                        raw_trainset.append((user, item, rating, None))
                trainset = data.construct_trainset(raw_trainset)
                algo.fit(trainset)
                predictions = algo.test(testset)
                rmse = accuracy.rmse(predictions, verbose=False)
                candidate_slice_rmse_list.append(rmse)
            candidate_slice_utility = np.mean(candidate_slice_rmse_list)
            self.logger.debug(f"Candidate Slice Mean RMSE over {n_splits} Splits: {candidate_slice_utility}")

            t_stat, p_value = ttest_rel(last_slice_rmse_list, candidate_slice_rmse_list)
            # only when candidate slice mean rmse < last slice mean rmse and
            # difference are significant, we do aggregation
            # if (candidate_slice_utility < last_slice_utility) and (p_value < 0.01):
            if (candidate_slice_utility < last_slice_utility) and (p_value < self.sig_level):
                # if (candidate_slice_utility < last_slice_utility):
                return True
            else:
                return False

    def _ice_breaker(self):
        """
        apply ice_breaker to generate training data for predicting student's
        next performance
        :return:
        """
        # if current test attempt is 1, then we don't need to aggregate data
        if self.current_test_time == 1:
            max_aggregated_attempt = self.num_time_index - 1
            for stud, _time, ques, obs in self.train_data:
                self.agg_train_data[(stud, _time, ques)] = (_time, obs)
                if self.data_str in ['morf', 'laura']:
                    # self.confidence_dict[(stud, _time, ques)] = _time + 1
                    self.confidence_dict[(stud, _time, ques)] = 1
                else:
                    self.confidence_dict[(stud, _time, ques)] = 1
                self.agg_time_index_mapping[_time] = _time
        else:
            # try to aggregate slice at current_test_time -2 and current_test_time -1
            # extract the last slice of aggregated train data
            # last_slice_index is the last slice index of aggregated tensor
            last_slice_index = self.agg_time_index_mapping[self.current_test_time - 2]
            last_slice_dict = {}
            for (stud, new_time_index, ques) in self.agg_train_data.keys():
                old_time_index, obs = self.agg_train_data[(stud, new_time_index, ques)]
                if new_time_index == last_slice_index:
                    last_slice_dict[(stud, ques)] = (old_time_index, obs)
            slice_before_current_test_time_dict = {}
            for stud, _time, ques, obs in self.train_data:
                if _time == self.current_test_time - 1:
                    slice_before_current_test_time_dict[(stud, ques)] = (_time, obs)

            third_candidate = copy.deepcopy(slice_before_current_test_time_dict)
            for (stud, ques) in last_slice_dict:
                old_time_index, obs = last_slice_dict[(stud, ques)]
                third_candidate[(stud, ques)] = (old_time_index, obs)

            if self._compare_utilities(last_slice_dict, third_candidate):
                self.logger.info("aggregate the new slice of training data: candidate 3")
                for old_time_index in self.agg_time_index_mapping.keys():
                    new_time_index = self.agg_time_index_mapping[old_time_index]
                    if old_time_index >= self.current_test_time - 1:
                        self.agg_time_index_mapping[old_time_index] = new_time_index - 1
                for (stud, ques) in third_candidate.keys():
                    old_time_index, obs = third_candidate[(stud, ques)]
                    self.agg_train_data[(stud, last_slice_index, ques)] = (old_time_index, obs)
                    self.confidence_dict[(stud, last_slice_index, ques)] = old_time_index
                # move forward by one slice
                max_aggregated_attempt = 0
                for (stud, new_time_index, ques) in list(self.agg_train_data.keys()):
                    old_time_index, obs = self.agg_train_data[(stud, new_time_index, ques)]
                    if old_time_index >= self.current_test_time:
                        self.agg_train_data.pop((stud, new_time_index, ques))
                        self.agg_train_data[(stud, new_time_index - 1, ques)] = (old_time_index, obs)
                        if self.data_str in ['morf', 'laura']:
                            self.confidence_dict[(stud, new_time_index - 1, ques)] = old_time_index + 1
                        else:
                            self.confidence_dict[(stud, new_time_index - 1, ques)] = 1
                        if new_time_index - 1 > max_aggregated_attempt:
                            max_aggregated_attempt = new_time_index - 1
            else:
                new_current_test_index = self.agg_time_index_mapping[self.current_test_time]
                for (stud, ques) in slice_before_current_test_time_dict.keys():
                    old_time_index, obs = slice_before_current_test_time_dict[(stud, ques)]
                    self.agg_train_data[(stud, new_current_test_index - 1, ques)] = (old_time_index, obs)
                    if self.data_str in ['morf', 'laura']:
                        self.confidence_dict[(stud, new_current_test_index - 1, ques)] = old_time_index + 1
                        # self.confidence_dict[(stud, new_current_test_index - 1, ques)] = 1
                    else:
                        self.confidence_dict[(stud, new_current_test_index - 1, ques)] = 1
                max_aggregated_attempt = 0
                for (stud, new_time_index, ques) in list(self.agg_train_data.keys()):
                    old_time_index, obs = self.agg_train_data[(stud, new_time_index, ques)]
                    if new_time_index > max_aggregated_attempt:
                        max_aggregated_attempt = new_time_index

        # add markovian data for training
        self.agg_train_data_markov = {}
        for stud, _time, ques in self.agg_train_data:
            upper_steps = min(max_aggregated_attempt + 1, _time + self.markovian_steps + 1)
            for i in range(_time + 1, upper_steps):
                if (stud, i, ques) not in self.agg_train_data:
                    if (stud, i, ques) not in self.agg_train_data_markov:
                        self.agg_train_data_markov[(stud, i, ques)] = True
            lower_steps = max(0, _time - self.markovian_steps)
            for i in range(lower_steps, _time):
                if (stud, i, ques) not in self.agg_train_data:
                    if (stud, i, ques) not in self.agg_train_data_markov:
                        self.agg_train_data_markov[(stud, i, ques)] = True
        self.logger.info("number of train data: {}".format(len(self.agg_train_data.keys())))
        self.logger.info("number of train data markovian: {}".format(len(self.agg_train_data_markov.keys())))
