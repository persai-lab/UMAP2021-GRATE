import numpy as np
import os
import pathlib
import logging
from scipy.stats import entropy
from numpy.linalg import norm

strBold = lambda skk: "\033[1m {}\033[00m".format(skk)
strBlue = lambda skk: "\033[34m {}\033[00m".format(skk)
strRed = lambda skk: "\033[91m {}\033[00m".format(skk)
strGreen = lambda skk: "\033[92m {}\033[00m".format(skk)
strYellow = lambda skk: "\033[93m {}\033[00m".format(skk)
strLightPurple = lambda skk: "\033[94m {}\033[00m".format(skk)
strPurple = lambda skk: "\033[95m {}\033[00m".format(skk)
strCyan = lambda skk: "\033[96m {}\033[00m".format(skk)
strLightGray = lambda skk: "\033[97m {}\033[00m".format(skk)
strBlack = lambda skk: "\033[98m {}\033[00m".format(skk)

prBold = lambda skk: print("\033[1m {}\033[00m".format(skk))
prBlue = lambda skk: print("\033[34m {}\033[00m".format(skk))
prRed = lambda skk: print("\033[91m {}\033[00m".format(skk))
prGreen = lambda skk: print("\033[92m {}\033[00m".format(skk))
prYellow = lambda skk: print("\033[93m {}\033[00m".format(skk))
prLightPurple = lambda skk: print("\033[94m {}\033[00m".format(skk))
prPurple = lambda skk: print("\033[95m {}\033[00m".format(skk))
prCyan = lambda skk: print("\033[96m {}\033[00m".format(skk))
prLightGray = lambda skk: print("\033[97m {}\033[00m".format(skk))
prBlack = lambda skk: print("\033[98m {}\033[00m".format(skk))


def create_logger(log_file, verbose=None):
    """
    create logger for different purpose
    :param verbose:
    :param log_file: the place to store the log
    :return:
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    # formatter = logging.Formatter(
    #     "[%(asctime)s-%(levelname)s-%(name)s]:%(message)s")
    # formatter = logging.Formatter("%(message)s")

    if verbose:
        file_handler = logging.FileHandler("{}.log".format(log_file))
        formatter = logging.Formatter("[%(asctime)s]:%(message)s")
        file_handler.setLevel(logging.DEBUG)  # show DEBUG on console
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        stream_handler = logging.StreamHandler()
        formatter = logging.Formatter("%(message)s")
        stream_handler.setLevel(logging.DEBUG)  # show DEBUG on console
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
    else:
        file_handler = logging.FileHandler("{}.log".format(log_file))
        formatter = logging.Formatter("[%(asctime)s]:%(message)s")
        file_handler.setLevel(logging.INFO)  # only ERROR in file
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        stream_handler = logging.StreamHandler()
        formatter = logging.Formatter("%(message)s")
        stream_handler.setLevel(logging.ERROR)  # show INFO on console
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    return logger


def make_dir(dir_path):
    if not os.path.exists(dir_path):
        pathlib.Path(dir_path).mkdir(parents=True, exist_ok=True)


def check_list_increasing_pattern(test_list):
    increasing = all(i <= j for i, j in zip(test_list, test_list[1:]))
    return increasing


def sigmoid(x, derivative=False):
    """
    compute the sigmoid function 1 / (1 + exp(-x))
    :param x: input of sigmoid function
    :param derivative: if True compute the derivative of sigmoid function
    :return:
    """
    if x > 100:
        sigm = 1.
    elif x < -100:
        sigm = 0.
    else:
        sigm = 1. / (1. + np.exp(-x))

    if derivative:
        return sigm * (1. - sigm)
    return sigm


def JSD(P, Q):
    _P = P / norm(P, ord=1)
    _Q = Q / norm(Q, ord=1)
    _M = 0.5 * (_P + _Q)
    return 0.5 * (entropy(_P, _M) + entropy(_Q, _M))


if __name__ == '__main__':
    p = np.array([0.5, 0.3, 0.2])
    q = np.array([0.3, 0.2, 0.5])
    print(JSD(p, q))
