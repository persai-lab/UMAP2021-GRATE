import numpy as np
import pickle
import random
from GRATE.utils import *


def config(data_str, material_type, fold,
           skill_dim=None,
           concept_dim=None,
           lambda_s=None,
           lambda_t=None,
           lambda_q=None,
           lambda_bias=None,
           sig_level=None,
           time_windows=None,
           penalty_weight=None,
           markovian=None,
           lr=None,
           max_iters=None,
           test_start_time_index=1,
           test_end_time_index=None,
           metrics=None,
           log_file=None,
           verbose=False):
    """
    prepare data for knowledge modeling and recommendation
    generate model configurations for training and testing
    such as initialization of each parameters and hyper-parameters
    :return: config dict
    """

    with open('data/{}/{}/{}_train_val_test.pkl'.format(
            data_str, material_type, fold), 'rb') as f:
        data = pickle.load(f)

    # correct the num_time_index and test_end_time_index based on configuration
    num_time_index = data['num_attempts']
    if test_end_time_index is None:
        test_end_time_index = num_time_index
    elif test_end_time_index < num_time_index:
        num_time_index = test_end_time_index
    else:
        test_end_time_index = num_time_index

    # find train_users, test_users, and collect all users_data
    users_data = {}
    test_data = []
    test_users = {}
    for (student, time_index, question, score, resource) in data['test']:
        assert 0. <= score <= 1.0
        assert type(student) is int
        assert type(time_index) is int
        assert type(question) is int
        assert resource == 0
        if time_index < test_end_time_index:
            test_data.append((student, time_index, question, score))
            if student not in users_data:
                users_data[student] = []
            users_data[student].append([student, time_index, question, score])
            if student not in test_users:
                test_users[student] = True

    train_data = []
    train_users = {}
    for (student, time_index, question, score, resource) in data['train']:
        assert 0. <= score <= 1.0
        assert type(student) is int
        assert type(time_index) is int
        assert type(question) is int
        assert resource == 0
        if time_index < test_end_time_index:
            train_data.append((student, time_index, question, score))
            if student not in users_data:
                users_data[student] = []
            users_data[student].append([student, time_index, question, score])
            if student not in train_users and student not in test_users:
                train_users[student] = True

    for (student, time_index, question, score, resource) in data['val']:
        assert 0. <= score <= 1.0
        assert type(student) is int
        assert type(time_index) is int
        assert type(question) is int
        assert resource == 0
        if time_index < test_end_time_index:
            train_data.append((student, time_index, question, score))
            if student not in users_data:
                users_data[student] = []
            users_data[student].append([student, time_index, question, score])
            if student not in train_users and student not in test_users:
                train_users[student] = True

    # generate next question dict, where key is question, value is a dict that
    # storing the next question (key) and count of transition (value) from
    # current question to next question from existing data
    next_questions_dict = {}
    for user in list(users_data.keys()):
        records = sorted(users_data[user], key=lambda x: x[1])
        users_data[user] = records
        for index, (_, _, question, score) in enumerate(records[:-1]):
            if question not in next_questions_dict:
                next_questions_dict[question] = {}
            next_question = records[index + 1][2]
            if next_question not in next_questions_dict[question]:
                next_questions_dict[question][next_question] = 0
            next_questions_dict[question][next_question] += 1

    train_set = []
    test_set = []

    # build question_scores_dict and user_records_by_questions
    question_scores_dict = {}
    user_records_by_questions = {}
    for user in sorted(list(users_data.keys())):
        records = users_data[user][:test_end_time_index]
        if user not in user_records_by_questions:
            user_records_by_questions[user] = {}

        for _, time_index, question, score in records:
            if question not in question_scores_dict:
                question_scores_dict[question] = []
            question_scores_dict[question].append(score)

            if question not in user_records_by_questions[user]:
                user_records_by_questions[user][question] = []
            user_records_by_questions[user][question].append(score)

        for question in user_records_by_questions[user]:
            scores = user_records_by_questions[user][question]

        if user in test_users:
            train_set.extend(records[:test_start_time_index])
            test_set.extend(records[test_start_time_index:])
        else:
            train_set.extend(records)

    print("=" * 100)
    print("train data: {}".format(len(train_data)))
    print("test data: {}".format(len(test_data)))
    print("number of train users: {}".format(len(train_users)))
    print("number of test users: {}".format(len(test_users)))
    print("Question: Mean Score, STD")
    for question in question_scores_dict:
        print("{}:{:5f},{:.5f}".format(question,
                                       np.mean(question_scores_dict[question]),
                                       np.std(question_scores_dict[question])))

    if concept_dim is not None:
        config_dict = {
            'data_str': data_str,
            'num_users': data['num_users'],
            'num_time_index': num_time_index,
            'num_questions': data['num_quizzes'],
            'num_skills': skill_dim,
            'num_concepts': concept_dim,
            'lambda_s': lambda_s,
            'lambda_t': lambda_t,
            'lambda_q': lambda_q,
            'lambda_bias': lambda_bias,
            'sig_level': sig_level,
            'time_windows': time_windows,
            'penalty_weight': penalty_weight,
            'markovian_steps': markovian,
            'lr': lr,
            'max_iter': max_iters,
            'tol': 1e-3,  # tolerance rate for early stopping
            'test_start_time_index': test_start_time_index,
            'test_end_time_index': test_end_time_index,
            'metrics': metrics,
            'log_file': log_file,
            'verbose': verbose
        }
    else:
        # config for PMF
        config_dict = {
            'data_str': data_str,
            'num_users': data['num_users'],
            'num_time_index': num_time_index,
            'num_questions': data['num_quizzes'],
            'test_start_time_index': test_start_time_index,
            'test_end_time_index': test_end_time_index,
            'log_file': log_file,
            'verbose': verbose
        }

    print("=" * 100)
    for key in config_dict.keys():
        print(strRed("{}={}".format(key, config_dict[key])))

    config_dict['users_data'] = users_data
    config_dict['train'] = train_set
    config_dict['test'] = test_set
    config_dict['test_users'] = test_users
    config_dict['question_scores_dict'] = question_scores_dict
    config_dict['next_questions_dict'] = next_questions_dict

    return config_dict
