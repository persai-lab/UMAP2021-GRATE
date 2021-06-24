import json
from multiprocessing import Pool, Lock

output_lock = Lock()
from GRATE.data_helper import *
from GRATE.models.grate import GRATE
from GRATE.utils import *


def sequential_prediction(data_str, material_type, model_name, fold,
                          skill_dim=None, concept_dim=None, lambda_s=None, lambda_t=None, lambda_q=None,
                          lambda_bias=None, sig_level=None, time_windows=None, penalty_weight=None, markovian=None,
                          lr=None, max_iters=None, test_start_time_index=1, test_end_time_index=None, metrics=None,
                          log_file=None, verbose=False, remark=None):
    """
    pipeline of single run of experiment
    :para: a list of parameters for a single case of experiment
    :return:
    """

    perf_dict = {}
    if model_name in ['grate']:
        model_config = config(data_str, material_type, fold, skill_dim, concept_dim, lambda_s, lambda_t, lambda_q,
                              lambda_bias, sig_level, time_windows, penalty_weight, markovian, lr, max_iters,
                              test_start_time_index=test_start_time_index, test_end_time_index=test_end_time_index,
                              metrics=metrics, log_file=log_file, verbose=verbose)
        test_data = model_config['test']
        print("=" * 100)

        if model_name == "grate":
            tutor = GRATE(model_config)
        else:
            raise ValueError

        for test_attempt in range(test_start_time_index, test_end_time_index):
            tutor.current_test_time = test_attempt
            tutor.lr = lr
            train_perf = tutor.training()
            test_set = []
            for (student, time_index, question, score) in test_data:
                if time_index == tutor.current_test_time:
                    test_set.append((student, time_index, question, score))
                    tutor.train_data.append((student, time_index, question, score))
            tutor.testing(test_set)
        score_prediction_perf = tutor.eval(tutor.test_obs_list, tutor.test_pred_list)
        perf_dict["score_prediction"] = score_prediction_perf
        print(score_prediction_perf)
    else:
        raise ValueError

    # save experimental results
    result_dir_path = "results/{}/{}/{}".format(data_str, material_type, model_name)
    if remark is None:
        make_dir(result_dir_path)
        result_file_path = "{}/eval_results.csv".format(result_dir_path)
        output_lock.acquire()
        if data_str in ["morf", "laura"]:
            metric = "rmse"
        else:
            metric = "auc"
        with open(result_file_path, "a") as f:
            f.write("{},{},{}\n".format(
                fold, score_prediction_perf[metric], tutor.agg_time_index_mapping[tutor.current_test_time])
            )
        output_lock.release()
        model_dir_path = "saved_models/{}/{}/{}/fold_{}".format(data_str, material_type, model_name, fold)

        make_dir(model_dir_path)
        para_str = f"skill_{skill_dim}_concept_{concept_dim}_ls_{lambda_s}_lt_{lambda_t}_lq_{lambda_q}_" \
                   f"lbias_{lambda_bias}_sl_{sig_level}_tw_{time_windows}_pw_{penalty_weight}_markov_{markovian}_" \
                   f"lr_{lr}_max_iter_{max_iters}"

        model_file_path = "{}/{}_model.pkl".format(model_dir_path, para_str)
        pickle.dump(tutor, open(model_file_path, "wb"))
    else:
        perf_dict["agg_time_index"] = tutor.agg_time_index_mapping
        save_exp_results(perf_dict, data_str, material_type, model_name, fold,
                         skill_dim, concept_dim, lambda_s, lambda_t, lambda_q, lambda_bias, sig_level, time_windows,
                         penalty_weight, markovian, lr, max_iters, test_start_time_index, test_end_time_index,
                         remark=remark)


def save_exp_results(perf_dict, data_str, material_type, model_name, fold,
                     skill_dim, concept_dim, lambda_s, lambda_t, lambda_q, lambda_bias,
                     sig_level, time_windows, penalty_weight, markovian, lr, max_iters, test_start_time_index,
                     test_end_time_index, remark=None):
    """
    save k-fold results
    """
    result_dir_path = "results/{}/{}/{}".format(data_str, material_type, model_name)
    make_dir(result_dir_path)

    if remark:
        result_file_path = "{}/fold_{}_test_results_{}.json".format(result_dir_path, fold, remark)
    else:
        result_file_path = "{}/fold_{}_test_results.json".format(result_dir_path, fold)

    if not os.path.exists(result_file_path):
        with open(result_file_path, "w") as f:
            pass

    result = {
        'skill_dim': skill_dim,
        'concept_dim': concept_dim,
        'lambda_s': lambda_s,
        'lambda_t': lambda_t,
        'lambda_q': lambda_q,
        'lambda_bias': lambda_bias,
        'sig_level': sig_level,
        'time_windows': time_windows,
        'penalty_weight': penalty_weight,
        'markovian_steps': markovian,
        'learning_rate': lr,
        'max_iters': max_iters,
        'test_start_time_index': test_start_time_index,
        'test_end_time_index': test_end_time_index,
        'perf': perf_dict
    }

    output_lock.acquire()
    with open(result_file_path, "a") as f:
        f.write(json.dumps(result) + "\n")
    output_lock.release()


def run_morf(model_name, verbose=False):
    data_str = "morf"
    material = "QuizIceBreaker"
    metrics = ["rmse", "mae"]

    remark = None
    if model_name == "rgbtf_3_candidates":
        skill_dim = 3
        concept_dim = 9
        lambda_s = 0.01
        lambda_t = 0.01
        lambda_q = 0
        lambda_bias = 0.001
        sig_level = 0.1
        time_windows = 3
        penalty_weight = 0.1
        markovian = 1
        lr = 0.1
        max_iters = 10
        test_start_time_index = 1
        test_end_time_index = 25
    elif model_name == "rgbtf_first_candidate":
        skill_dim = 9
        concept_dim = 9
        lambda_s = 0.001
        lambda_t = 0.001
        lambda_q = 0
        lambda_bias = 0.001
        sig_level = 1.0
        time_windows = 3
        penalty_weight = 0.01
        markovian = 1
        lr = 0.1
        max_iters = 10
        test_start_time_index = 1
        test_end_time_index = 25
    elif model_name == "rgbtf_third_candidate":
        skill_dim = 3
        concept_dim = 9
        lambda_s = 0.001
        lambda_t = 0.001
        lambda_q = 0
        lambda_bias = 0
        sig_level = 0.2
        time_windows = 1
        penalty_weight = 0.1
        markovian = 1
        lr = 0.1
        max_iters = 10
        test_start_time_index = 1
        test_end_time_index = 25
    elif model_name == "agtf":
        skill_dim = 3
        concept_dim = 7
        lambda_s = 0
        lambda_t = 0
        lambda_q = 0
        lambda_bias = 0.01
        sig_level = 0.2
        time_windows = 1
        penalty_weight = 0.001
        markovian = 1
        lr = 0.1
        max_iters = 10

        test_start_time_index = 1
        test_end_time_index = 25
    elif model_name == "rbtf":
        skill_dim = 3
        concept_dim = 9
        lambda_s = 0.001
        lambda_t = 0
        lambda_q = 0
        lambda_bias = 0
        sig_level = 0.2
        time_windows = 1
        penalty_weight = 0.01
        markovian = 1
        lr = 0.1
        max_iters = 10

        test_start_time_index = 1
        test_end_time_index = 25
    else:
        raise ValueError

    result_dir_path = "results/{}/{}/{}".format(data_str, material, model_name)
    make_dir(result_dir_path)
    result_file_path = "{}/eval_results.csv".format(result_dir_path)
    with open(result_file_path, "a") as f:
        para_str = f"skill_{skill_dim}_concept_{concept_dim}_ls_{lambda_s}_lt_{lambda_t}_lq_{lambda_q}_" \
                   f"lbias_{lambda_bias}_sl_{sig_level}_tw_{time_windows}_pw_{penalty_weight}_markov_{markovian}_" \
                   f"lr_{lr}_max_iters_{max_iters}"
        f.write("{}\n".format(para_str))
        f.write("fold, metric, last_time_index\n")

    num_proc = 5
    para_list = []
    for fold in [1, 2, 3, 4, 5]:
        log_path = "logs/{}/{}/{}/test_fold_{}/".format(data_str, material, model_name, fold)
        make_dir(log_path)
        para = [data_str, material, model_name, fold, skill_dim, concept_dim, lambda_s, lambda_t, lambda_q,
                lambda_bias, sig_level, time_windows, penalty_weight, markovian, lr, max_iters,
                test_start_time_index, test_end_time_index]

        delimiter = '_'
        log_name = delimiter.join([str(e) for e in para[4:]])
        log_file = "{}/{}".format(log_path, log_name)
        para.append(metrics)
        para.append(log_file)
        para.append(verbose)
        para.append(remark)
        para_list.append(para)
        # sequential_prediction(*para)
        # return
    pool = Pool(processes=num_proc)
    pool.starmap(sequential_prediction, para_list)
    pool.close()


def run_csintro(model_name, verbose=False):
    data_str = "laura"
    material = "QuizIceBreaker"
    metrics = ["rmse", "mae"]

    remark = None
    if model_name == "rgbtf_3_candidates":
        skill_dim = 9
        concept_dim = 5
        lambda_s = 0
        lambda_t = 0.01
        lambda_q = 0
        lambda_bias = 0.001
        sig_level = 1.0
        time_windows = 1
        penalty_weight = 0.001
        markovian = 1
        lr = 0.1
        max_iters = 10
        test_start_time_index = 1
        test_end_time_index = 50
    elif model_name == "rgbtf_first_candidate":
        skill_dim = 5
        concept_dim = 9
        lambda_s = 0
        lambda_t = 0.01
        lambda_q = 0
        lambda_bias = 0
        sig_level = 0.5
        time_windows = 1
        penalty_weight = 0.01
        markovian = 1
        lr = 0.1
        max_iters = 10
        test_start_time_index = 1
        test_end_time_index = 50
    elif model_name == "rgbtf_third_candidate":
        skill_dim = 7
        concept_dim = 9
        lambda_s = 0
        lambda_t = 0.01
        lambda_q = 0
        lambda_bias = 0
        sig_level = 0.2
        time_windows = 1
        penalty_weight = 0.1
        markovian = 1
        lr = 0.1
        max_iters = 10
        test_start_time_index = 1
        test_end_time_index = 50
    elif model_name == "grate":
        skill_dim = 7
        concept_dim = 9
        lambda_s = 0
        lambda_t = 0.01
        lambda_q = 0
        lambda_bias = 0
        sig_level = 0
        time_windows = 1
        penalty_weight = 0.2
        markovian = 1
        lr = 0.1
        max_iters = 10
        test_start_time_index = 1
        test_end_time_index = 50
    elif model_name == "agtf":
        skill_dim = 7
        concept_dim = 9
        lambda_s = 0
        lambda_t = 0.01
        lambda_q = 0
        lambda_bias = 0
        sig_level = 0.2
        time_windows = 1
        penalty_weight = 0
        markovian = 1
        lr = 0.1
        max_iters = 10
        test_start_time_index = 1
        test_end_time_index = 50
    elif model_name == "rbtf":
        skill_dim = 3
        concept_dim = 7
        lambda_s = 0.01
        lambda_t = 0.01
        lambda_q = 0
        lambda_bias = 0
        sig_level = 0
        time_windows = 3
        penalty_weight = 0.001
        markovian = 1
        lr = 0.1
        max_iters = 10
        test_start_time_index = 1
        test_end_time_index = 50
    else:
        raise ValueError

    result_dir_path = "results/{}/{}/{}".format(data_str, material, model_name)
    make_dir(result_dir_path)
    result_file_path = "{}/eval_results.csv".format(result_dir_path)
    with open(result_file_path, "a") as f:
        para_str = f"skill_{skill_dim}_concept_{concept_dim}_ls_{lambda_s}_lt_{lambda_t}_lq_{lambda_q}_" \
                   f"lbias_{lambda_bias}_sl_{sig_level}_tw_{time_windows}_pw_{penalty_weight}_markov_{markovian}_" \
                   f"lr_{lr}_max_iters_{max_iters}"
        f.write("{}\n".format(para_str))
        f.write("fold, metric, last_time_index\n")

    num_proc = 5
    para_list = []
    # for fold in [4]:
    for fold in [1, 2, 3, 4, 5]:
        log_path = "logs/{}/{}/{}/test_fold_{}/".format(data_str, material, model_name, fold)
        make_dir(log_path)
        para = [data_str, material, model_name, fold, skill_dim, concept_dim, lambda_s, lambda_t, lambda_q,
                lambda_bias, sig_level, time_windows, penalty_weight, markovian, lr, max_iters,
                test_start_time_index, test_end_time_index]
        delimiter = '_'
        log_name = delimiter.join([str(e) for e in para[4:]])
        log_file = "{}/{}".format(log_path, log_name)
        para.append(metrics)
        para.append(log_file)
        para.append(verbose)
        para.append(remark)
        para_list.append(para)
        # sequential_prediction(*para)
        # return
    pool = Pool(processes=num_proc)
    pool.starmap(sequential_prediction, para_list)
    pool.close()


def run_mastery_grids(model_name, verbose):
    data_str = "mastery_grids"
    material = 'QuizIceBreaker'
    metrics = ["rmse", "mae", "auc"]

    remark = None
    if model_name == "rgbtf_3_candidates":
        skill_dim = 9
        concept_dim = 7
        lambda_s = 0
        lambda_t = 0
        lambda_q = 0
        lambda_bias = 0
        sig_level = 0.2
        time_windows = 5
        penalty_weight = 0.01
        markovian = 2
        lr = 0.1
        max_iters = 10
        test_start_time_index = 1
    elif model_name == "grate":
        skill_dim = 5
        concept_dim = 9
        lambda_s = 0
        lambda_t = 0
        lambda_q = 0
        lambda_bias = 0
        sig_level = 0.1
        time_windows = 5
        penalty_weight = 0
        markovian = 0
        lr = 0.1
        max_iters = 10
        test_start_time_index = 1
        test_end_time_index = 70
    elif model_name == "agtf":
        skill_dim = 5
        concept_dim = 7
        lambda_s = 0
        lambda_t = 0.01
        lambda_q = 0
        lambda_bias = 0
        sig_level = 0.2
        time_windows = 5
        penalty_weight = 0.001
        markovian = 1
        lr = 0.1
        max_iters = 10

        test_start_time_index = 1
        test_end_time_index = 70
    elif model_name == "rbtf":
        skill_dim = 3
        concept_dim = 3
        lambda_s = 0.01
        lambda_t = 0.001
        lambda_q = 0
        lambda_bias = 0
        sig_level = 0.2
        time_windows = 5
        penalty_weight = 0.1
        markovian = 1
        lr = 0.1
        max_iters = 10
        test_start_time_index = 1
        test_end_time_index = 70
    else:
        raise ValueError

    result_dir_path = "results/{}/{}/{}".format(data_str, material, model_name)
    make_dir(result_dir_path)
    result_file_path = "{}/eval_results.csv".format(result_dir_path)
    with open(result_file_path, "a") as f:
        para_str = f"skill_{skill_dim}_concept_{concept_dim}_ls_{lambda_s}_lt_{lambda_t}_lq_{lambda_q}_" \
                   f"lbias_{lambda_bias}_sl_{sig_level}_tw_{time_windows}_pw_{penalty_weight}_markov_{markovian}_" \
                   f"lr_{lr}_max_iters_{max_iters}"
        f.write("{}\n".format(para_str))
        f.write("fold, metric, last_time_index\n")

    num_proc = 5
    para_list = []
    for fold in [1, 2, 3, 4, 5]:
        log_path = "logs/{}/{}/{}/test_fold_{}/".format(
            data_str, material, model_name, fold)
        make_dir(log_path)
        para = [data_str, material, model_name, fold, skill_dim, concept_dim, lambda_s, lambda_t, lambda_q,
                lambda_bias, sig_level, time_windows, penalty_weight, markovian, lr, max_iters,
                test_start_time_index, test_end_time_index]
        delimiter = '_'
        log_name = delimiter.join([str(e) for e in para[4:]])
        log_file = "{}/{}".format(log_path, log_name)
        para.append(metrics)
        para.append(log_file)
        para.append(verbose)
        para.append(remark)
        para_list.append(para)
        sequential_prediction(*para)
        return
    # pool = Pool(processes=num_proc)
    # pool.starmap(sequential_prediction, para_list)
    # pool.close()


if __name__ == '__main__':
    model_str = "grate"
    verbose = True
    # run_morf(model_str, verbose)
    # run_csintro(model_str, verbose)
    run_mastery_grids(model_str, verbose)
