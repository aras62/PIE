import numpy as np
from scenarioEval.scenario_generator import get_scenarios
from utilities.utils import write_res_to_file, get_scen_key, assertion
from utilities.utils import print_results, print_msg
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, average_precision_score


def evaluate_action_scenario(pred, gt, configs=None, data=None, scenarios=None, d_type=None, verbose=False):
    """
    Evaluates action prediction scenarios (if specified) and the overall performance
    Args:
        pred: predictions
        gt: ground-truth
        configs: configurations
        data: data. Primarily used for scenario analysis. for basic evaluation is not needed
        scenarios: scenario specifications
        d_type: data type, whether it is intention, action, or risk
        verbose: whether to display evaluation results
    Returns:
        Scenario ids
    """
    scenarios = scenarios or []
    assertion(len(scenarios) == 0 or (len(scenarios) != 0 and data) is not None,
              "Base data is required for generating scenarios")
    scen_res = {}
    configs = configs or {}
    if d_type is None:
        if pred.shape[-1] == 1:
            d_type = 'act'
        elif pred.shape[-1] == 3:
            d_type = 'intention'
        else:
            d_type = 'risk'
    for (scenario, parameters) in scenarios:
        data_ids, scenario = get_scenarios(data, scenario, parameters, verbose)
        scenario = get_scen_key(scen_res.keys(), scenario)
        scen_res[scenario] = {}
        for sub_sc, ids in data_ids.items():
            scen_res[scenario][sub_sc] = {'count': len(ids)}
            _gt = gt[ids, ...]
            # Get per class count in a given scenario
            for i in range(np.max(gt) + 1):
                scen_res[scenario][sub_sc][f'count_{i}'] = _gt[_gt == i].shape[0]
            if len(ids) != 0:
                evaluate_action(pred, gt, scen_res[scenario][sub_sc],
                                d_type, configs, data, ids)

    # Compute metrics for all samples
    scen_res['all'] = {'all': {'count': len(gt)}}
    for i in range(np.max(gt) + 1):
        scen_res['all']['all'][f'count_{i}'] = gt[gt == i].shape[0]
    evaluate_action(pred, gt, scen_res['all']['all'], d_type, configs, data)

    # Print results
    if verbose:
        print_results(scen_res)
    if configs.get('write_to_file', False):
        write_res_to_file(scen_res, file_path=configs.get('save_file_path', 'scenario_results'))
    return scen_res


def evaluate_action(pred, gt, results, d_type='act', configs=None, data=None, ids=None):
    """
    Computes the metrics for action tasks
    Args:
        pred: predictions
        gt: ground-truth
        results: dictionary of results for scenarios
        d_type: data type for evaluation
        configs: configurations
        data: data
        ids: scenario ids

    Returns:
        results
    """
    if ids is not None:
        pred = pred[ids, ...]
        gt = gt[ids, ...]
    if pred.shape[-1] > 1:
        avg = 'macro'  # For tasks with more than a single output, namely intention and risk
        if 'risk' in d_type:
            wts = get_risk_weights(pred.shape[-1], configs)
            samp_wts = [wts[w] for w in gt]
            results.update(get_base_metrics(gt, pred, d_type, avg, sample_weight=np.array(samp_wts).squeeze()))
            results.update(get_curve_metrics(gt, pred, d_type, avg))
    else:
        avg = 'binary'  # For action task

    results.update(get_base_metrics(gt, pred, d_type, avg, sample_weight=None))
    results.update(get_curve_metrics(gt, pred, d_type, avg))
    if data is None:
        print_msg("No base data provided. Soft, Hard, and weighted action is not computed", 'yellow')
        return
    if 'pid' in data:
        pid_data = data['pid'][:, 0, 0] if ids is None else data['pid'][ids, 0, 0]
        results.update(instance_wise_metrics(gt, pred, pid_data, d_type, avg))
    else:
        print_msg("Pedestrian IDs are not provided. Soft and hard metrics are not computed", 'yellow')
    if 'act' in d_type:
        if 'tte' not in data:
            print_msg('TTE of samples are not found in the data. Weighted action metrics are not computed', 'yellow')
            return
        tte_data = data['tte'][:, 0, 0] if ids is None else data['tte'][ids, 0, 0]
        act_wts = get_act_weights(tte_data, configs)
        results.update(get_base_metrics(gt, pred, d_type, avg, sample_weight=act_wts))


def get_curve_metrics(gt, pred, d_type, avg):
    """
    Computes curve-based metrics
    Args:
        gt: ground-truth
        pred: predictions
        d_type: data type
        avg: the type of averaging

    Returns:
        Results
    """
    results = {}
    mltc = 'raise' if pred.shape[-1] == 1 else 'ovr'
    try:
        results[f'{d_type}_mAP'] = average_precision_score(gt, pred, average=avg if avg != 'binary' else 'macro')
    except:
        results[f'{d_type}_mAP'] = -1
    # in case not all elements are the same
    try:
        results[f'{d_type}_AUC'] = roc_auc_score(gt, pred, multi_class=mltc)  # Before it was rounded to classes
    except:
        results[f'{d_type}_AUC'] = -1
    return results


def get_base_metrics(gt, pred, d_type, avg, sample_weight=None):
    """
    Computes the base metrics
    Args:
        gt: ground-truth
        pred: prediction
        d_type: data type
        avg: averaging type
        sample_weight: sample weights for weighted metrics

    Returns:
        Results
    """
    results = {}
    if sample_weight is not None:
        d_type = f'{d_type}_w'
    if pred.ndim > 1:
        _pred = np.round(pred) if pred.shape[-1] == 1 else np.argmax(pred, axis=-1)
    else:
        _pred = pred
    results[f'{d_type}_Acc'] = accuracy_score(gt, _pred, sample_weight=sample_weight)
    results[f'{d_type}_bAcc'] = balanced_accuracy_score(gt, _pred, sample_weight=sample_weight)
    results[f'{d_type}_Prec'] = precision_score(gt, _pred, average=avg, sample_weight=sample_weight)
    results[f'{d_type}_Recall'] = recall_score(gt, _pred, average=avg, sample_weight=sample_weight)
    results[f'{d_type}_F1'] = f1_score(gt, _pred, average=avg, sample_weight=sample_weight)
    return results


def instance_wise_metrics(gt, pred, pids, d_type, avg):
    """
    Computes instance-wise metrics per pedestrian
    Args:
        gt: ground-truth
        pred: predictions
        pids: pedestrian dis
        d_type: data type
        avg: averaging type

    Returns:
        Results
    """
    ped_res = {}  # collection of performance metrics per ped
    ped_data = {}  # collection of all raw data
    # Rerrange results per pedestrian sample
    for idx in range(len(gt)):
        if pids[idx] not in ped_data:
            ped_data[pids[idx]] = {'pred': [], 'gt': gt[idx, 0]}
        ped_data[pids[idx]]['pred'].append(pred[idx])
    # Generate values per pedestrian
    delta_met = {'max_delta': [],
                 'mean_delta': []}
    # Per pedestrian classification samples
    sample_res = {'mean_conf': [],
                  'conf_vote': [],
                  'gt': []}
    for pid in ped_data:
        fields = np.array(ped_data[pid]['pred'])
        delta = np.abs(fields - np.roll(fields, -1, axis=0))[:-1]
        ped_res[pid] = {'max_delta': np.max(np.max(delta, axis=0)) if delta.any() else 0,
                        'mean_delta': np.mean(delta) if delta.any() else 0}
        sample_res['mean_conf'].append(np.mean(fields, axis=0))
        sample_res['gt'].append(ped_data[pid]['gt'])

        # Aggregate all samples
        for metric in delta_met:
            delta_met[metric].append(ped_res[pid][metric])
        if pred.shape[-1] == 1:
            pred_rd = fields.round().mean()
            # If mean is not either extreme values, means there is disagreement for a given pedestrian
            if pred_rd != 0 and pred_rd != 1:
                # If no agreement set to the wrong gt
                cvote = 1 - ped_data[pid]['gt']
            else:
                cvote = pred_rd
        else:
            pred_am = fields.argmax(axis=-1)
            if pred_am.mean() != pred_am[0]:
                cvote = 0 if ped_data[pid]['gt'] != 0 else 1
            else:
                cvote = pred_am.mean()
        sample_res['conf_vote'].append(cvote)
    # Compute overall delta metrics
    results = {}
    for m_name, metric in delta_met.items():
        results[f'{d_type}_{m_name}'] = np.mean(metric)

    # Compute classification metrics
    # Soft: Conf average per pedestrian
    # Hard: Only true if all pedestrian samples have the same vote
    results.update(
        get_base_metrics(np.array(sample_res['gt']), np.array(sample_res['mean_conf']), f'{d_type}_soft', avg))
    results.update(
        get_base_metrics(np.array(sample_res['gt']), np.array(sample_res['conf_vote']), f'{d_type}_hard', avg))

    return results


def get_risk_weights(num_regions, configs=None):
    """
    Computes the weights of risk classes
    Args:
        num_regions: number of risk regions
        configs: configurations

    Returns:
        Risk weights
    """
    sigma = configs.get('risk_w_sigma', 0.5)
    x = []
    i = cidx = np.ceil(num_regions / 2)
    for j in range(num_regions):
        if j < cidx:
            i -= 1
        elif j > cidx:
            i += 1
        elif j == cidx and num_regions % 2 != 0:
            i += 1
        x.append(i)
    w = np.exp(-(((np.array(x) / np.ceil(num_regions / 2)) / sigma) ** 2) / 2)
    return w / np.sum(w)


def get_act_weights(ttes, configs=None):
    """
    Computes weights of action predictions according to time to event values
    Args:
        ttes: time to event values
        configs: configurations

    Returns:
        Action weights
    """
    sigma = configs.get('act_w_sigma', 0.3)
    tte_dist = -(ttes - np.max(ttes)) / np.max(ttes)
    w = np.exp(-1 / 2 * (tte_dist / sigma) ** 2)

    return w / np.sum(w)
