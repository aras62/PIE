import numpy as np
from scenarioEval.scenario_generator import get_scenarios
from utilities.utils import write_res_to_file, get_scen_key, assertion
from utilities.utils import get_areas, print_results, print_msg


def evaluate_trajectory_scenario(pred, gt, configs=None, data=None, scenarios=None, verbose=False):
    """
    Evaluates trajectory predictions on scenarios
    Args:
        pred: predictions
        gt: ground-truth
        configs: configurations
        data: data
        scenarios: scenarios
        verbose: whether to display output

    Returns:
        Results
    """
    assertion(scenarios is None or (scenarios is not None and data is not None),
              "Base data is required for generating scenarios")
    scen_res = {}
    configs = configs or {}
    base_results = get_error_trajectory(pred, gt)
    scenarios = scenarios or []
    if data is not None:
        bbox_areas = get_areas(data, configs.get('adj_area', True), configs.get('adj_wh_ratio', 0.34))
    else:
        bbox_areas = None
        print_msg("No data provided. Scaled metrics not computed", 'yellow')

    # Compute metrics per scenario
    for (scenario, parameters) in scenarios:
        data_ids, scenario = get_scenarios(data, scenario, parameters, verbose)
        scenario = get_scen_key(scen_res.keys(), scenario)
        scen_res[scenario] = {}
        for sub_sc, ids in data_ids.items():
            scen_res[scenario][sub_sc] = {'count': len(ids)}
            for metric in base_results:
                scen_res[scenario][sub_sc][metric] = base_results[metric][ids, ...].mean()
            if configs.get('scaled', True):
                get_scaled_metrics(scen_res[scenario][sub_sc], bbox_areas, pred, ids)

    # Compute base metrics
    scen_res['all'] = {'all': {'count': base_results[list(base_results.keys())[0]].shape[0]}}
    for metric in base_results:
        scen_res['all']['all'][metric] = base_results[metric].mean()
    if configs.get('scaled', True) and bbox_areas is not None:
        get_scaled_metrics(scen_res['all']['all'], bbox_areas, pred)

    # Print results
    if verbose:
        print_results(scen_res)
    if configs.get('write_to_file', False):
        write_res_to_file(scen_res, file_path=configs.get('save_file_path', 'scenario_results'))
    return scen_res


def get_scaled_metrics(scen_result, bbox_areas, res, ids=None):
    barea = bbox_areas[ids, ...] if ids is not None else bbox_areas
    metrics = tuple(scen_result.keys())
    for m in metrics:
        if 'count' in m:
            continue
        if 'f' in m:
            ba = barea[..., -1]
        else:
            sidx = barea.shape[1] - res.shape[1]
            p_horz = int(m.split('_')[-1]) if m.split('_')[-1].isdigit() else res.shape[1]
            ba = barea[..., sidx: sidx + p_horz]
        scen_result[f's{m}'] = scen_result[m] / np.mean(ba)


def get_error_trajectory(pred, gt):
    """
    Computes metrics for trajectory outputs
    Args:
        pred: Prediction results
        gt: Ground truth data
    Returns:
        Computed results
    """
    metrics = {}
    pred_len = gt.shape[1]
    error = np.square(gt - pred)
    metrics['B_mse_' + str(pred_len // 3)] = error[:, 0:pred_len // 3, :]
    metrics['B_mse_' + str(pred_len * 2 // 3)] = error[:, 0:pred_len * 2 // 3, :]
    metrics['B_mse'] = error
    metrics['BF_mse'] = error[:, -1, :]

    gt_center = np.stack([(gt[..., 2] + gt[..., 0]) / 2, (gt[..., 3] + gt[..., 1]) / 2], axis=-1)
    res_center = np.stack([(pred[..., 2] + pred[..., 0]) / 2, (pred[..., 3] + pred[..., 1]) / 2], axis=-1)
    center_error = np.square(gt_center - res_center)

    metrics['C_mse_' + str(pred_len // 3)] = center_error[:, 0:pred_len // 3, :]
    metrics['C_mse_' + str(pred_len * 2 // 3)] = center_error[:, 0:pred_len * 2 // 3, :]
    metrics['C_mse'] = center_error
    metrics['CF_mse'] = center_error[:, -1, :]

    return metrics
