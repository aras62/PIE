import numpy as np
from utilities.utils import print_msg, print_stat, assertion
from utilities.utils import exception


def get_scenarios(data, scenario, parameters, verbose=False):
    """
    Generates single or double scenario data id
    Args:
        data: data list. Sequences that are divided into obs/prediction sequences
        scenario: name of a single scenario (string) or two scenarios (list(string))
        parameters: dictionary scenario generation parameters
        verbose:  whether to print the results

    Return:
        Dictionary of scenario ids and scenario name (in case of two scenarios or actions the name changes)
    """
    print_msg(f"#### Generating samples for {scenario} ####", 'green')
    if isinstance(scenario, str):
        data_ids = get_data_scenario_ids(data, scenario, verbose, **parameters)
        if "ped_actions" in scenario:
            scenario = f"{scenario}_{parameters['act_type']}"
        if "ped_occlusion" in scenario:
            scenario = f"{scenario}_{parameters['occ_type']}"
    elif isinstance(scenario, list):
        assertion(scenario[0] != scenario[1],
                  f"sc1({scenario[0]}) = sc2({scenario[1]}). Scenarios should not be the same")
        scen_ids = {}
        _scales = None
        for i, scen in enumerate(scenario):
            sc_ids = get_data_scenario_ids(data, scen, **parameters)
            if "ped_actions" in scen:
                scen = f"{scen}_{parameters['act_type']}"
                scenario[i] = scen
            if "ped_occlusion" in scen:
                scen = f"{scen}_{parameters['occ_type']}"
                scenario[i] = scen
            scen_ids[scen] = sc_ids
        data_ids = get_shared_ids(scen_ids, scenario, data_key='', shared_ids=None, scen_idx=0)
        scenario = ';'.join(scenario)
    else:
        exception(f"scenario format {type(scenario)} is incorrect. Use either string or list of two strings")

    return data_ids, scenario


def get_shared_ids(scen_ids, scenario, data_ids=None, data_key='', shared_ids=None, scen_idx=0):
    """
    A recursive function that generates shared ids across multiple sub-scenarios
    Args:
        scen_ids: dictionary of ids belonging to different individual scenarios
        scenario: list of scenarios
        data_ids: a dictionary of shared data ids. This is empty at the start
        data_key: the empty key to be populated based on the name of sub scenarios
        shared_ids: ids shared across scenarios
        scen_idx: index of the current scenario

    Return:
        A dictionary of ids per sub-scenario combination
    """
    if data_ids is None:
        data_ids = {}
    if scen_idx == len(scenario):
        data_ids[data_key] = shared_ids.tolist()
        return
    for sub_sc, ids in scen_ids[scenario[scen_idx]].items():
        if scen_idx == 0:
            new_shared_id = ids
            new_data_key = f"{sub_sc}"
        else:
            new_shared_id = np.intersect1d(shared_ids, ids, return_indices=False)
            new_data_key = f"{data_key};{sub_sc}"

        get_shared_ids(scen_ids, scenario, data_ids, new_data_key, new_shared_id, scen_idx + 1)

    return data_ids


def get_data_scenario_ids(data, scen_type, verbose=False, **kwargs):
    """
    Generates list of data ids for given scenarios
    Args:
        data: raw data to be processed
        scen_type: type of scenario to generate
        verbose: whether to display ids
        kwargs:
            ped_state:
                       act_type: action types 'actions', 'cross', 'look'
                       obs_length: integer specifying the length of observation
            ped_scale: 
                       scales: list of scales for each category
                       save_hist: whether to save a histogram of categories
            veh_speed:
                       speeds: list of speeds for creating bins
                       save_hist: whether to save a histogram of categories
            veh_speed_change: 
                       speed_th: speed threshold between obs/pred sequences
                       obs_length: integer specifying the length of observation
            veh_turns:
                       turn_thr: The angle (in degrees) used as threshold for detecting turns
            veh_acceleration:
                       acc_th: acceleration threshold used to trigger changes in speed
            ped_occlusion:
                       occ_perc: list of occlusion percentages for each category
            signal:
                        ref_type: Applies only to traffic light. whether 'avg'erage over the sequence or take the 'last' step.
                        group_tfl: if True, groups (undefined + red) and (yellow+green)
            road_type:
                        group_road: if True groups num_lanes < 3 into narrow and rest wide 
            all: sends the id of all samples

    Returns:
        Scenario ids
    """

    if scen_type == 'ped_state':
        scen_ids = get_ped_state_ids(data, **kwargs)
    elif scen_type == 'ped_scale':
        scen_ids = get_ped_scale_ids(data, **kwargs)
    elif scen_type == 'veh_speed':
        scen_ids = get_veh_speed_ids(data, **kwargs)
    elif scen_type == 'veh_speed_change':
        scen_ids = get_veh_speed_change_ids(data, **kwargs)
    elif scen_type == 'veh_turns':
        scen_ids = get_veh_turns_ids(data, **kwargs)
    elif scen_type == 'veh_acceleration':
        scen_ids = get_veh_acceleration_ids(data, **kwargs)
    elif scen_type == 'ped_occlusion':
        scen_ids = get_ped_occlusion_ids(data, **kwargs)
    elif scen_type == 'signal':
        scen_ids = get_signal_ids(data, **kwargs)
    elif scen_type == 'road_type':
        scen_ids = get_road_type_ids(data, **kwargs)
    elif scen_type == 'all':
        scen_ids = {'all': np.arange(data['bbox_org'].shape[0])}
    else:
        raise Exception(f"{scen_type} scenario type is invalid")
    if verbose:
        print_stat(scen_ids)
    return scen_ids


# Only on Trajectory
def get_veh_speed_change_ids(data, speed_th=10, obs_length=15, **kwargs):
    """
    Generates categories of data in which whether the speed is constant
    across obs and pred or not
    Args:
        data: data
        speed_th: float, specify the speed threshold for computing speed change
        obs_length: observation length
    between observation and prediction

    Return:
        Dictionary of data categories with keys constant and change
    """
    scen_ids = {'constant': [], 'change': []}
    speed_data = data['obd_speed'] if 'obd_speed' in data else data['speed']
    for i, seq in enumerate(speed_data):
        obs_avg = np.mean(seq[:obs_length])
        pred_avg = np.mean(seq[obs_length:])
        if abs(obs_avg - pred_avg) <= speed_th:
            scen_ids['constant'].append(i)
        else:
            scen_ids['change'].append(i)
    return scen_ids


# On any sequence
def get_ped_state_ids(data, act_type='actions', obs_length=15, **kwargs):
    """
    Extracts samples based on actions (walking, crossing, looking) aka states of the pedestrian
    during  observation and prediction stages.
    Args:
        data: a dictionary of lists. Each element corresponds to one type of label
        act_type: from actions, look, cross corresponding to the momentarily actions of the pedestrian
        obs_length: for trajectory, obs length will be used to separate the data. if 0, only computed for entire sequence

    Returns:
        Counts of each category and data indices
    """

    assert act_type in ['actions', 'looks', 'cross']
    if act_type == 'actions':
        p = 'w'
        n = 's'  # Walking, stopping
    else:
        p = f'{act_type[0]}'
        n = f'n{act_type[0]}'  # doing, not doing
    if obs_length > 0:
        return _get_ped_state_traj(data, act_type, p, n, obs_length)
    else:
        return _get_ped_state(data, act_type, p, n)


def _get_ped_state_traj(data, act_type, p, n, obs_length):
    """
    Generates ids for sequences of pedestrians according to their state for trajectory prediction
    Args:
        data: a dictionary of lists. Each element corresponds to one type of label
        act_type: from actions, look, cross corresponding to the momentarily actions of the pedestrian
        p: positive label
        n: negative label
        obs_length: for trajectory, obs length will be used to separate the data. if 0, is only computed for the entire sequence

    Returns:
        Scenario ids
    """
    scen_ids = {f'o{p}_p{p}': [], f'o{p}_p{n}': [], f'o{n}_p{p}': [], f'o{n}_p{n}': []}
    for i, seq in enumerate(data[act_type]):
        obs_avg = np.mean(seq[:obs_length])
        pred_avg = np.mean(seq[obs_length:])
        if obs_avg > 0.5 and pred_avg > 0.5:
            tag = f'o{p}_p{p}'
        elif obs_avg > 0.5 >= pred_avg:
            tag = f'o{p}_p{n}'
        elif obs_avg <= 0.5 < pred_avg:
            tag = f'o{n}_p{p}'
        else:
            tag = f'o{n}_p{n}'
        scen_ids[tag].append(i)
    return scen_ids


def _get_ped_state(data, act_type, p, n):
    """
    Generates ids for scenarios based on pedestrian state
    Args:
        data: a dictionary of lists. Each element corresponds to one type of label
        act_type: from actions, look, cross corresponding to the momentarily actions of the pedestrian
        p: positive label
        n: negative label

    Returns:
        Scenario ids
    """
    scen_ids = {f'{p}': [], f'{n}': []}
    for i, seq in enumerate(data[act_type]):
        act_avg = np.mean(seq)
        if act_avg > 0.5:
            tag = f'{p}'
        else:
            tag = f'{n}'
        scen_ids[tag].append(i)
    return scen_ids


def get_ped_scale_ids(data, scales=None, return_scales=False, **kwargs):
    """
    Categorizes pedestrian data according to scale (height of bboxes) of pedestrians
    Args:
        data: dictionary of lists
        scales: list of scales. e.g. [30, 80] would create 3 bins, <30, 30-80, >80
        return_scales: whether to return the computed scales

    Returns:
        Ids of data points falling in each category
    """
    # 'bbox': list([x1, y1, x2, y2]) (float)  
    box_data = data['bbox_org'] if 'bbox_org' in data else data['bbox']
    if scales is None:
        scales = [50, 100, 150, 200]
    scen_ids = {}
    scale_dict = {}
    scales = sorted(scales)
    for i in range(len(scales)):
        if i == 0:
            scen_ids[f'0-{scales[i]}'] = []
            scale_dict[f'0-{scales[i]}'] = []
        if i == len(scales) - 1:
            scen_ids[f'{scales[i]}+'] = []
            scale_dict[f'{scales[i]}+'] = []
        else:
            scen_ids[f'{scales[i]}-{scales[i + 1]}'] = []
            scale_dict[f'{scales[i]}-{scales[i + 1]}'] = []

    for i, seq in enumerate(box_data):
        scale = np.mean(abs(np.array(seq)[:, 1] - np.array(seq)[:, 3]))
        for j, s in enumerate(scales):
            if scale <= s:
                if j == 0:
                    scen_ids[f'0-{s}'].append(i)
                    scale_dict[f'0-{s}'].append(scale)
                else:
                    scen_ids[f'{scales[j - 1]}-{scales[j]}'].append(i)
                    scale_dict[f'{scales[j - 1]}-{scales[j]}'].append(scale)
                break
            else:
                if j == len(scales) - 1:
                    scen_ids[f'{scales[j]}+'].append(i)
                    scale_dict[f'{scales[j]}+'].append(scale)
                    break
    if return_scales:
        return scen_ids, scale_dict
    else:
        return scen_ids


def get_ped_occlusion_ids(data, occ_perc=None, occ_type='full', **kwargs):
    """
    Categorizes pedestrian data according to occlusion state of pedestrians
    Args:
        data: dictionary of lists
        occ_perc: list of occlusion percentages. e.g. [25, 50] would create 3 bins, <25, 25-50, >50
        occ_type: type of occlusion. 'part' or 'full'

    Returns:
        Ids of data points falling in each category
    """
    # 'bbox': list([x1, y1, x2, y2]) (float)  
    occ = data['occlusion']
    assert occ_type in ['part', 'full'], f"{occ_type} is not valid option"
    if occ_perc is None:
        occ_perc = [25, 50, 75]
    scen_ids = {}
    occ_perc = sorted(occ_perc)
    for i in range(len(occ_perc)):
        if i == 0:
            scen_ids[f'0-{occ_perc[i]}'] = []
        if i == len(occ_perc) - 1:
            scen_ids[f'{occ_perc[i]}+'] = []
        else:
            scen_ids[f'{occ_perc[i]}-{occ_perc[i + 1]}'] = []
    occ_id = 0 if occ_type == 'part' else 1
    for i, seq in enumerate(occ):
        num_occ = len(np.where(np.array(seq) > occ_id)[0])
        o_per = num_occ / len(seq)
        for j, o in enumerate(occ_perc):
            if o_per <= o / 100:
                if j == 0:
                    scen_ids[f'0-{o}'].append(i)
                else:
                    scen_ids[f'{occ_perc[j - 1]}-{occ_perc[j]}'].append(i)
                break
            else:
                if j == len(occ_perc) - 1:
                    scen_ids[f'{occ_perc[j]}+'].append(i)
                    break
    return scen_ids


def get_veh_speed_ids(data, speeds=None,  **kwargs):
    """
    Categorizes pedestrian data according to the speed of the ego-vehicle
    Args:
        data: dictionary of lists
        speeds: list of speed. e.g. [30, 80] would create 3 bins, <30, 30-80, >80

    Returns:
        Ids of data points falling in each category
    """
    # 'bbox': list([x1, y1, x2, y2]) (float)  
    if speeds is None:
        speeds = [0, 5, 10, 20, 30, 40]
    scen_ids = {}
    speeds = sorted(speeds)
    for i in range(len(speeds)):
        if i == 0:
            scen_ids[f'0-{speeds[i]}'] = []
        if i == len(speeds) - 1:
            scen_ids[f'{speeds[i]}+'] = []
        else:
            scen_ids[f'{speeds[i]}-{speeds[i + 1]}'] = []
    speed_data = data['obd_speed'] if 'obd_speed' in data else data['speed']
    for i, seq in enumerate(speed_data):
        speed = np.mean(seq)
        for j, s in enumerate(speeds):
            if speed <= s:
                if j == 0:
                    scen_ids[f'0-{s}'].append(i)
                else:
                    scen_ids[f'{speeds[j - 1]}-{speeds[j]}'].append(i)
                break
            else:
                if j == len(speeds) - 1:
                    scen_ids[f'{speeds[j]}+'].append(i)
                    break
    return scen_ids


def get_veh_turns_ids(data, turn_thr=10, **kwargs):
    """
    Generates categories of data for different vehicle behavior (turn/straight)
    Args:
        data: raw data
        turn_thr: threshold for detecting turns

    Returns:
        Dictionary of data categories with keys constant and change
    """
    turn_thr_rad = turn_thr * np.pi / 180
    scen_ids = {'straight': [], 'turn': []}
    for i, seq in enumerate(data['yrp']):
        angle_chg = np.subtract(seq[0][0], seq[-1][0])
        if abs(angle_chg) >= turn_thr_rad:
            scen_ids['turn'].append(i)
        else:
            scen_ids['straight'].append(i)
    return scen_ids


def get_veh_acceleration_ids(data, acc_th=0.3, **kwargs):
    """
    Generates categories of data in which the speed changes, increases/decreases/constant
    Args:
        acc_th: float. threshold below which the speed is considered constant
        comfort acc [-4,4]. 0.3 default for lower bound

    Returns:
        Dictionary of data categories with keys constant and change
    """
    scen_ids = {'constant': [], 'speed_up': [], 'slow_down': []}

    for i, seq in enumerate(data['acc']):
        acc_avg = np.mean(seq)
        if abs(acc_avg) <= acc_th:
            scen_ids['constant'].append(i)
        elif acc_avg < 0:
            scen_ids['slow_down'].append(i)
        else:
            scen_ids['speed_up'].append(i)
    return scen_ids


def get_signal_ids(data, ref_type='last', group_tfl=False, **kwargs):
    """
    Categorizes sequences according to signal status
    Args:
        data: dictionary of lists
        ref_type: applies only to traffic light. Whether to 'avg'erage over the sequence or take the 'last' step.
        group_tfl: if True, groups (undefined + red) and (yellow+green)

    Returns:
        Id of data points falling in each category
    """
    scen_ids = {'ped_sign': [],
                'ped_crossing': [],
                'stop_sign': [],
                'no_tfl': []}
    if group_tfl:
        scen_ids['tfl_forbid'] = []
        scen_ids['tfl_allow'] = []
    else:
        scen_ids['undefined'] = []
        scen_ids['red'] = []
        scen_ids['yellow'] = []
        scen_ids['green'] = []
    tfl_map = {0: 'undefined', 1: 'red', 2: 'yellow', 3: 'green'}

    for i in range(len(data['signalized'])):
        # means there is a traffic signal
        if data['signalized'][i][0][0] > 1:
            if ref_type == 'last':
                tfl_state = data['traffic'][i][-1][0]['traffic_light']
            else:
                tfl_state = np.average(
                    [data['traffic'][i][j][0]['traffic_light'] for j in range(len(data['traffic'][i]))])
            if group_tfl:
                if tfl_state < 2:
                    scen_ids['tfl_forbid'].append(i)
                else:
                    scen_ids['tfl_allow'].append(i)
            else:
                tfl_state = round(tfl_state)
                scen_ids[tfl_map[tfl_state]].append(i)
        else:
            scen_ids['no_tfl'].append(i)
        for s in ['ped_sign', 'ped_crossing', 'stop_sign']:
            if data['traffic'][i][0][0][s] > 0:
                scen_ids[s].append(i)
    return scen_ids


def get_road_type_ids(data, group_road=False, **kwargs):
    """
    Categorizes sequences according to the road structure
    Args:
        data: dictionary of lists
        group_road: if True, groups (< 3) = narrow and rest wide

    Returns:
        Ids of data points falling in each category
    """
    scen_ids = {'one_way': [],
                'two_way': []}
    if group_road:
        scen_ids['narrow'] = []
        scen_ids['wide'] = []

    for i in range(len(data['traffic'])):
        # means there is a traffic signal

        if data['traffic'][i][0][0]['traffic_direction'] == 0:
            scen_ids['one_way'].append(i)
        else:
            scen_ids['two_way'].append(i)

        if group_road:
            if data['traffic'][i][0][0]['num_lanes'] < 3:
                scen_ids['narrow'].append(i)
            else:
                scen_ids['wide'].append(i)
        else:
            nl = data['traffic'][i][0][0]['num_lanes']
            if f'{nl}_lanes' not in scen_ids:
                scen_ids[f'{nl}_lanes'] = []
            scen_ids[f'{nl}_lanes'].append(i)

    return scen_ids
