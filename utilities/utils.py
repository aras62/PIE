import os
import sys
import yaml
import pickle
import numpy as np

# Dictionary of text color codes for color text printing
text_color = {'pink': '\033[95m', 'purple': '\033[35m', 'blue': '\033[94m',
              'cyan': '\033[96m', 'green': '\033[92m',
              'yellow': '\033[93m', 'red': '\033[91m',
              'default': '\033[0m', 'bold': '\033[1m',
              'underline': '\033[4m'}


def get_config(config_path='configs.yaml'):
    """
    Reads configuration file
    Args:
        config_path: path to the configuration

    Return:
        Dictionary of configurations
    """
    with open(config_path, 'r', encoding='utf8') as outfile:
        configs = yaml.safe_load(outfile)
    return configs

# dict_keys(['norm_bbox', 'activities', 'actions', 'image', 'bbox', 'looks', 'scaled_bbox', 'pid', 'speed',
# 'gps_coord', 'yrp', 'acc', 'signalized', 'traffic', 'risk_class'])

def write_res_to_file(results, file_path='scenario_results'):
    """
    Writes scenario-based results into a csv file
    Args:
        results: evaluation results
        file_path: the name of the file

    Returns:
        None
    """
    dirname = os.path.dirname(file_path)
    filename = os.path.basename(file_path)
    os.makedirs(os.path.join(dirname, 'scenario_results'), exist_ok=True)
    f_path = os.path.join(dirname, 'scenario_results', f'{filename}.csv')
    f_path = get_file_path(f_path)
    scn = list(results.keys())[0]
    sub_scn = list(results[scn].keys())[0]
    metrics = list(results[scn][sub_scn].keys())
    with open(f_path, 'wt') as f:
        f.write(f"scenario, sub-scen,{','.join(metrics)}\n")
        for scenario, subscen in results.items():
            for subsc, metrics in subscen.items():
                f.write(f"{scenario}, {subsc}")
                for metric, value in metrics.items():
                    if 'count' in metric:
                        f.write(f",{value}")
                    else:
                        f.write(f",{value:.7f}")
                f.write("\n")
    print(f"{text_color['green']}Scenario results written to {f_path}{text_color['default']}")


def get_file_path(file_path):
    """
    Checks whether a file path exists. If that is the case, creates a new
    file name by augmenting it with a digit
    Args:
        file_path: original file path

    Return:
        A new file_path if original one exists, otherwise returns the original path
    """
    ext = ''
    filename = os.path.basename(file_path)
    if '.' in filename:
        filename, ext = filename.split('.')
        ext = '.' + ext
    dirname = os.path.dirname(file_path)

    while os.path.exists(os.path.join(dirname, filename + ext)):
        _idx = filename.split("_")[-1]
        if _idx.isdigit():
            del_idx = filename.rindex("_")
            filename = f"{filename[: del_idx]}_{int(_idx) + 1}"
        else:
            filename = f"{filename}_1"
    return os.path.join(dirname, filename + ext)


def get_scen_key(dict_keys, key):
    """
    Generates a new key for repeated entry. This is for similar scenarios with different parameters
    Args:
        dict_keys: dictionary keys
        key: a given key in the dict

    Return:
        New key
    """
    if key not in dict_keys:
        return key
    while key in dict_keys:
        _idx = key.split("_")[-1]
        if _idx.isdigit():
            del_idx = key.rindex("_")
            key = f"{key[: del_idx]}_{int(_idx) + 1}"
        else:
            key = f"{key}_1"
    return key


def get_areas(data, adj_area=True, wh_ratio=0.34):
    """
    Computes the average scale for each given sequence
    Args:
        data: data
        adj_area: to handle edge cases where bounding boxes appear very thin
        wh_ratio: width to height ratio of boxes based on the data stats

    Return:
        List of scales (areas of bounding boxes)
    """
    # 'bbox': list([x1, y1, x2, y2]) (float)  
    box_data = data['bbox_org'] if 'bbox_org' in data else data['bbox']

    width = np.array(box_data)[:, :, 2] - np.array(box_data)[:, :, 0]
    height = np.array(box_data)[:, :, 3] - np.array(box_data)[:, :, 1]
    if adj_area:
        width = np.where(width / height < wh_ratio, height * wh_ratio, width)
    area = np.multiply(width, height)
    return area


def print_results(scen_res, sc_color='purple', ssc_color='blue', m_color='cyan'):
    """
    Prints the results (metrics)
    Args:
        scen_res: results of the scenario
        sc_color: scenario text color
        ssc_color: sub-scenario text color
        m_color: metric text color

    Returns:
        None
    """
    for scen in scen_res:
        print(f"{text_color[sc_color]}### Scenario {scen} ###{text_color['default']}")
        for sub_scen in scen_res[scen]:
            print(f"{text_color[ssc_color]}## Sub-scenario {sub_scen} ##{text_color['default']}")
            for metric, value in scen_res[scen][sub_scen].items():
                if isinstance(value, int):
                    print(f"{text_color[m_color]}{metric}{text_color['default']}:{value}")
                elif isinstance(value, np.ndarray):
                    print(f"{value}", ','.join('{:.2f}'.format(x) for x in value))
                else:
                    print(f"{text_color[m_color]}{metric}{text_color['default']}:{value:.3f}")


def update_progress(progress):
    """
    Creates a progress bar
    Args:
         progress: Progress thus far

    Returns:
        None
    """
    bar_len = 20  # Modify this to change the length of the progress bar
    status = ""
    if isinstance(progress, int):
        progress = float(progress)

    block = int(round(bar_len * progress))
    text = "\r[{}] {:0.2f}% {}".format("#" * block + "-" * (bar_len - block), progress * 100, status)
    sys.stdout.write(text)
    sys.stdout.flush()


def read_pickle(file_path):
    """
    Loads and returns a pickle file
    Args:
        file_path: path to the pickle file

    Returns:
        Content of the pickle file

    """
    with open(file_path, 'rb') as fid:
        try:
            return pickle.load(fid)
        except:
            return pickle.load(fid, encoding='bytes')


def _check_data_ids(img_name, root_path, seqid_cnt=None, g_setid=None,
                    vis_config=None, ped_id=None, file_ext=None, is_file_path=True):
    """
    If any videos or set specified, checks whether the data sample is to be selected.
    This also sets up the name of the file.
    Args:
        img_name: name of the image file
        root_path: root path name where the file to be saved
        seqid_cnt: sequence id counter for each video
        g_setid: global set id over all sequences
        vis_config: visualization configuration
        ped_id: pedestrian id of a given sequence
        file_ext: extension of the file
        is_file_path: whether to generate a file (for videos)
        or folder (for images) path
    Return:
        True or false depending on whether the given sequence is to be visualized and
        file/folder path for saving the outcome
    """
    im_name = img_name.split('/')
    setid, vidid = im_name[-3:-1]
    # if set or video are not predefined, global sequence id is used
    if seqid_cnt is not None:
        if vis_config['set_ids'] is None and vis_config['vid_ids'] is None:
            seqid = g_setid
        else:
            if not (setid == seqid_cnt[0] and vidid == seqid_cnt[1]):
                seqid_cnt[0:] = [setid, vidid, -1]
            seqid_cnt[-1] += 1
            seqid = seqid_cnt[-1]
    else:
        seqid = None

    if is_file_path:
        seq_txt = "" if seqid is None else f"-{seqid}"
        file_ext = "" if file_ext is None else f"-{file_ext}"
        save_path = os.path.join(root_path, f"{setid}-{vidid}-{ped_id}{seq_txt}{file_ext}")
    else:
        save_path = os.path.join(root_path, setid, vidid)

    vis_data = True
    if vis_config['ped_ids'] is not None:
        if ped_id not in vis_config['ped_ids']:
            vis_data = False
    if vis_config['set_ids'] is not None:
        if setid not in vis_config['set_ids']:
            vis_data = False
    if vis_config['vid_ids'] is not None:
        if vidid not in vis_config['vid_ids']:
            vis_data = False
    if vis_config['seq_ids'] is not None:
        if seqid not in vis_config['seq_ids']:
            vis_data = False
    return vis_data, save_path


def print_stat(data_ids, ssc_color='blue'):
    """
    Prints scenarios' statistics
    Args:
        data_ids: data ids
        ssc_color: sub-scenario color

    Returns:
        None
    """
    total = 0
    for k, v in data_ids.items():
        print(f"{text_color[ssc_color]}{k}{text_color['default']}: {len(v)}")
        total += len(v)
    print(f"{text_color[ssc_color]}total num samples:{text_color['default']} {total}")


def print_msg(text, color='blue'):
    """
    Prints a message
    Args:
        text: text to be printed
        color: color of the text. Options are according to the text_color above

    Returns:
        None
    """
    print(f"{text_color[color]}{text}{text_color['default']}")


def print_2msg(text1, text2, color1='blue', color2='default'):
    """
    Prints a two-part message with different colors
    Args:
        text1: text 1 for printing
        text2: text 2 for printing
        color1: color of text 1
        color2: color of text 2

    Returns:
        None
    """
    print(f"{text_color[color1]}{text1}{text_color['default']} {text_color[color2]}{text2}{text_color['default']}")


def print_tracks_stats(_data, msg, color='purple'):
    """
    Prints the statistics of the tracks
    Args:
        _data: data
        msg: message to display
        color: color of the message

    Returns:
        None
    """
    # Number of pedestrian instances
    print(f"{text_color[color]}Total number of pedestrian {msg}:{text_color['default']} {len(_data['bbox'])}")
    new_gt_labels = [gt[0] for gt in _data['activities']]
    num_pos_samples = np.count_nonzero(np.array(new_gt_labels))
    num_neg_samples = len(_data['activities']) - num_pos_samples
    print(
        f"\t {text_color['red']}Negative:{text_color['default']} {num_neg_samples}\t {text_color['green']}Positive:{text_color['default']} {num_pos_samples}\n")


def assertion(condition, msg, color='red'):
    """
    Performs assertion with colored printing
    Args:
        condition: assertion condition
        msg: message to display
        color: color of the message

    Returns:
        None
    """
    assert condition, f"{text_color[color]}{msg}{text_color['default']}"


def exception(msg, color='red'):
    """
    Raises an exception with colored message printing
    Args:
        msg: message to display
        color: color of the message

    Returns:
        None
    """
    raise Exception(f"{text_color[color]}{msg}{text_color['default']}")


def get_predictions(fname='outputs/traj_pred.csv', delim=',', reshape=False):
    """
    A helper function to load the prediction results saved as a csv file
    Args:
        fname: file name
        delim: delimiter of the csv file
        reshape: whether the data needs to be reshaped. This is for trajectory results
        which are 3D arrays of shape NxTx4

    Returns:
        An array of predictions
    """
    mout = np.loadtxt(fname, delimiter=delim)
    if reshape:
        return mout.reshape(mout.shape[0], -1, 4)
    return mout if mout.ndim > 1 else np.expand_dims(mout, -1)
