import os
import copy
import numpy as np
# corresponding library for the model used
# from tensorflow.keras.models import Model, load_model
from visualization.visualize import VisualizeObj
from utilities.utils import update_progress, _check_data_ids, print_msg
from utilities.utils import exception, assertion


def model_vis_continuous_traj(data_test, data_fields, model_path=None, configs=None):
    """
    Test function for generating video for a single trajectory prediction model
    Args:
        data_test: the raw data received from the dataset interface
        data_fields: the data types needed for testing the model
        model_path: the path to the folder where the model and config files are saved
        configs: configurations

    Returns:
        None
    """
    img_annots = convert_to_imagewise_annt(data_test, data_fields)
    ped_annots = convert_to_pedwise_annt(data_test, data_fields)
    # Load the model. This should be replaced with a method for model a given model
    test_model = load_model(model_path)
    max_count = configs['vis_config']['max_count']
    video_ids = configs['vis_config']['vid_ids']
    vid_root = configs['vis_config']['save_path']
    set_ids = configs['vis_config']['set_ids'] or sorted(list(img_annots.keys()))
    if configs['vis_config']['seq_ids'] is not None or \
            configs['vis_config']['ped_ids'] is not None:
        print_msg(f'Preset sequence and pedestrian ids are not applicable for continous demo. They are ignored!',
                  'yellow')

    observe_data = {}
    for setid in sorted(set_ids):
        if setid not in img_annots:
            print_msg(f'{setid} does not exist in the data. Skipping!', 'yellow')
            continue
        videoids = video_ids or sorted(img_annots[setid].keys())
        for vidid in videoids:
            if vidid not in img_annots[setid]:
                print_msg(f'{vidid} does not exist in {setid}. Skipping!', 'yellow')
                continue
            configs['vis_config']['save_path'] = os.path.join((vid_root),
                                                              f'traj-{setid}-{vidid}') if max_count is None else \
                os.path.join((vid_root), f'traj-{setid}-{vidid}-fc{max_count}')
            if os.path.isfile(f"{configs['vis_config']['save_path']}.{configs['vis_config']['video_format']}") and \
                    not configs['vis_config']['vis_regen']:
                print_msg(
                    f"{configs['vis_config']['save_path']}.{configs['vis_config']['video_format']} already exists!",
                    'green')
                continue
            vis_obj = VisualizeObj(configs['vis_config'])
            update_len = max_count or len(img_annots[setid][vidid].keys())
            print_msg(f"Generating demo for {setid}/{vidid}")
            counter = 0
            for imgid, fields in sorted(img_annots[setid][vidid].items()):
                counter += 1
                update_progress(counter / update_len)
                # Get the data
                get_data_continuous_traj(imgid, fields, ped_annots, observe_data, configs['data_gen_opts'])
                # Draw the trajectories
                for i, bbox in enumerate(fields['bbox']):
                    pid = fields['pid'][i]
                    pred_traj = None
                    if observe_data[pid]['predict_traj']:
                        pred_traj = test_model.predict(observe_data[pid]['data_input'],
                                                       batch_size=1, verbose=0)
                        # reverse normalization
                        pred_traj = denorm_track(pred_traj,
                                                 observe_data[pid]['data_input_dict']['bbox'],
                                                 configs['data_gen_opts'])
                        observe_data[pid]['pred_traj'] = pred_traj[0]
                vis_obj.visualize_traj(fields['image'], fields, ped_annots, observe_data, configs)
                if max_count is not None:
                    if counter == max_count:
                        vis_obj.finish_visualize()
                        break
            vis_obj.finish_visualize()


def model_vis_continuous_act(data_test, data_fields, model_path=None, configs=None):
    """
    Test function for generating video for a single action prediction model
    Args:
        data_test: the raw data received from the dataset interface
        data_fields: the data types needed for testing th emodel
        model_path: the path to the folder where the model and config files are saved.
        configs: configurations

    Returns:
        None
    """
    img_annots = convert_to_imagewise_annt(data_test, data_fields)
    ped_annots = convert_to_pedwise_annt(data_test, data_fields)
    # Load the model. This should be replaced with a method for model a given model
    test_model = load_model(model_path)

    max_count = configs['vis_config']['max_count']
    video_ids = configs['vis_config']['vid_ids']
    vid_root = configs['vis_config']['save_path']
    set_ids = configs['vis_config']['set_ids'] or sorted(list(img_annots.keys()))
    if configs['vis_config']['seq_ids'] is not None or \
            configs['vis_config']['ped_ids'] is not None:
        print_msg(f'Preset sequence and pedestrian ids are not applicable for continous demo. They are ignored!',
                  'yellow')

    observe_data = {}
    for setid in sorted(set_ids):
        if setid not in img_annots:
            print_msg(f'{setid} does not exist in the data. Skipping!', 'yellow')
            continue
        videoids = video_ids or sorted(img_annots[setid].keys())
        for vidid in videoids:
            if vidid not in img_annots[setid]:
                print_msg(f'{vidid} does not exist in {setid}. Skipping!', 'yellow')
                continue
            configs['vis_config']['save_path'] = os.path.join((vid_root),
                                                              f'act-{setid}-{vidid}') if max_count is None else \
                os.path.join((vid_root), f'act-{setid}-{vidid}-fc{max_count}')
            if os.path.isfile(f"{configs['vis_config']['save_path']}.{configs['vis_config']['video_format']}") and \
                    not configs['vis_config']['vis_regen']:
                print_msg(
                    f"{configs['vis_config']['save_path']}.{configs['vis_config']['video_format']} already exists!",
                    'green')
                continue
            vis_obj = VisualizeObj(configs['vis_config'])
            update_len = max_count or len(img_annots[setid][vidid].keys())
            counter = 0
            for imgid, fields in sorted(img_annots[setid][vidid].items()):
                counter += 1
                update_progress(counter / update_len)
                # Get the data
                get_data_continuous_act(imgid, fields, ped_annots, observe_data, configs['data_gen_opts'])
                # Draw the trajectories
                for i, bbox in enumerate(fields['bbox']):
                    pid = fields['pid'][i]
                    if observe_data[pid]['predict_act']:
                        observe_data[pid]['pred_activity'] = test_model.predict(observe_data[pid]['data_input'],
                                                                                batch_size=1, verbose=0)
                vis_obj.visualize_act(fields['image'], fields, ped_annots, observe_data, configs)
                if max_count is not None:
                    if counter == max_count:
                        break
            vis_obj.finish_visualize()


def model_vis_continuous_mt(data_test, data_fields, model_path=None, configs=None):
    """
    Test function for generating video for a model with multi-task outputs (action and trajectory)
    Args:
        data_test: the raw data received from the dataset interface
        data_fields: the data types needed for testing th emodel
        model_path: the path to the folder where the model and config files are saved.
        configs: configurations

    Returns:
        None
    """
    img_annots = convert_to_imagewise_annt(data_test, data_fields)
    ped_annots = convert_to_pedwise_annt(data_test, data_fields) \
        # Load the model. This should be replaced with a method for model a given model
    test_model = load_model(model_path)

    max_count = configs['vis_config']['max_count']
    video_ids = configs['vis_config']['vid_ids']
    vid_root = configs['vis_config']['save_path']
    set_ids = configs['vis_config']['set_ids'] or sorted(list(img_annots.keys()))
    if configs['vis_config']['seq_ids'] is not None or \
            configs['vis_config']['ped_ids'] is not None:
        print_msg(f'Preset sequence and pedestrian ids are not applicable for continous demo. They are ignored!',
                  'yellow')

    observe_data = {}
    for setid in sorted(set_ids):
        if setid not in img_annots:
            print_msg(f'{setid} does not exist in the data. Skipping!', 'yellow')
            continue
        videoids = video_ids or sorted(img_annots[setid].keys())
        for vidid in videoids:
            if vidid not in img_annots[setid]:
                print_msg(f'{vidid} does not exist in {setid}. Skipping!', 'yellow')
                continue
            configs['vis_config']['save_path'] = os.path.join((vid_root),
                                                              f'mt-{setid}-{vidid}') if max_count is None else \
                os.path.join((vid_root), f'mt-{setid}-{vidid}-fc{max_count}')
            vis_obj = VisualizeObj(configs['vis_config'])
            update_len = max_count or len(img_annots[setid][vidid].keys())
            print(f"{vidid}")
            counter = 0
            for imgid, fields in sorted(img_annots[setid][vidid].items()):
                counter += 1
                update_progress(counter / update_len)
                # Get the data
                get_data_continuous_mt(imgid, fields, ped_annots, observe_data, configs['data_gen_opts'])
                # Draw the trajectories
                for i, bbox in enumerate(fields['bbox']):
                    pid = fields['pid'][i]
                    pred_traj = None
                    pred_act = None
                    if observe_data[pid]['predict_traj']:
                        pred_traj, pred_act = test_model.predict(observe_data[pid]['data_input'],
                                                                 batch_size=1, verbose=0)
                        # reverse normalization
                        pred_traj = denorm_track(pred_traj,
                                                 observe_data[pid]['data_input_dict']['bbox'],
                                                 configs['data_gen_opts'])
                        observe_data[pid]['pred_traj'] = pred_traj[0]
                    if observe_data[pid]['predict_act'] and pred_act is not None:
                        observe_data[pid]['pred_activity'] = pred_act
                vis_obj.visualize_mt(fields['image'], fields, ped_annots, observe_data, configs)
                if max_count is not None:
                    if counter == max_count:
                        break
            vis_obj.finish_visualize()


def model_vis_img(model_output, data_test, configs=None):
    """
    Visualizes the output of a trajectory prediction model on images
    Args:
        model_output: output of the model
        data_test: the raw data received from the dataset interface
        configs: configurations

    Returns:
        None
    """

    max_count = configs['vis_config']['max_count']
    vis_obj = VisualizeObj(configs['vis_config'])
    update_len = max_count or len(model_output)
    # This is to keep track of sequence (sample) id within a given set#/video#
    seqid_cnt = ['', '', -1]  # set, video, seqid
    for i, pred_traj in enumerate(model_output):
        update_progress(i / update_len)
        # reverse normalization
        pred_traj = denorm_track(pred_traj,
                                 data_test['bbox'][i],
                                 configs['data_gen_opts'])
        gt_traj = data_test['bbox'][i][configs['data_gen_opts']['obs_len']:]
        ped_box = data_test['bbox'][i][configs['data_gen_opts']['obs_len'] - 1]
        img_path = data_test['image'][i][configs['data_gen_opts']['obs_len'] - 1]
        vis_data, save_path = _check_data_ids(img_path, configs['vis_config']['save_path'],
                                              seqid_cnt, i, configs['vis_config'],
                                              data_test['pid'][i][0, 0], is_file_path=False)
        if not vis_data:
            continue
        vis_obj.save_path = save_path
        if not os.path.isdir(vis_obj.save_path):
            os.makedirs(vis_obj.save_path)

        vis_obj.visualize_traj_single(img_path,
                                      data_test['pid'][i][0, 0],
                                      ped_box, pred_traj, gt_traj,
                                      configs)
        if max_count is not None:
            if i == max_count:
                break


# Utilities
def convert_to_imagewise_annt(seq_data, data_fields):
    """
    Reorganizes the database according to image ids
    Args:
        seq_data: original data
        data_fields: data fields for visualization

    Returns:
        Converted annotations
    """
    d = {}
    for k in data_fields:
        if k == 'speed':
            d['speed'] = seq_data.get('obd_speed', False)
            if not d['speed']:
                d['speed'] = seq_data['vehicle_act']
        else:
            d[k] = seq_data.get(k)
    img_annots = {}
    for i, img_track in enumerate(d['image']):
        for j, img in enumerate(img_track):
            path_spl = img.split('/')
            setid, vidid, imgid = path_spl[-3:]
            imgid = imgid.split('.')[0]
            if setid not in img_annots:
                img_annots[setid] = {}
            if vidid not in img_annots[setid]:
                img_annots[setid][vidid] = {}
            if imgid not in img_annots[setid][vidid]:
                img_annots[setid][vidid][imgid] = {}
                for k in data_fields:
                    img_annots[setid][vidid][imgid][k] = []
            for k in d:
                if k != 'image':
                    if k == 'event_frames':
                        img_annots[setid][vidid][imgid][k].append(d[k][i])
                    else:
                        if k == 'pid':
                            _data = d[k][i][j][0]
                        else:
                            _data = d[k][i][j]
                        img_annots[setid][vidid][imgid][k].append(_data)
                else:
                    img_annots[setid][vidid][imgid][k] = img
    return img_annots


def convert_to_pedwise_annt(seq_data, data_fields):
    """
    Reorganizes the annotations according to pedestrian ids
    Args:
        seq_data: original data
        data_fields: data fields for visualization

    Returns:
        Converted annotations
    """
    d = {}
    for k in data_fields:
        if k == 'speed':
            d['speed'] = seq_data.get('obd_speed', False)
            if not d['speed']:
                d['speed'] = seq_data['vehicle_act']
        else:
            d[k] = seq_data.get(k, None)  # seq_data[k]
    img_annots = {}
    for i, pid_track in enumerate(d['pid']):
        for j, pid in enumerate(pid_track):
            pid = pid[0]
            if pid not in img_annots:
                img_annots[pid] = {}
                for k in data_fields:
                    img_annots[pid][k] = []
            for k in d:
                if k != 'pid':
                    if k == 'event_frames':
                        img_annots[pid][k] = d[k][i]
                    else:
                        img_annots[pid][k].append(d[k][i][j])
    return img_annots


def get_norm_track(_track, opts):
    """
    Generates normalized tracks
    Args:
        _track: tracks
        opts: data generation options

    Returns:
        normalized tracks
    """
    track = copy.deepcopy(_track)
    if opts.get('norm_pos'):
        norm_pos = opts.get('norm_pos')
        assertion(norm_pos in ['obs', 'last', 'first'], f"{norm_pos} is not a valid option. Options are 'obs', "
                                                        f"'last', 'first'")
        if norm_pos == 'obs':
            norm_idx = opts['obs_len'] - 1
        elif norm_pos == 'last':
            norm_idx = -1
        elif norm_pos == 'first':
            norm_idx = 0
    track = np.subtract(track, track[norm_idx])
    track = track.tolist()
    return track


def denorm_track(_track, org_track, opts):
    """
    Denormalizes the tracks
    Args:
        _track: normalized tracks
        org_track: original tracks
        opts: data generation options

    Returns:
        Denormalized tracks
    """
    track = copy.deepcopy(_track)
    if opts.get('norm_pos'):
        norm_pos = opts.get('norm_pos')
        assertion(norm_pos in ['obs', 'last', 'first'], f"{norm_pos} is not a valid option. Options are 'obs', "
                                                        f"'last', 'first'")
        if norm_pos == 'obs':
            norm_idx = opts['obs_len'] - 1
        elif norm_pos == 'last':
            norm_idx = -1
        elif norm_pos == 'first':
            norm_idx = 0
    track = np.add(track, org_track[norm_idx])
    track = track.tolist()

    return track


# The following functions should be modified depending on the input and output of the evaluated model
def get_data_continuous_traj(imgid, fields, ped_annots, observe_data, opts):
    """
    Generates data for visualization of trajectory prediction models. The samples are generated for all pedestrians in
    a given image
    Args:
        imgid: id of a given image
        fields: fields in annotations
        ped_annots: pedestrian annotations
        observe_data: This keeps track of observation data, maintains obs length per image
        opts: options for data generation

    Returns:
        None
    """
    obs_length = opts['obs_len']
    pred_length = opts['pred_len']

    for i, pid in enumerate(fields['pid']):
        if pid not in observe_data:
            # predict 0: unknown, 1: predict, 2: gt
            observe_data[pid] = {'predict_traj': False,
                                 'pred_traj': None,
                                 'gt_bbox': [],
                                 'data_input_dict': {'bbox': []},
                                 'data_input': [],
                                 'color': np.random.choice(range(256), size=3).tolist()}
        # Add bbox field to the dictionary
        observe_data[pid]['gt_bbox'].append(fields['bbox'][i])

        if len(observe_data[pid]['gt_bbox']) > obs_length:
            # If observed longer than observation length, start predictiong trajectory
            observe_data[pid]['predict_traj'] = True

        # Populate the data dictionary
        observe_data[pid]['data_input_dict']['bbox'].append(fields['bbox'][i])

        # Clip data and only keep the last observations equal to observe_length
        if len(observe_data[pid]['data_input_dict']['bbox']) > obs_length:
            for k in observe_data[pid]['data_input_dict']:
                observe_data[pid]['data_input_dict'][k] = observe_data[pid]['data_input_dict'][k][-obs_length:]

        # Create data input list
        if observe_data[pid]['predict_traj']:
            data_input = []
            for k in opts['obs_input_type']:
                if k == 'norm_bbox':
                    norm_box = get_norm_track(observe_data[pid]['data_input_dict']['bbox'], opts)
                    features = np.array(norm_box)
                else:
                    # Other data types, e.g. vehicle, may need preprocessing depending on the model architecture
                    features = np.array(observe_data[pid]['data_input_dict'][k])
                data_input.append(np.expand_dims(features, axis=0))
            observe_data[pid]['data_input'] = data_input
    # Remove pedestrians that no longer are in the scene
    peds_outside = [k for k in observe_data if k not in fields['pid']]
    for k in peds_outside:
        observe_data.pop(k)


def get_data_continuous_act(imgid, fields, ped_annots, observe_data, opts):
    """
    Generates data for visualization of action prediction models. The samples are generated for all pedestrian in a given image
    Args:
        imgid: id of a given image
        fields: fields in annotations
        ped_annots: pedestrian annotations
        observe_data: This keeps track of observation data, maintains obs length per image
        opts: options for data generation
    Returns:
        None
    """
    obs_length = opts['obs_len']
    pred_length = opts['pred_len']
    time_to_event = opts['time_to_event']
    for i, pid in enumerate(fields['pid']):
        if pid not in observe_data:
            # predict 0: unknown, 1: predict, 2: gt
            observe_data[pid] = {'predict_act': False,
                                 'gt_activity': fields['activities'][i][0],
                                 'pred_activity': None,
                                 'event_frame': fields['event_frames'][i],
                                 'gt_bbox': [],
                                 'data_input_dict': {'bbox': []},
                                 'data_input': [],
                                 'color': np.random.choice(range(256), size=3).tolist()}
        # Add bbox field to the dictionary
        observe_data[pid]['gt_bbox'].append(fields['bbox'][i])
        observe_data[pid]['predict_act'] = False
        if len(observe_data[pid]['gt_bbox']) > obs_length:
            # If within the given time to event, start predicting action
            if isinstance(time_to_event, list) and len(time_to_event) == 2:
                observe_data[pid]['predict_act'] = observe_data[pid]['event_frame'] - time_to_event[0] > \
                                                   int(imgid) > observe_data[pid]['event_frame'] - time_to_event[1]
            elif isinstance(time_to_event, int):
                observe_data[pid]['predict_act'] = observe_data[pid]['event_frame'] - time_to_event == int(imgid)
            elif (isinstance(time_to_event, list) and len(time_to_event) == 0) or time_to_event is None:
                observe_data[pid]['predict_act'] = True
            else:
                exception(
                    f"Time to event is {time_to_event}. It should be a list of two numbers, an empty list, a number, or None")
        # Populate the data dictionary
        observe_data[pid]['data_input_dict']['bbox'].append(fields['bbox'][i])
        # Clip data and only keep the last observations equal to observe_length
        if len(observe_data[pid]['data_input_dict']['bbox']) > obs_length:
            for k in observe_data[pid]['data_input_dict']:
                observe_data[pid]['data_input_dict'][k] = observe_data[pid]['data_input_dict'][k][-obs_length:]
        # Create data input list
        if observe_data[pid]['predict_act']:
            data_input = []
            for k in opts['obs_input_type']:
                if k == 'norm_bbox':
                    norm_box = get_norm_track(observe_data[pid]['data_input_dict']['bbox'], opts)
                    features = np.array(norm_box)
                else:
                    # Other data types, e.g. vehicle, may need preprocessing depending on the model architecture
                    features = np.array(observe_data[pid]['data_input_dict'][k])
                data_input.append(np.expand_dims(features, axis=0))
            observe_data[pid]['data_input'] = data_input
    # Remove pedestrians that no longer are in the scene
    peds_outside = [k for k in observe_data if k not in fields['pid']]
    for k in peds_outside:
        observe_data.pop(k)


def get_data_continuous_mt(imgid, fields, ped_annots, observe_data, opts):
    """
    Generates data for visualization of multi-task methods. The samples are generated  for all pedestrians in a given image
    Args:
        imgid: id of a given image
        fields: fields in annotations
        ped_annots: pedestrian annotations
        observe_data: This keeps track of observation data, maintains obs length per image
        opts: options for data generation
    Returns:
        None
    """
    obs_length = opts['obs_len']
    pred_length = opts['pred_len']
    time_to_event = opts['time_to_event']
    for i, pid in enumerate(fields['pid']):
        if pid not in observe_data:
            # predict 0: unknown, 1: predict, 2: gt
            observe_data[pid] = {'predict_act': False,
                                 'predict_traj': False,
                                 'gt_activity': fields['activities'][i][0],
                                 'pred_activity': None,
                                 'pred_traj': None,
                                 'event_frame': fields['event_frames'][i],
                                 'gt_bbox': [],
                                 'data_input_dict': {'bbox': []},
                                 'data_input': [],
                                 'color': np.random.choice(range(256), size=3).tolist()}
        # Add bbox field to the dictionary
        observe_data[pid]['gt_bbox'].append(fields['bbox'][i])
        observe_data[pid]['predict_act'] = False
        if len(observe_data[pid]['gt_bbox']) > obs_length:
            observe_data[pid]['predict_traj'] = True
            # If within the given time to event, start predicting action
            if isinstance(time_to_event, list):
                observe_data[pid]['predict_act'] = observe_data[pid]['event_frame'] - time_to_event[0] > \
                                                   int(imgid) > observe_data[pid]['event_frame'] - time_to_event[1]
            elif isinstance(time_to_event, int):
                observe_data[pid]['predict_act'] = observe_data[pid]['event_frame'] - time_to_event == int(imgid)
            else:
                observe_data[pid]['predict_act'] = True
        # Populate the data dictionary
        observe_data[pid]['data_input_dict']['bbox'].append(fields['bbox'][i])
        # Clip data and only keep the last observations equal to observe_length
        if len(observe_data[pid]['data_input_dict']['bbox']) > obs_length:
            for k in observe_data[pid]['data_input_dict']:
                observe_data[pid]['data_input_dict'][k] = observe_data[pid]['data_input_dict'][k][-obs_length:]
        # Create data input list
        if observe_data[pid]['predict_traj'] or observe_data[pid]['predict_act']:
            data_input = []
            for k in opts['obs_input_type']:
                if k == 'norm_bbox':
                    norm_box = get_norm_track(observe_data[pid]['data_input_dict']['bbox'], opts)
                    features = np.array(norm_box)
                else:
                    # Other data types, e.g. vehicle, may need preprocessing depending on the model architecture
                    features = np.array(observe_data[pid]['data_input_dict'][k])
                data_input.append(np.expand_dims(features, axis=0))
            observe_data[pid]['data_input'] = data_input
    # Remove pedestrians that no longer are in the scene
    peds_outside = [k for k in observe_data if k not in fields['pid']]
    for k in peds_outside:
        observe_data.pop(k)

