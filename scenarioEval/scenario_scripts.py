"""
This script is to reproduce evaluation results on scenarios extracted from data as presented in the following papers:

* A. Rasouli, "A Novel Benchmarking Paradigm and a Scale- and Motion-Aware Model for Egocentric Pedestrian Trajectory Prediction", ICRA, 2024
* A. Rasouli, I. Kotseruba, "Diving Deeper Into Pedestrian Behavior Understanding: Intention Estimation, Action Prediction, and Event Risk Assessment", IV, 2024

"""

import sys
import os
sys.path.append('../')
from utilities.data_gen_utils import get_tracks, get_dataset
from utilities.utils import get_config, get_predictions
from scenarioEval.trajectory_evaluate import evaluate_trajectory_scenario
from scenarioEval.action_evaluate import evaluate_action_scenario


# Assign environment paths for datasets if already not defined
if 'JAAD_PATH' not in os.environ:
    os.environ['JAAD_PATH'] = '../../JAAD'
if 'PIE_PATH' not in os.environ:
    os.environ['PIE_PATH'] = '../../../PIE'


def evaluate_scenario_trajectory(config_path='../utilities/configs.yaml',
                                 pred_path='../model_outputs/ec_traj_pie.csv'):
    """
    Evaluate a given trajectory model based on scenarios.
    Parameters below are set based on standard benchmark on PIE according to
    following papers:
    "A Novel Benchmarking Paradigm and a Scale-and Motion-Aware Model for Egocentric Pedestrian Trajectory Prediction"
    "Pie: A large-scale dataset and models for pedestrian intention estimation and trajectory prediction"
    Args:
        config_path: path to the configuration file
        pred_path: path to the predictions output of the evaluated model

    Returns:
        A dictionary of results
    """
    configs = get_config(config_path)
    # The assumption is that model's predictions are recorded as a csv file
    # The output is in the format of
    # array([[bbox_t1], [bbox_t2], ..., [bbox_tn]]^1...
    # [[bbox_t1], [bbox_t2], ..., [bbox_tn]]^k)
    # where n is the length of the prediction and k is the total number of samples. 
    # i.e. an ordered list of predictions
    # NOTE:  the results are based on an ordered list of sequences and should 
    # match the order of the generated data
    predictions = get_predictions(pred_path, reshape=True)

    configs['data_gen_opts']['obs_len'] = 15 # 0.5s or 15 frames
    configs['data_gen_opts']['pred_len'] = 45 # 1.5s or 45 frames
    # NOTE: if time to event is to be set, pred_len should be <= min(TTE)
    # If not, sequences can potentially become shorter than obs_len + pred_len, so
    # future steps should be padded (not currently implemented in the data generation code)
    configs['data_gen_opts']['time_to_event'] = []
    configs['data_gen_opts']['overlap'] = 0.5
    imdb = get_dataset(configs['data_gen_opts']['dataset'])
    data_raw = imdb.generate_data_trajectory_sequence('test', **configs['dataset_opts'])
    data = get_tracks(data_raw, configs['data_gen_opts'], subset='test')

    # Grab the ground truth. Note if boxes are actual coordinates, 'bbox' should be used
    gt = data['norm_bbox'][:, configs['data_gen_opts']['obs_len']:, ...]
    return evaluate_trajectory_scenario(predictions, gt,
                                              configs=configs, data=data,
                                              scenarios=configs['scenarios']['traj_sf'],
                                              verbose=True)


def evaluate_scenario_action(config_path='../utilities/configs.yaml', pred_path=None):
    """
    Evaluate a given action/intention/risk prediction model based on scenarios.
    Parameters below are set based on the following papers:
    "Diving Deeper Into Pedestrian Behavior Understanding: Intention Estimation, Action Prediction, and Event Risk Assessment"
    "Pie: A large-scale dataset and models for pedestrian intention estimation and trajectory prediction"
    Args:
        config_path: path to the configuration file
        pred_path: path to the predictions output of the evaluated model

    Returns:
        A dictionary of results
    """
    configs = get_config(config_path)

    # Generate data for a given data type: intention, action or risk
    dset = 'action'
    configs['data_gen_opts']['sequence_type'] = dset
    # Load model output
    # For different tasks, the shape of the second column varies, [<num_seq>, <int:3/act:1/risk:12>]
    # NOTE:  the results are based on an ordered list of sequences and should 
    # match the order of generated data  
    pred_path = pred_path or f'../model_outputs/pf_{dset}_pie.csv'
    predictions = get_predictions(pred_path)
    # We se the prediction horizon similar to PedFormer which was used as the default for all evaluations
    # Note that if prediction horizon changes, even though the number of samples and observation sequences
    # are identical, scenario analysis can change significantly. This is because scenario statistics are generated
    # according to full tracks (obs+pred) NOT only the observation portion
    configs['data_gen_opts']['pred_len'] = 30 # 1s or 30 frames
    configs['data_gen_opts']['time_to_event'] = [30, 90]
    configs['data_gen_opts']['overlap'] = 0.3
    imdb = get_dataset(configs['data_gen_opts']['dataset'])
    data_raw = imdb.generate_data_trajectory_sequence('test', **configs['dataset_opts'])
    data = get_tracks(data_raw, configs['data_gen_opts'], subset='test')
    if dset == 'risk':
        gt = data['risk_class'][:, 0, :]
    elif dset == 'action':
        gt = data['activities'][:, 0, :]
    else:
        gt = data['intention'][:, 0, :]
    return evaluate_action_scenario(predictions,
                                          gt,
                                          configs=configs, data=data,
                                          scenarios=configs['scenarios']['act_sf'],
                                          verbose=True)


evaluate_scenario_trajectory()
evaluate_scenario_action()
