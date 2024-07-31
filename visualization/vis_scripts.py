import sys
import os
sys.path.append('../')
from utilities.data_gen_utils import get_tracks, get_dataset
from utilities.utils import get_config, read_pickle
from scenarioEval.scenario_generator import get_scenarios
from visualization.visualize import visualize_data_with_graph_video, visualize_beh_data
from visualization.model_visualize import model_vis_img
from visualization.model_visualize import model_vis_continuous_traj, model_vis_continuous_act, model_vis_continuous_mt

# Assign environment paths for datasets if already not defined
if 'JAAD_PATH' not in os.environ:
    os.environ['JAAD_PATH'] = '../../JAAD'
if 'PIE_PATH' not in os.environ:
    os.environ['PIE_PATH'] = '../../../PIE'

# Path to the configurations
config_path = '../utilities/configs.yaml'


# Visualization scripts
def data_visualize_behaviors():
    """
    Visualizes behavioral tags for given pedestrian sequences in the data
    """
    # NOTE: accel and decel for visualization on PIE needs tuning. They are not accurate at the moment
    configs = get_config()
    configs['data_gen_opts']['dataset'] = 'PIE'
    # Visualize samples video_0010 in set03
    configs['vis_config']['set_ids'] = ['set03']
    configs['vis_config']['vid_ids'] = ['video_0010']
    imdb = get_dataset(configs['data_gen_opts']['dataset'])
    data_raw = imdb.generate_data_trajectory_sequence('all', **configs['dataset_opts'])
    visualize_beh_data(data_raw, configs)


def visualize_data_label():
    """
    Script demo that plots data label for a given pedestrian instance sequence.
    Only works on PIE
    """
    configs = get_config(config_path)
    imdb = get_dataset(configs['data_gen_opts']['dataset'])
    data = imdb.generate_data_trajectory_sequence('all', **configs['dataset_opts'])

    # Visualize only the first pedestrian sequence from video_0009 in set04
    configs['vis_config']['set_ids'] = ['set04']
    configs['vis_config']['vid_ids'] = ['video_0009']
    # if vid_ids is null, global ids are used
    configs['vis_config']['seq_ids'] = [0]
    # Alternatively one can visualize according to pedestrian ids.
    # A pedestrian id has precedent over other ids (video or set), i.e. if set,
    # other ids become irrelevant
    # configs['vis_config']['ped_ids'] = ['3_8_526'] 
    configs['graph_config']['vis_data_h'] = 'gyro'
    # For label types with more than one component select the index.
    # For gyro this will be Y
    configs['graph_config']['vis_data_h_index'] = 1
    # To draw a 3D plot with two labels, choose additional label as z axis
    # configs['graph_config']['vis_data_z'] = 'obd_speed'
    visualize_data_with_graph_video(data, configs['vis_config'],
                                    configs['graph_config'])


def visualize_data_label_scenario():
    """
    Script demo that plots data label for a given pedestrian instance sequence in a scenario
    Only works for PIE dataset
    """
    configs = get_config(config_path)
    imdb = get_dataset(configs['data_gen_opts']['dataset'])
    subset = 'test'
    data_raw = imdb.generate_data_trajectory_sequence(subset, **configs['dataset_opts'])
    configs['data_gen_opts']['time_to_event'] = []
    data = get_tracks(data_raw, configs['data_gen_opts'], subset=subset)

    # Grab data_ids corresponding to a give scenario
    # data_ids are in the form of {'sub_scen1':<ids>,'sub_scen2':<ids>,...}
    data_ids, scenario = get_scenarios(data, 'veh_speed', {'speeds': [0, 5, 10, 20, 30]}, True)

    # Choose only the first sequence of sub-scenario
    # Since scenarios are extracted globally, global sequence numbers are used
    configs['vis_config']['set_ids'] = configs['vis_config']['vid_ids'] = None
    configs['vis_config']['seq_ids'] = data_ids['10-20'][0:1]
    configs['graph_config']['vis_data_h'] = 'speed'
    # To draw 3D plot with two labels, choose additional label as z axis
    # configs['graph_config']['vis_data_z'] = 'speed'
    visualize_data_with_graph_video(data, configs['vis_config'],
                                    configs['graph_config'])


def model_visualize_traj_image(model_path='model_dir/model_pred.pkl'):
    """
    Generates images with results of a trajectory prediction model.
    The output are images at time t of each sequence showing the predicted and
    ground-truth trajectories
    """
    configs = get_config(config_path)
    #Here the assumption is the output is recorded as a pickle file
    # The output is in the format of
    # array([[bbox_t1], [bbox_t2], ..., [bbox_tn]]^1...
    # [[bbox_t1], [bbox_t2], ..., [bbox_tn]]^k)
    # where n is the length of the prediction and k is the total number of samples. 
    # i.e. an ordered list of predictions
    # NOTE: the results are based on an ordered list of sequences and should 
    # match the order of generated data  
    model_output = read_pickle(model_path)
    configs['vis_config']['save_path'] = 'demo_images'
    # When set to false, images are saved
    configs['vis_config']['save_as_video'] = False
    # Display trajectories as pedestrian bounding box at time t and
    # final predicted box along with the ground truth
    # For full trajectories, 'circle' and 'line' options can be used
    configs['vis_config']['traj_mode'] = 'box'
    # color of predicted and gt boxes/points
    configs['vis_config']['traj_color'] = {'predict': 'blue', 'gt': 'green'}
    # Visualize only the first pedestrian sequence from video_0008 in set03
    configs['vis_config']['set_ids'] = ['set03']
    configs['vis_config']['vid_ids'] = ['video_0008']

    # Model observation/prediction lengths
    configs['data_gen_opts']['obs_len'] = 15
    configs['data_gen_opts']['pred_len'] = 45
    configs['data_gen_opts']['overlap'] = 0.5 # sample overlap for tracks extracted from pedestrian sequences
    configs['data_gen_opts']['time_to_event'] = []

    imdb = get_dataset(configs['data_gen_opts']['dataset'])
    data_raw = imdb.generate_data_trajectory_sequence('test', **configs['dataset_opts'])
    data = get_tracks(data_raw, configs['data_gen_opts'], subset='test')
    model_vis_img(model_output, data, configs)


# Generate continuous demos on videos with a moving window
def model_visualize_cont_traj(model_path='model_dir/model.h5'):
    """
    Generates a demo video of predicted trajectories. The demo is generated by feeding
    a continuous stream of sequences (using a moving window) to the model. All available
    samples in the frames are displayed. NOTE: A model is required.
    """
    configs = get_config(config_path)
    imdb = get_dataset(configs['data_gen_opts']['dataset'])
    beh_seq_test = imdb.generate_data_trajectory_sequence('test', **configs['dataset_opts'])

    # Data fields to be extracted. It should be set according to the model's input and output
    data_fields = ['image', 'bbox', 'pid']
    configs['vis_config']['save_as_video'] = True
    # Only record up to 100 frames of each video. If None, all will be recorded
    configs['vis_config']['max_count'] = 100

    # Set up for displaying trajectory points (see configs.yaml for details)
    configs['vis_config']['traj_mode'] = 'circle'
    # Color of predicted and gt boxes/points
    configs['vis_config']['traj_color'] = {'predict': 'blue', 'gt': 'green'}
    # Visualize samples video_0009 in set03
    configs['vis_config']['set_ids'] = ['set03']
    configs['vis_config']['vid_ids'] = ['video_0009']
    model_vis_continuous_traj(beh_seq_test, data_fields, model_path=model_path, configs=configs)


def model_visualize_cont_act(model_path='model_dir/model.h5'):
    """
    Generate demo video for predicted actions. The color of pedestrian bounding shows whether the
    pedestrian will cross (red) or not (green). Predictions are shown as smaller boxes on the top of
    bounding boxes. Depending on the prediction confidence, the intensity of red or green colors changes.
    Depending on time_to_event value predictions may var (see configs.yaml for details). NOTE: A model is required.
    """
    configs = get_config(config_path)
    imdb = get_dataset(configs['data_gen_opts']['dataset'])
    beh_seq_test = imdb.generate_data_trajectory_sequence('test', **configs['dataset_opts'])

    # Data fields to be extracted. It should be set according to the model's input and output
    data_fields = ['image', 'bbox', 'activities', 'event_frames', 'pid']
    # Time to event, can either be [num1, num2], num, or null (or empty list)(continuously predicts and displays)
    configs['data_gen_opts']['time_to_event'] = [30, 90]
    configs['vis_config']['save_as_video'] = True
    # Visualize samples in video_0010 in set03
    configs['vis_config']['set_ids'] = ['set03']
    configs['vis_config']['vid_ids'] = ['video_0010']
    # Only record up to 100 frames of each video. if None, all will be recorded
    configs['vis_config']['max_count'] = 100
    model_vis_continuous_act(beh_seq_test, data_fields, model_path=model_path, configs=configs)


def model_visualize_cont_mt(model_path='model_dir/model.h5'):
    """
    Generates demo video for multi predictions, namely action and trajectory.
    The output is the combination of both trajectory and action demos. NOTE: A model is required.
    """
    configs = get_config(config_path)
    imdb = get_dataset(configs['data_gen_opts']['dataset'])
    beh_seq_test = imdb.generate_data_trajectory_sequence('test', **configs['dataset_opts'])

    # Data fields to be extracted. It should be set according to the model input and output
    data_fields = ['image', 'bbox', 'activities', 'event_frames', 'pid']
    configs['data_gen_opts']['time_to_event'] = [30, 90]
    configs['data_gen_opts']['obs_input_type'] = ['norm_bbox', 'bbox']
    configs['data_gen_opts']['pred_len'] = 30
    # Visualize samples video_0010 in set03
    configs['vis_config']['set_ids'] = ['set03']
    configs['vis_config']['vid_ids'] = ['video_0010']
    # Only record up to 100 frames of each video. if None, all will be recorded
    configs['vis_config']['max_count'] = 200
    model_vis_continuous_mt(beh_seq_test, data_fields, model_path=model_path, configs=configs)


visualize_data_label()
# visualize_data_label_scenario()
# model_visualize_traj_image()
# model_visualize_cont_traj()
# model_visualize_cont_act()
# model_visualize_cont_mt()
# data_visualize_behaviors()
