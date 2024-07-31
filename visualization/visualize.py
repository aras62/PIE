import cv2
import os
import copy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from utilities.utils import update_progress, _check_data_ids, print_msg
from utilities.utils import assertion


def visualize_data_with_graph_video(data, vis_config=None, graph_config=None):
    """
    Creates a video based on the results of a single model
    Args:
        data: data
        vis_config: visualization configurations
        graph_config: graph configurations

    Returns:
        None
    """
    # Adjust the size of the images. If change dimension of image, this should be changed
    img = cv2.imread(data['image'][0][0], flags=cv2.IMREAD_COLOR)
    vis_config['frame_size'] = (int(img.shape[1] * vis_config['scale']), img.shape[0])
    value_plot = _get_graph_data(data, 'vis_data_h', graph_config)

    # Get additional data type if any
    value_plot_z = None
    if graph_config['vis_data_z']:
        value_plot_z = _get_graph_data(data, 'vis_data_z', graph_config)
        for i in range(len(value_plot)):
            for j in range(len(value_plot[i])):
                value_plot[i][j].extend(value_plot_z[i][j])
        file_ext = f"{graph_config['vis_data_h']}_{graph_config['vis_data_z']}"
    else:
        file_ext = graph_config['vis_data_h']

    if vis_config['txt_ext']:
        file_ext = f"{file_ext}_{vis_config['txt_ext']}"

    seqid_cnt = ['', '', -1]  # set, video, seqid
    root_path = vis_config['save_path']
    for g_seqid, (img_seq, bbox_seq, pedid_seq, val_seq) in enumerate(
            zip(data['image'], data['bbox'], data['pid'], value_plot)):
        vis_data, save_path = _check_data_ids(img_seq[0], root_path, seqid_cnt, g_seqid, vis_config,
                                              pedid_seq[0][0], file_ext)
        if not vis_data:
            continue
        if os.path.isfile(f"{save_path}.{vis_config['video_format']}") and not vis_config['vis_regen']:
            print_msg(f"{save_path}.{vis_config['video_format']} already exists!", 'yellow')
            continue
        vis_config['save_path'] = save_path
        counter = 0
        # Get sequence length
        update_len = graph_config['seq_len'] = vis_config['max_count'] or len(img_seq)

        # Create raw graph
        img_graph = draw_3D_graph(graph_config) if value_plot_z else draw_graph(img.shape, graph_config)

        # Create visualize object
        vis_obj = VisualizeObj(vis_config)
        for t, (img_path, bbox, value) in enumerate(zip(img_seq, bbox_seq, val_seq)):
            counter += 1
            update_progress(counter / update_len)
            vis_obj.visualize_data_graph(img_path, img_graph, bbox, value, t, graph_config)
            if vis_config['max_count'] is not None:
                if counter == vis_config['max_count']:
                    vis_obj.finish_visualize()
                    break
        vis_obj.finish_visualize()


def visualize_beh_data(data, configs):
    """
    Visualizes the behavioral labels of the data in the form of a dynamic graphics
    Args:
        data: data
        configs: configurations

    Returns:
        None
    """
    vis_config = configs['vis_config']
    graph_config = configs['beh_graph_config']
    img_ref = cv2.imread(data['image'][0][0], flags=cv2.IMREAD_COLOR)
    vis_config['frame_size'] = (
    int(img_ref.shape[1] * vis_config['scale']), img_ref.shape[0] - graph_config['top_margin'])
    # Dimensions
    gw = img_ref.shape[1]
    gh = img_ref.shape[0] - graph_config['top_margin']
    vid_root = vis_config['save_path']
    dataset = configs['data_gen_opts']['dataset']
    if vis_config['seq_ids'] is not None:
        vis_config['seq_ids'] = None
        print_msg("Sequence ids are not applicable for behavior data demo! They are ignored.", 'yellow')
    for sidx, img_seq in enumerate(data['image']):
        pid = data['pid'][sidx][0][0]
        vis_data, save_path = _check_data_ids(img_seq[0], vid_root,
                                              vis_config=vis_config,
                                              ped_id=pid,
                                              file_ext='beh_demo')
        if not vis_data:
            continue
        vis_config['save_path'] = save_path

        if dataset.lower() == 'jaad' and graph_config['beh_only'] and 'b' not in pid:
            continue
        if os.path.isfile(f"{save_path}.{vis_config['video_format']}") and not vis_config['vis_regen']:
            print_msg(f"{save_path}.{vis_config['video_format']} already exists!", 'yellow')
            continue
        vis_obj = VisualizeObj(vis_config)
        graph_config['seq_len'] = len(img_seq)
        img_graph = np.full((gh, gw, 3), [255, 255, 255], dtype=np.uint8)
        # Create the graph
        vert_margin, labels = _draw_lines_data_graph(img_graph, gh, gw, graph_config,
                                                     ped_beh=False if dataset.lower() == 'jaad' and 'b' not in pid else True)
        _draw_horiz_graph_axis(img_graph, gh, gw, graph_config)
        _add_horiz_axis_label(img_graph, gh, gw, graph_config)
        _add_vertical_labels(img_graph, gh, 'Pedestrian', (vert_margin[0] + vert_margin[1]) // 2, graph_config)
        _add_vertical_labels(img_graph, gh, 'Driver', (vert_margin[1] + vert_margin[2]) // 2, graph_config)
        for idx, img_path in enumerate(img_seq):
            update_progress(idx / len(img_seq))
            vis_obj.visualize_beh_data_graph(img_path, img_graph,
                                             data, labels,
                                             sidx, idx, graph_config, b_color='blue')
        vis_obj.finish_visualize()


def draw_bars_on_graph(img_graph, t, h_pos, graph_config, color):
    """
    Maps a single point based on value and time and draws on the graph
    Args:
        img_graph: graph
        t: timestep
        h_pos: horizontal position
        graph_config: graph configuration
        color: colors of the graph elements

    Returns:
        None
    """
    gw = img_graph.shape[1]
    gh = img_graph.shape[0]

    min_w_pt = graph_config['w_border']
    max_w_pt = gw - graph_config['w_border']
    max_w_value = graph_config['seq_len'] / graph_config['seq_frate']  # assumption 30hz data collection
    w_pt_map = int(((t / graph_config['seq_frate']) * (max_w_pt - min_w_pt)) / max_w_value + min_w_pt)
    cv2.rectangle(img_graph, (w_pt_map, h_pos - graph_config['bar_width'] // 2),
                  (w_pt_map + (max_w_pt - min_w_pt) // graph_config['seq_len'],
                   h_pos + graph_config['bar_width'] // 2),
                  color, -1)


def _draw_lines_data_graph(img_graph, gh, gw, graph_config, ped_beh=True):
    """
    Draws the horizontal lines on the graph
    Args:
        img_graph: the graph image to draw on
        gh: graph height
        gw: graph width
        graph_config: graph configuration
        ped_beh: only for JAAD dataset to display the samples with behavioral tags

    Return:
        final left margin for the text on the vertical axis
    """
    # Heigth (vertical) step size
    _labels = {'pedestrian': {'cross': 0, 'walk': 0, 'look': 0, 'gest': 0},
               'driver': {'accel': 0, 'decel': 0, 'stop': 0}}
    labels = list(_labels['pedestrian'].keys()) + list(_labels['driver'].keys())
    h_st_size = (gh - 2 * graph_config['h_border']) // (len(labels) + 1)  # for

    lb = len(labels) - 1
    lower_margin = gh - graph_config['h_border']
    # Margins for putting vertical axis label
    vert_margin = (lower_margin - (len(labels) + 1) * h_st_size,
                   lower_margin - (len(_labels['driver'].keys()) + 1) * h_st_size,
                   lower_margin)
    graph_clr = graph_config['graph_clr']
    # Draw horizontal lines
    for idx, i in enumerate(range(lower_margin, graph_config['h_border'] - 1, - h_st_size)):  # Draw from bottom up
        # Draw the full line
        if (idx == len(_labels['driver'].keys()) + 1 and not draw_mid_ln) or idx == 0:
            cv2.line(img_graph, (graph_config['w_border'], i),
                     (gw - graph_config['w_border'], i),
                     graph_clr,
                     graph_config['axis_ln_width'])
            if idx == len(_labels['driver'].keys()) + 1:
                if not ped_beh:
                    graph_clr = (70, 22, 224)

            draw_mid_ln = True if idx == 3 else False
            continue
        cv2.line(img_graph, (graph_config['w_border'], i),
                 (gw - graph_config['w_border'], i),
                 graph_clr,
                 graph_config['hor_ln_width'])
        # Draw tick line
        cv2.line(img_graph, (graph_config['w_border'] - graph_config['tick_ln_len'], i),
                 (graph_config['w_border'], i),
                 graph_clr, graph_config['axis_ln_width'])

        # Shift the text to left so it is centered at tick
        (tw, th), _ = cv2.getTextSize(labels[lb], cv2.FONT_HERSHEY_COMPLEX_SMALL, graph_config['text_scale'], 1)
        txt_margin = tw + graph_config['tick_txt_margin']
        cv2.putText(img_graph, labels[lb],
                    (graph_config['w_border'] - graph_config['tick_ln_len'] - txt_margin, i + (th // 2)), \
                    cv2.FONT_HERSHEY_COMPLEX_SMALL, graph_config['text_scale'], graph_clr, 1)
        if labels[lb] in _labels['pedestrian']:
            _labels['pedestrian'][labels[lb]] = i
        else:
            _labels['driver'][labels[lb]] = i
        lb -= 1
    return vert_margin, _labels


def _add_vertical_labels(img_graph, gh, text, center, graph_config):
    """
    Adds a label to vertical axis. The label is crated on an image, rotated and imposed into the graph image
    Args:
        img_graph: graph image to draw on
        gh: graph height
        text: labels for vertical axis
        center: center point of the graph element
        graph_config: graph configuration

    Returns:
        None
    """
    # Add label to vertical axis
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_COMPLEX_SMALL, graph_config['text_scale'], 1)
    th += baseline  # a small margin to make sure the text appears properly
    text_gr = np.full((th, tw, 3), [255, 255, 255], dtype=np.uint8)
    # Text coordinate is bottom left
    cv2.putText(text_gr, text, (0, th - baseline),
                cv2.FONT_HERSHEY_COMPLEX_SMALL, graph_config['text_scale'],
                graph_config['graph_clr'], 1)
    text_gr_rot = np.rot90(text_gr, 1, (0, 1))  # rotate the text by 90 degrees
    x2 = th + graph_config['axis_txt_margin']  # how far from the axis
    x1 = graph_config['axis_txt_margin']
    y1 = center - tw // 2  # gh//2
    y2 = center + tw // 2 if (tw % 2) == 0 else center + tw // 2 + 1
    img_graph[y1:y2, x1:x2, :] = text_gr_rot


### Utils ###
def _get_graph_data(data, data_key, graph_config):
    """
    Extracts the data sequence for graph visualization
    Args:
        data: all data
        data_key: data for a given axis in the graph
        graph_config: configurations for the graph

    Returns:
        Values to be plotted
    """
    vis_data = graph_config[data_key]
    if vis_data in ['speed', 'obd_speed']:
        vis_data = 'obd_speed'
        if vis_data not in data:
            ValueError(f"The dataset does not have speed label")
    if not isinstance(data[vis_data], list):
        data[vis_data] = data[vis_data].tolist()

    # For data types with more than 1 element, e.g. gyro, select the given value
    if len(data[vis_data][0][0]) > 1:
        vis_idx = graph_config.get(f"{data_key}_index", 0)
        assertion(vis_idx < len(data[vis_data][0][0]), f"Data index {vis_idx} is out of range")
        value_plot = copy.deepcopy(data[vis_data])
        for i in range(len(value_plot)):
            for j in range(len(value_plot[i])):
                value_plot[i][j] = [value_plot[i][j][vis_idx]]
        idx_txt = f"{'yrp'[vis_idx]}" if vis_data == 'yrp' else f"{['x', 'y', 'z'][vis_idx]}"
        graph_config[data_key] = f"{vis_data}_{idx_txt}"
    else:
        value_plot = data[vis_data]

        # Set the min and max of axis values
    graph_config[f"{data_key}_max"] = graph_config[f"{data_key}_max"] or max(max(value_plot))[0]
    graph_config[f"{data_key}_min"] = graph_config[f"{data_key}_min"] or min(min(value_plot))[0]
    return value_plot


### 2D Graph ###
def draw_graph(graph_shape, graph_config):
    """
    Draws a raw graph to be attached to the data image
    Args:
        graph_shape: shape of the graph
        graph_config: configs for the graph specifications

    Returns:
        Raw graph image
    """
    img_graph = np.full(graph_shape, [255, 255, 255], dtype=np.uint8)

    # Dimensions
    gw = img_graph.shape[1]
    gh = img_graph.shape[0]

    h_txt_margin = _draw_horiz_graph_lines(img_graph, gh, gw, graph_config)

    # Draw vertical axis
    cv2.line(img_graph, (graph_config['w_border'], graph_config['h_border']),
             (graph_config['w_border'], gh - graph_config['h_border']),
             graph_config['graph_clr'],
             graph_config['axis_ln_width'])

    _add_vertical_axis_label(img_graph, gh, h_txt_margin, graph_config)

    # Draw horizontal axis
    cv2.line(img_graph, (graph_config['w_border'], gh - graph_config['h_border']),
             (gw - graph_config['w_border'], gh - graph_config['h_border']),
             graph_config['graph_clr'],
             graph_config['axis_ln_width'])
    _draw_horiz_graph_axis(img_graph, gh, gw, graph_config)
    _add_horiz_axis_label(img_graph, gh, gw, graph_config)
    return img_graph


# Graph draw helpers
def _draw_horiz_graph_lines(img_graph, gh, gw, graph_config):
    """
    Draws the Horizontal lines on the graph
    Args:
        img_graph: the graph image to draw on
        gh: graph height
        gw: graph width
        graph_config: graph configuration

    Returns:
        Final left margin for the text on the vertical axis
    """
    # Height (vertical) step size
    h_st_size = (gh - 2 * graph_config['h_border']) // graph_config['h_ax_step']

    # Values for ticks and text
    step_val = (graph_config['vis_data_h_max'] - graph_config['vis_data_h_min']) / graph_config['h_ax_step']
    tick_tag = graph_config['vis_data_h_min']
    tick_step = (gh - 2 * graph_config['h_border']) // graph_config['h_ax_step']

    # Draw horizontal lines
    for i in range(gh - graph_config['h_border'], graph_config['h_border'] - 1, - h_st_size):  # Draw from bottom up
        # Draw the full line
        cv2.line(img_graph, (graph_config['w_border'], i),
                 (gw - graph_config['w_border'], i),
                 graph_config['graph_clr'],
                 graph_config['hor_ln_width'])

        # Draw tick line
        cv2.line(img_graph, (graph_config['w_border'] - graph_config['tick_ln_len'], i),
                 (graph_config['w_border'], i),
                 graph_config['graph_clr'], graph_config['axis_ln_width'])

        # Shift the text to left so it is centered at tick
        text = f"{tick_tag:.{graph_config['h_txt_decimal']}f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_COMPLEX_SMALL, graph_config['text_scale'], 1)
        txt_margin = tw + graph_config['tick_txt_margin']

        cv2.putText(img_graph, text,
                    (graph_config['w_border'] - graph_config['tick_ln_len'] - txt_margin, i + (th // 2)), \
                    cv2.FONT_HERSHEY_COMPLEX_SMALL, graph_config['text_scale'], graph_config['graph_clr'], 1)
        tick_tag += step_val
    return txt_margin


def _add_vertical_axis_label(img_graph, gh, h_txt_margin, graph_config):
    """
    Adds a label to vertical axis. The label is crated on an image, rotated and imposed into the graph image.
    Args:
        img_graph: graph image to draw on
        gh: graph height
        h_txt_margin: text margin (tick texts)
        graph_config: graph configuration

    Returns:
        None
    """

    # Add label to vertical axis
    h_axis_text = graph_config['h_axis_txt'] or graph_config['vis_data_h']
    (tw, th), baseline = cv2.getTextSize(h_axis_text, cv2.FONT_HERSHEY_COMPLEX_SMALL, graph_config['text_scale'], 1)
    th += baseline  # a small margin to make sure the text appears properly
    h_text_gr = np.full((th, tw, 3), [255, 255, 255], dtype=np.uint8)
    # Text coordinate is bottom left
    cv2.putText(h_text_gr, h_axis_text, (0, th - baseline),
                cv2.FONT_HERSHEY_COMPLEX_SMALL, graph_config['text_scale'],
                graph_config['graph_clr'], 1)
    h_text_gr_rot = np.rot90(h_text_gr, 1, (0, 1))  # rotate the text by 90 degrees
    x2 = graph_config['w_border'] - graph_config['tick_ln_len'] - h_txt_margin - graph_config[
        'axis_txt_margin']  # how far from the axis
    x1 = x2 - th
    y1 = gh // 2 - tw // 2
    y2 = gh // 2 + tw // 2 if (tw % 2) == 0 else gh // 2 + tw // 2 + 1
    img_graph[y1:y2, x1:x2, :] = h_text_gr_rot


def _draw_horiz_graph_axis(img_graph, gh, gw, graph_config):
    """
    Draws the horizontal axis and adds ticks and text
    Args:
        img_graph: the graph image to draw on
        gh: graph height
        gw: graph weight
        graph_config: graph configuration

    Returns:
        None
    """
    # Draw horizontal axis unit tick lines
    time = graph_config['seq_len'] / graph_config['seq_frate']  # assumption 30hz data collection
    time /= graph_config['w_ax_step']
    tick_tag = 0
    tick_step = (gw - 2 * graph_config['w_border']) // graph_config['w_ax_step']

    for t in range(graph_config['w_border'],
                   gw - graph_config['w_border'] + tick_step, tick_step):
        cv2.line(img_graph, (t, gh - graph_config['h_border']),
                 (t, gh - graph_config['h_border'] + graph_config['tick_ln_len']),
                 graph_config['graph_clr'], graph_config['axis_ln_width'])
        # Shift the text to left so it is centered at tick
        text = f"{tick_tag:.{graph_config['w_txt_decimal']}f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_COMPLEX_SMALL, graph_config['text_scale'], 1)

        cv2.putText(img_graph, text, (t - (tw // 2), gh - graph_config['h_border'] +
                                      graph_config['tick_ln_len'] + graph_config['tick_txt_margin'] + th), \
                    cv2.FONT_HERSHEY_COMPLEX_SMALL,
                    graph_config['text_scale'],
                    graph_config['graph_clr'], 1)
        tick_tag += time


def _add_horiz_axis_label(img_graph, gh, gw, graph_config):
    """
    Adds a label to horizontal axis.
    Args:
        img_graph: graph image to draw on
        gh: graph height
        gw: graph weight
        graph_config: graph configuration

    Returns:
        None
    """
    (tw, th), _ = cv2.getTextSize(graph_config['w_axis_txt'], cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                  graph_config['text_scale'], 1)
    y = gh - graph_config['h_border'] + graph_config['tick_ln_len'] + \
        graph_config['tick_txt_margin'] + th * 2 + graph_config[
            'axis_txt_margin']  # th*2 for both tick values and axis text
    cv2.putText(img_graph, graph_config['w_axis_txt'], (gw // 2 - tw // 2, y),
                cv2.FONT_HERSHEY_COMPLEX_SMALL, graph_config['text_scale'],
                graph_config['graph_clr'], 1)


def draw_point_on_2D_graph(img_graph, value, t, graph_config):
    """
    Maps a single point based on value and time and draws it on the graph
    Args:
        img_graph: graph image
        value: value to be mapped to height axis
        t: timestep
        graph_config: graph configuration

    Returns:
        None
    """
    gw = img_graph.shape[1]
    gh = img_graph.shape[0]

    min_h_pt = graph_config['h_border']
    max_h_pt = gh - graph_config['h_border']

    # Shift with min value
    mapped_value = value + abs(graph_config['vis_data_h_min'])
    max_h_value = graph_config['vis_data_h_max'] + abs(graph_config['vis_data_h_min'])
    mapped_value = max_h_value - mapped_value  # flip due to changes in the coordinates
    h_pt_map = int((mapped_value * (max_h_pt - min_h_pt)) / max_h_value + min_h_pt)

    min_w_pt = graph_config['w_border']
    max_w_pt = gw - graph_config['w_border']
    max_w_value = graph_config['seq_len'] / graph_config['seq_frate']  # assumption 30hz data collection
    w_pt_map = int(((t / graph_config['seq_frate']) * (max_w_pt - min_w_pt)) / max_w_value + min_w_pt)
    cv2.circle(img_graph, (w_pt_map, h_pt_map), 5, graph_config['pt_clr'], -1)


### 3D Graph ###
def draw_3D_graph(graph_config):
    """
    Creates a 3D graph figure
    Args:
        graph_config: graph configurations

    Returns:
        Graph object handler
    """
    fig = plt.figure(figsize=(19.2, 10.8))
    fig.tight_layout()
    ax = fig.add_subplot(projection='3d')

    # X axis
    ax.set_xlabel(graph_config['w_axis_txt'])
    time = graph_config['seq_len'] / graph_config['seq_frate']  # assumption 30hz data collection
    ax.set_xlim(0, time)
    x_step = time / graph_config['w_ax_step']
    ax.set_xticks([x_step * i for i in range(graph_config['w_ax_step'] + 1)])
    ax.xaxis.set_tick_params(rotation=45, labelsize=8, pad=-6)
    dec = f"%.{graph_config['w_txt_decimal']}f"
    ax.xaxis.set_major_formatter(FormatStrFormatter(dec))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Draw y and z axes
    _setup_graph_axis(ax, 'vis_data_h', graph_config, pad=-4, rotation=-15)
    _setup_graph_axis(ax, 'vis_data_z', graph_config, pad=2, rotation=0)
    return fig, ax


# Graph draw helpers
def draw_point_on_3D_graph(img_graph, value, t, graph_config):
    """
    Maps a single point based on value and time
    Args:
        img_graph: 3D graph file (fig,ax)
        value: value to be mapped to height axis
        t: timestep
        graph_config: graph configuration

    Returns:
        Graph image
    """
    fig, ax = img_graph
    ax.scatter(t / graph_config['seq_frate'], value[0], value[1], marker='o', c='blue')
    fig.canvas.draw()
    rgba_buf = fig.canvas.buffer_rgba()
    (w, h) = fig.canvas.get_width_height()
    rgba_arr = np.frombuffer(rgba_buf, dtype=np.uint8).reshape((h, w, 4))
    rgba_arr[:, :, :3] = np.flip(rgba_arr[:, :, :3], 2)
    return rgba_arr[:, :, :3]


def _setup_graph_axis(ax, data_key, graph_config, lbl_size=8, pad=0, rotation=0):
    """
    Sets up the axis parameters
    Args:
        ax: axis handler
        data_key: data keys
        graph_config: graph configurations
        lbl_size: size of labels
        pad: amount of padding
        rotation: rotation

    Returns:
        None
    """
    ## y axis
    if 'h' in data_key:
        axis_text = 'h_axis_txt'
        axis_step = 'h_ax_step'
        lable_fun = ax.set_ylabel
        lim_fun = ax.set_ylim
        ticks_fun = ax.set_yticks
        param_fun = ax.yaxis.set_tick_params
        plt.setp(ax.yaxis.get_majorticklabels(), rotation=rotation, ha="left", rotation_mode="anchor")
    else:
        axis_text = 'z_axis_txt'
        axis_step = 'z_ax_step'
        lable_fun = ax.set_zlabel
        lim_fun = ax.set_zlim
        ticks_fun = ax.set_zticks
        param_fun = ax.zaxis.set_tick_params

    _label = graph_config[axis_text] or graph_config[data_key]
    lable_fun(_label)
    _step = (graph_config[f'{data_key}_max'] - graph_config[f'{data_key}_min']) / graph_config[axis_step]
    lim_fun(graph_config[f'{data_key}_min'], graph_config[f'{data_key}_max'])
    ticks = []
    tick_tag = graph_config[f'{data_key}_min']
    for i in range(graph_config[axis_step] + 1):
        ticks.append(tick_tag)
        tick_tag += _step
    ticks_fun(ticks)
    param_fun(rotation=rotation, labelsize=lbl_size, pad=pad)


class VisualizeObj(object):
    """
    The visualization class
    """

    def __init__(self, configs=None):
        # BGR
        self.colors = {'yellow': (32, 223, 236), 'green': (14, 241, 13),
                       'green_box': (71, 220, 69), 'orange': (11, 144, 243),
                       'cyan': (232, 222, 22), 'brown': (131, 157, 199),
                       'red': (70, 22, 224), 'purple': (224, 30, 196),
                       'blue': (227, 76, 19)}  # 79, 62, 213
        self.scale = 0.5
        self.intensity_base = 200
        self.video_format = 'mp4'
        self.codec = 'mp4v'
        self.image_format = 'png'
        self.frame_rate = 30.0
        self.model_srate = 1
        self.video_obj = None
        self.frame_size = None
        self.save_results = False
        self.save_as_video = False
        self.max_traj_length = None
        self.save_path = 'video'
        self.add_text = False
        self.line_width = 3
        self.box_width = 3
        if configs is not None:
            self.__dict__.update(configs)
        if self.save_results:
            if self.save_as_video:
                self.create_video_writer_obj()
            elif not os.path.isdir(self.save_path):
                os.makedirs(self.save_path)

    def visualize_data_graph(self, img_path, img_graph, bbox, value, t, graph_config, b_color='blue'):
        """
        Draws graph points
        Args:
            img_path: path to the image
            img_graph: graph plane
            bbox: bounding box coordinate
            value: graph value
            t: index of the data point
            graph_config: graph configuration
            b_color: bounding box configuration

        Returns:
            None
        """
        img = cv2.imread(img_path, flags=cv2.IMREAD_COLOR)
        self.draw_box_color(img, bbox, b_color)
        img = self.scale_image(img)
        if len(value) == 1:
            draw_point_on_2D_graph(img_graph, value[0], t, graph_config)
        else:
            img_graph = draw_point_on_3D_graph(img_graph, value, t, graph_config)

        img_graph = self.scale_image(img_graph)
        img = np.concatenate((img, img_graph), axis=0)
        frame_num = img_path.split('/')[-1].split('.')[0]
        self.save_result_output(img, frame_num)

    def visualize_beh_data_graph(self, img_path, img_graph, data, labels,
                                 sidx, t, graph_config, b_color='blue'):
        """
        Draws graph points
        Args:
            img_path: path to the image
            img_graph: graph plane
            data: data
            labels: labels
            sidx: sequence id
            t: index of the data point
            graph_config: graph configuration
            b_color: bounding box configuration

        Returns:
            None
        """
        mapping = {'walk': ['actions', 'orange'], 'look': ['looks', 'green'],
                   'cross': ['cross', 'red'], 'gest': ['gesture', 'brown'],
                   'stop': [0, 'blue'], 'accel': [4, 'yellow'], 'decel': [3, 'purple']}
        img = cv2.imread(img_path, flags=cv2.IMREAD_COLOR)
        self.draw_box_color(img, data['bbox'][sidx][t], b_color='blue')
        img = img[graph_config['top_margin']:, :, :]
        img = self.scale_image(img)
        for tlbl in labels:
            for lbl in labels[tlbl]:
                add_bar = False
                if tlbl == 'driver':
                    if 'vehicle_act' in data:
                        if data['vehicle_act'][sidx][t][0] == mapping[lbl][0]:
                            add_bar = True
                    else:
                        if lbl == 'stop' and data['obd_speed'][sidx][t][0] == 0:
                            add_bar = True
                        elif abs(data['acc'][sidx][t][0]) > graph_config.get('acc_thr', 0.3):
                            if (lbl == 'accel' and data['acc'][sidx][t][0] > 0) or \
                                    (lbl == 'decel' and data['acc'][sidx][t][0] < 0):
                                add_bar = True
                elif data[mapping[lbl][0]][sidx][t][0] > 0:
                    add_bar = True
                if add_bar:
                    draw_bars_on_graph(img_graph, t, labels[tlbl][lbl], graph_config,
                                       color=self.colors[mapping[lbl][1]])
        img_graph = self.scale_image(img_graph)
        img = np.concatenate((img, img_graph), axis=0)
        frame_num = img_path.split('/')[-1].split('.')[0]
        self.save_result_output(img, frame_num)

    # Util functions
    def save_result_output(self, image, frame_num=None):
        """
        Saves the results depending on setup in either video or image format
        Args:
            image: image array to write
            frame_num: id of the given image

        Returns:
            None
        """
        if self.save_results:
            if self.save_as_video:
                self.video_obj.write(image)
            else:
                cv2.imwrite(os.path.join(self.save_path, 'frame{}.{}'.format(frame_num, self.image_format)), image)

    def draw_box_color(self, img, bbox, b_color='green'):
        """
        Draw a box with a given color
        Args:
            img: image array
            bbox: bounding box
            b_color: bounding box color

        Returns:
            None
        """
        if isinstance(b_color, str):
            b_color = self.colors[b_color]

        bbox = list(map(int, bbox))
        # Get gt color for box
        cv2.rectangle(img, (bbox[0] - 4, bbox[1] - 4), (bbox[2] + 4, bbox[3] + 4),
                      b_color, self.box_width)

    def scale_image(self, img):
        """
        Scales the image array
        Args:
            img: image array

        Returns:
            Scaled image
        """
        # scale the image
        if self.scale != 1:
            newX, newY = int(img.shape[1] * self.scale), int(img.shape[0] * self.scale)
            img = cv2.resize(img, (newX, newY))
        return img

    def create_video_writer_obj(self):
        """
        Creates an opencv video object writer

        Returns:
            None
        """

        self.frame_size = self.frame_size or (int(1920 * self.scale), int(1080 * self.scale))

        if self.video_format == 'avi':
            self.codec = 'DIVX'
        elif self.video_format not in ['mp4', 'avi']:
            print_msg('Video format is not supported. Using default mp4', 'yellow')

        path_dir = os.path.dirname(self.save_path)
        if path_dir:
            os.makedirs(path_dir, exist_ok=True)
        self.video_obj = cv2.VideoWriter(f'{self.save_path}.{self.video_format}',
                                         cv2.VideoWriter_fourcc(*self.codec),
                                         self.frame_rate, self.frame_size)
        print_msg('Saving the result video to %s' % f'{self.save_path}.{self.video_format}', 'green')

    def finish_visualize(self):
        """
        Closes video object if one exists

        Returns:
            None
        """
        if self.video_obj is not None:
            print_msg("Video recording is complete", 'green')
            self.video_obj.release()

    def visualize_traj(self, img_path, fields, ped_annots, observe_data,
                       configs):
        """
        Draws trajectories and boxes around pedestrians in the image
        Args:
            img_path: path to the image
            fields: fields of visualization
            ped_annots: pedestrian annotations
            observe_data: observation data
            configs: configurations

        Returns:
            None
        """
        img = cv2.imread(img_path, flags=cv2.IMREAD_COLOR)
        for pid in observe_data:
            if observe_data[pid]['pred_traj'] is not None:
                # Draw trajectory
                self.draw_trajectories(img, fields, ped_annots[pid],
                                       observe_data[pid], configs)
        img = self.scale_image(img)
        frame_num = img_path.split('/')[-1].split('.')[0]
        self.save_result_output(img, frame_num)

    def visualize_act(self, img_path, fields, ped_annots, observe_data,
                      configs):
        """
        Draws action boxes around pedestrians in the image
        Args:
            img_path: path to the image
            fields: fields of visualization
            ped_annots: pedestrian annotations
            observe_data: observation data
            configs: configurations

        Returns:
            None
        """
        img = cv2.imread(img_path, flags=cv2.IMREAD_COLOR)
        for pid in observe_data:
            self.draw_boxes(img, observe_data[pid])
        img = self.scale_image(img)
        frame_num = img_path.split('/')[-1].split('.')[0]
        self.save_result_output(img, frame_num)

    def visualize_mt(self, img_path, fields, ped_annots, observe_data,
                     configs):
        """
        Draws trajectories and action boxes around pedestrians in the image for multitask predictions
        Args:
            img_path: path to the image
            fields: fields of visualization
            ped_annots: pedestrian annotations
            observe_data: observation data
            configs: configurations

        Returns:
            None
        """
        img = cv2.imread(img_path, flags=cv2.IMREAD_COLOR)
        for pid in observe_data:
            if observe_data[pid]['pred_traj'] is not None:
                # Draw trajectory
                self.draw_trajectories(img, fields, ped_annots[pid],
                                       observe_data[pid], configs)
            self.draw_boxes(img, observe_data[pid])
        img = self.scale_image(img)
        frame_num = img_path.split('/')[-1].split('.')[0]
        self.save_result_output(img, frame_num)

    def draw_trajectories(self, img, fields, ped_annots, observe_data, configs):
        """
        Draws pedestrian predicted and ground-truth trajectories
        Args:
            img: image array
            fields: fields used for visualization
            ped_annots: pedestrian annotations
            observe_data: observed data
            configs: configurations

        Returns:
            None
        """
        # Draw trajectory
        img_paths = ped_annots['image']
        pred_traj = observe_data['pred_traj']
        traj_pred_length = configs['data_gen_opts']['pred_len']

        # Do the future ground truth trajectories
        frame_idx = img_paths.index(fields['image'])
        gt_traj = ped_annots['bbox'][frame_idx + 1:frame_idx + traj_pred_length + 1:self.model_srate]
        a = ped_annots['bbox'][frame_idx + 1 + self.model_srate - 1:frame_idx + traj_pred_length + 1:self.model_srate]
        traj_range = len(gt_traj)
        if self.max_traj_length is not None:
            if self.max_traj_length < len(gt_traj):
                traj_range = self.max_traj_length
        self._draw_traj(img, traj_range, pred_traj, gt_traj, observe_data['gt_bbox'][-1],
                        configs, observe_data['color'])

    def _draw_traj(self, img, traj_range, pred_traj, gt_traj, ped_box, configs, ped_color=(227, 76, 19)):
        """
        Draws trajectory points on the image
        Args:
            img: image
            traj_range: range of trajectories to draw
            pred_traj: predicted trajectories
            gt_traj: ground-truth trajectories
            ped_box: pedestrian bounding boxes
            configs: configurations
            ped_color: pedestrian color

        Returns:
            None
        """
        draw_mode = configs['vis_config'].get('traj_mode', 'line')
        gt_color = self.colors[configs['vis_config']['traj_color'].get('gt', 'orange')]
        pred_color = self.colors[configs['vis_config']['traj_color'].get('predict', 'blue')]
        # the loop uses index so predictions beyond gt donot get included
        if draw_mode == 'circle':
            for i in range(traj_range):
                b_pred = pred_traj[i]
                b_gt = gt_traj[i]
                center_gt = (int((b_gt[2] + b_gt[0]) / 2), int((b_gt[3] + b_gt[1]) / 2))
                cv2.circle(img, center_gt, 5, gt_color, -1)
                center_pred = (int((b_pred[2] + b_pred[0]) / 2), int((b_pred[3] + b_pred[1]) / 2))
                img_copy = img.copy()
                cv2.circle(img_copy, center_pred, 5, pred_color, -1)
                alpha = 0.5
                cv2.addWeighted(img_copy, alpha, img, 1 - alpha, 0, img)

        if draw_mode == 'line':
            for i in range(traj_range - 1):
                # Draw gt trajectory
                b_gt = gt_traj[i]
                center_gt_1 = (int((b_gt[2] + b_gt[0]) / 2), int((b_gt[3] + b_gt[1]) / 2))
                b_gt = gt_traj[i + 1]
                center_gt_2 = (int((b_gt[2] + b_gt[0]) / 2), int((b_gt[3] + b_gt[1]) / 2))
                cv2.line(img, center_gt_1, center_gt_2, gt_color, self.line_width)

                # Draw predictions
                b_pred = pred_traj[i]
                center_pred_1 = (int((b_pred[2] + b_pred[0]) / 2), int((b_pred[3] + b_pred[1]) / 2))
                b_pred = pred_traj[i + 1]
                center_pred_2 = (int((b_pred[2] + b_pred[0]) / 2), int((b_pred[3] + b_pred[1]) / 2))
                img_copy = img.copy()
                cv2.line(img_copy, center_pred_1, center_pred_2, pred_color, self.line_width)
                alpha = 0.5
                cv2.addWeighted(img_copy, alpha, img, 1 - alpha, 0, img)

        if draw_mode == 'box' and traj_range > 0:
            bbox = list(map(int, ped_box))
            b_pred = list(map(int, pred_traj[traj_range - 1]))
            b_gt = list(map(int, gt_traj[traj_range - 1]))

            cv2.rectangle(img, (bbox[0] - 4, bbox[1] - 4), (bbox[2] + 4, bbox[3] + 4),
                          ped_color, self.box_width)
            cv2.rectangle(img, (b_pred[0] - 4, b_pred[1] - 4), (b_pred[2] + 4, b_pred[3] + 4),
                          ped_color, self.box_width)
            cv2.rectangle(img, (b_gt[0] - 4, b_gt[1] - 4), (b_gt[2] + 4, b_gt[3] + 4),
                          gt_color, self.box_width)

    def draw_boxes(self, img, observe_data):
        """
        Draws two boxes: the top part is the prediction and the main part around the pedestrian is
        the ground truth
        Args:
            img: image array
            observe_data: observed data used for prediction

        Returns:
            None
        """
        bbox = list(map(int, observe_data['gt_bbox'][-1]))
        text, text_color = self.get_text_and_color(observe_data['pred_activity'])
        # Get gt color for box
        box_color = self.colors['red'] if observe_data['gt_activity'] else self.colors['green']
        if self.add_text:
            cv2.putText(img, text, (bbox[0], bbox[1] - 15), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,
                        text_color, -1)
        else:
            box_height = bbox[3] - bbox[1]
            act_box_height = box_height // 10
            # Draw prediction box
            cv2.rectangle(img, (bbox[0] - 4, bbox[1] - 4 - act_box_height), (bbox[2] + 4, bbox[1] - 4),
                          text_color, -1)
        # Draw the main box
        cv2.rectangle(img, (bbox[0] - 4, bbox[1] - 4), (bbox[2] + 4, bbox[3] + 4),
                      box_color, self.box_width)

    def get_text_and_color(self, pred_result):
        """
        Creates text and color for visualizing actions. The color is adjusted according to the
        probability value.
        Args:
            pred_result: action prediction probability

        Returns:
            text and color of the text
        """
        if pred_result is None:
            text_color = (200, 200, 200)  # (53, 212, 231)
            text = 'UKN'
        else:
            text_color = (50, int(self.intensity_base - self.intensity_base * pred_result),
                          int(self.intensity_base * pred_result))
            text = 'C' if pred_result > 0.5 else 'NC'
        return text, text_color

    def visualize_traj_single(self, img_path, pid, ped_box,
                              pred_traj, gt_traj, configs):
        """
        Visualizes trajectories on a single image
        Args:
            img_path: path to the image
            pid: pedestrian id
            ped_box: current bounding box around pedestrian
            pred_traj: predicted trajectory (boxes)
            gt_traj: future bounding boxes
            configs: configurations

        Returns:
            None
        """
        img = cv2.imread(img_path, flags=cv2.IMREAD_COLOR)
        traj_range = len(gt_traj)
        if self.max_traj_length is not None:
            if self.max_traj_length < len(gt_traj):
                traj_range = self.max_traj_length
        self._draw_traj(img, traj_range, pred_traj, gt_traj, ped_box, configs)
        img = self.scale_image(img)
        frame_num = img_path.split('/')[-1].split('.')[0]
        frame_name = f"{frame_num}-{pid}"
        self.save_result_output(img, frame_name)
