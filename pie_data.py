"""
Interface for the PIE dataset:

A. Rasouli, I. Kotseruba, T. Kunic, and J. Tsotsos, "PIE: A Large-Scale Dataset and Models for Pedestrian Intention Estimation and
Trajectory Prediction", ICCV 2019.

MIT License

Copyright (c) 2019 Amir Rasouli, Iuliia Kotseruba

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

"""
import pickle
import cv2
import sys

import xml.etree.ElementTree as ET
import numpy as np

from os.path import join, abspath, isfile, isdir
from os import makedirs, listdir
from sklearn.model_selection import train_test_split, KFold


class PIE(object):
    def __init__(self, regen_database=False, data_path=''):
        """
        Class constructor
        :param regen_database: Whether generate the database or not
        :param data_path: The path to wh
        """
        self._year = '2019'
        self._name = 'pie'
        self._image_ext = '.png'
        self._regen_database = regen_database

        # Paths
        self._pie_path = data_path if data_path else self._get_default_path()
        assert isdir(self._pie_path), \
            'pie path does not exist: {}'.format(self._pie_path)

        self._annotation_path = join(self._pie_path, 'annotations')
        self._annotation_attributes_path = join(self._pie_path, 'annotations_attributes')
        self._annotation_vehicle_path = join(self._pie_path, 'annotations_vehicle')

        self._clips_path = join(self._pie_path, 'PIE_clips')
        self._images_path = join(self._pie_path, 'images')

    # Path generators
    @property
    def cache_path(self):
        """
        Generates a path to save cache files
        :return: Cache file folder path
        """
        cache_path = abspath(join(self._pie_path, 'data_cache'))
        if not isdir(cache_path):
            makedirs(cache_path)
        return cache_path

    def _get_default_path(self):
        """
        Returns the default path where pie is expected to be installed.
        """
        return 'data/pie'

    def _get_image_set_ids(self, image_set):
        """
        Returns default image set ids
        :param image_set: Image set split
        :return: Set ids of the image set
        """
        image_set_nums = {'train': ['set01', 'set02', 'set04'],
                          'val': ['set05', 'set06'],
                          'test': ['set03'],
                          'all': ['set01', 'set02', 'set03',
                                  'set04', 'set05', 'set06']}
        return image_set_nums[image_set]

    def _get_image_path(self, sid, vid, fid):
        """
        Generates and returns the image path given ids
        :param sid: Set id
        :param vid: Video id
        :param fid: Frame id
        :return: Return the path to the given image
        """
        return join(self._images_path, sid, vid,
                    '{:05d}.png'.format(fid))

    # Visual helpers
    def update_progress(self, progress):
        """
        Creates a progress bar
        :param progress: The progress thus far
        """
        barLength = 20  # Modify this to change the length of the progress bar
        status = ""
        if isinstance(progress, int):
            progress = float(progress)

        block = int(round(barLength * progress))
        text = "\r[{}] {:0.2f}% {}".format("#" * block + "-" * (barLength - block), progress * 100, status)
        sys.stdout.write(text)
        sys.stdout.flush()

    def _print_dict(self, dic):
        """
        Prints a dictionary, one key-value pair per line
        :param dic: Dictionary
        """
        for k, v in dic.items():
            print('%s: %s' % (str(k), str(v)))

    # Data processing helpers
    def _get_width(self):
        """
        Returns image width
        :return: Image width
        """
        return 1920

    def _get_height(self):
        """
        Returns image height
        :return: Image height
        """
        return 1080

    def _get_dim(self):
        """
        Returns the image dimensions
        :return: Image dimensions
        """
        return 1920, 1080

    # Image processing helpers
    def get_annotated_frame_numbers(self, set_id):
        """
        Generates and returns a dictionary of videos and annotated frames for each video in the give set
        :param set_id: Set to generate annotated frames
        :return: A dictionary of form
                {<video_id>: [<number_of_frames>,<annotated_frame_id_0>,... <annotated_frame_id_n>]}
        """

        print("Generating annotated frame numbers for", set_id)
        annotated_frames_file = join(self._pie_path, "annotations", set_id, set_id + '_annotated_frames.csv')
        # If the file exists, load from the file
        if isfile(annotated_frames_file):
            with open(annotated_frames_file, 'rt') as f:
                annotated_frames = {x.split(',')[0]:
                                        [int(fr) for fr in x.split(',')[1:]] for x in f.readlines()}
            return annotated_frames
        else:
            # Generate annotated frame ids for each video
            annotated_frames = {v.split('_annt.xml')[0]: [] for v in sorted(listdir(join(self._annotation_path,
                                                                                         set_id))) if
                                v.endswith("annt.xml")}
            for vid, annot_frames in sorted(annotated_frames.items()):
                _frames = []
                path_to_file = join(self._annotation_path, set_id, vid + '_annt.xml')
                tree = ET.parse(path_to_file)
                tracks = tree.findall('./track')
                for t in tracks:
                    if t.get('label') != 'pedestrian':
                        continue
                    boxes = t.findall('./box')
                    for b in boxes:
                        # Exclude the annotations that are outside of the frame
                        if int(b.get('outside')) == 1:
                            continue
                        _frames.append(int(b.get('frame')))
                _frames = sorted(list(set(_frames)))
                annot_frames.append(len(_frames))
                annot_frames.extend(_frames)

            with open(annotated_frames_file, 'wt') as fid:
                for vid, annot_frames in sorted(annotated_frames.items()):
                    fid.write(vid)
                    for fr in annot_frames:
                        fid.write("," + str(fr))
                    fid.write('\n')

        return annotated_frames

    def get_frame_numbers(self, set_id):
        """
        Generates and returns a dictionary of videos and  frames for each video in the give set
        :param set_id: Set to generate annotated frames
        :return: A dictionary of form
                {<video_id>: [<number_of_frames>,<frame_id_0>,... <frame_id_n>]}
        """
        print("Generating frame numbers for", set_id)
        frame_ids = {v.split('_annt.xml')[0]: [] for v in sorted(listdir(join(self._annotation_path,
                                                                              set_id))) if
                     v.endswith("annt.xml")}
        for vid, frames in sorted(frame_ids.items()):
            path_to_file = join(self._annotation_path, set_id, vid + '_annt.xml')
            tree = ET.parse(path_to_file)
            num_frames = int(tree.find("./meta/task/size").text)
            frames.extend([i for i in range(num_frames)])
            frames.insert(0, num_frames)
        return frame_ids

    def extract_and_save_images(self, extract_frame_type='annotated'):
        """
        Extracts images from clips and saves on hard drive
        :param extract_frame_type: Whether to extract 'all' frames or only the ones that are 'annotated'
                             Note: extracting 'all' frames requires approx. 3TB space whereas
                                   'annotated' requires approx. 1TB
        """
        set_folders = [f for f in sorted(listdir(self._clips_path))]
        for set_id in set_folders:
            print('Extracting frames from', set_id)
            set_folder_path = join(self._clips_path, set_id)
            if extract_frame_type == 'annotated':
                extract_frames = self.get_annotated_frame_numbers(set_id)
            else:
                extract_frames = self.get_frame_numbers(set_id)

            set_images_path = join(self._pie_path, "images", set_id)
            for vid, frames in sorted(extract_frames.items()):
                print(vid)
                video_images_path = join(set_images_path, vid)
                num_frames = frames[0]
                frames_list = frames[1:]
                if not isdir(video_images_path):
                    makedirs(video_images_path)
                vidcap = cv2.VideoCapture(join(set_folder_path, vid + '.mp4'))
                success, image = vidcap.read()
                frame_num = 0
                img_count = 0
                if not success:
                    print('Failed to open the video {}'.format(vid))
                while success:
                    if frame_num in frames_list:
                        self.update_progress(img_count / num_frames)
                        img_count += 1
                        if not isfile(join(video_images_path, "%05.f.png") % frame_num):
                            cv2.imwrite(join(video_images_path, "%05.f.png") % frame_num, image)
                    success, image = vidcap.read()
                    frame_num += 1
                if num_frames != img_count:
                    print('num images don\'t match {}/{}'.format(num_frames, img_count))
                print('\n')

    # Annotation processing helpers
    def _map_text_to_scalar(self, label_type, value):
        """
        Maps a text label in XML file to scalars
        :param label_type: The label type
        :param value: The text to be mapped
        :return: The scalar value
        """
        map_dic = {'occlusion': {'none': 0, 'part': 1, 'full': 2},
                   'action': {'standing': 0, 'walking': 1},
                   'look': {'not-looking': 0, 'looking': 1},
                   'gesture': {'__undefined__': 0, 'hand_ack': 1, 'hand_yield': 2,
                               'hand_rightofway': 3, 'nod': 4, 'other': 5},
                   'cross': {'not-crossing': 0, 'crossing': 1, 'crossing-irrelevant': -1},
                   'crossing': {'not-crossing': 0, 'crossing': 1, 'irrelevant': -1},
                   'age': {'child': 0, 'young': 1, 'adult': 2, 'senior': 3},
                   'designated': {'ND': 0, 'D': 1},
                   'gender': {'n/a': 0, 'female': 1, 'male': 2},
                   'intersection': {'midblock': 0, 'T': 1, 'T-left': 2, 'T-right': 3, 'four-way': 4},
                   'motion_direction': {'n/a': 0, 'LAT': 1, 'LONG': 2},
                   'traffic_direction': {'OW': 0, 'TW': 1},
                   'signalized': {'n/a': 0, 'C': 1, 'S': 2, 'CS': 3},
                   'vehicle': {'car': 0, 'truck': 1, 'bus': 2, 'train': 3, 'bicycle': 4, 'bike': 5},
                   'sign': {'ped_blue': 0, 'ped_yellow': 1, 'ped_white': 2, 'ped_text': 3, 'stop_sign': 4,
                            'bus_stop': 5, 'train_stop': 6, 'construction': 7, 'other': 8},
                   'traffic_light': {'regular': 0, 'transit': 1, 'pedestrian': 2},
                   'state': {'__undefined__': 0, 'red': 1, 'yellow': 2, 'green': 3}}

        return map_dic[label_type][value]

    def _map_scalar_to_text(self, label_type, value):
        """
        Maps a scalar value to a text label
        :param label_type: The label type
        :param value: The scalar to be mapped
        :return: The text label
        """
        map_dic = {'occlusion': {0: 'none', 1: 'part', 2: 'full'},
                   'action': {0: 'standing', 1: 'walking'},
                   'look': {0: 'not-looking', 1: 'looking'},
                   'hand_gesture': {0: '__undefined__', 1: 'hand_ack',
                                    2: 'hand_yield', 3: 'hand_rightofway',
                                    4: 'nod', 5: 'other'},
                   'cross': {0: 'not-crossing', 1: 'crossing', -1: 'crossing-irrelevant'},
                   'crossing': {0: 'not-crossing', 1: 'crossing', -1: 'irrelevant'},
                   'age': {0: 'child', 1: 'young', 2: 'adult', 3: 'senior'},
                   'designated': {0: 'ND', 1: 'D'},
                   'gender': {0: 'n/a', 1: 'female', 2: 'male'},
                   'intersection': {0: 'midblock', 1: 'T', 2: 'T-left', 3: 'T-right', 4: 'four-way'},
                   'motion_direction': {0: 'n/a', 1: 'LAT', 2: 'LONG'},
                   'traffic_direction': {0: 'OW', 1: 'TW'},
                   'signalized': {0: 'n/a', 1: 'C', 2: 'S', 3: 'CS'},
                   'vehicle': {0: 'car', 1: 'truck', 2: 'bus', 3: 'train', 4: 'bicycle', 5: 'bike'},
                   'sign': {0: 'ped_blue', 1: 'ped_yellow', 2: 'ped_white', 3: 'ped_text', 4: 'stop_sign',
                            5: 'bus_stop', 6: 'train_stop', 7: 'construction', 8: 'other'},
                   'traffic_light': {0: 'regular', 1: 'transit', 2: 'pedestrian'},
                   'state': {0: '__undefined__', 1: 'red', 2: 'yellow', 3: 'green'}}

        return map_dic[label_type][value]

    def _get_annotations(self, setid, vid):
        """
        Generates a dictionary of annotations by parsing the video XML file
        :param setid: The set id
        :param vid: The video id
        :return: A dictionary of annotations
        """
        path_to_file = join(self._annotation_path, setid, vid + '_annt.xml')
        print(path_to_file)

        tree = ET.parse(path_to_file)
        ped_annt = 'ped_annotations'
        traffic_annt = 'traffic_annotations'

        annotations = {}
        annotations['num_frames'] = int(tree.find("./meta/task/size").text)
        annotations['width'] = int(tree.find("./meta/task/original_size/width").text)
        annotations['height'] = int(tree.find("./meta/task/original_size/height").text)
        annotations[ped_annt] = {}
        annotations[traffic_annt] = {}
        tracks = tree.findall('./track')
        for t in tracks:
            boxes = t.findall('./box')
            obj_label = t.get('label')
            obj_id = boxes[0].find('./attribute[@name=\"id\"]').text

            if obj_label == 'pedestrian':
                annotations[ped_annt][obj_id] = {'frames': [], 'bbox': [], 'occlusion': []}
                annotations[ped_annt][obj_id]['behavior'] = {'gesture': [], 'look': [], 'action': [], 'cross': []}
                for b in boxes:
                    # Exclude the annotations that are outside of the frame
                    if int(b.get('outside')) == 1:
                        continue
                    annotations[ped_annt][obj_id]['bbox'].append(
                        [float(b.get('xtl')), float(b.get('ytl')),
                         float(b.get('xbr')), float(b.get('ybr'))])
                    occ = self._map_text_to_scalar('occlusion', b.find('./attribute[@name=\"occlusion\"]').text)
                    annotations[ped_annt][obj_id]['occlusion'].append(occ)
                    annotations[ped_annt][obj_id]['frames'].append(int(b.get('frame')))
                    for beh in annotations['ped_annotations'][obj_id]['behavior']:
                        # Read behavior tags for each frame and add to the database
                        annotations[ped_annt][obj_id]['behavior'][beh].append(
                            self._map_text_to_scalar(beh, b.find('./attribute[@name=\"' + beh + '\"]').text))

            else:
                obj_type = boxes[0].find('./attribute[@name=\"type\"]')
                if obj_type is not None:
                    obj_type = self._map_text_to_scalar(obj_label,
                                                        boxes[0].find('./attribute[@name=\"type\"]').text)

                annotations[traffic_annt][obj_id] = {'frames': [], 'bbox': [], 'occlusion': [],
                                                     'obj_class': obj_label,
                                                     'obj_type': obj_type,
                                                     'state': []}

                for b in boxes:
                    # Exclude the annotations that are outside of the frame
                    if int(b.get('outside')) == 1:
                        continue
                    annotations[traffic_annt][obj_id]['bbox'].append(
                        [float(b.get('xtl')), float(b.get('ytl')),
                         float(b.get('xbr')), float(b.get('ybr'))])
                    annotations[traffic_annt][obj_id]['occlusion'].append(int(b.get('occluded')))
                    annotations[traffic_annt][obj_id]['frames'].append(int(b.get('frame')))
                    if obj_label == 'traffic_light':
                        annotations[traffic_annt][obj_id]['state'].append(self._map_text_to_scalar('state',
                                                                                                    b.find(
                                                                                                        './attribute[@name=\"state\"]').text))
        return annotations

    def _get_ped_attributes(self, setid, vid):
        """
        Generates a dictionary of attributes by parsing the video XML file
        :param setid: The set id
        :param vid: The video id
        :return: A dictionary of attributes
        """
        path_to_file = join(self._annotation_attributes_path, setid, vid + '_attributes.xml')
        tree = ET.parse(path_to_file)

        attributes = {}
        pedestrians = tree.findall("./pedestrian")
        for p in pedestrians:
            ped_id = p.get('id')
            attributes[ped_id] = {}
            for k, v in p.items():
                if 'id' in k:
                    continue
                try:
                    if k == 'intention_prob':
                        attributes[ped_id][k] = float(v)
                    else:
                        attributes[ped_id][k] = int(v)
                except ValueError:
                    attributes[ped_id][k] = self._map_text_to_scalar(k, v)

        return attributes

    def _get_vehicle_attributes(self, setid, vid):
        """
        Generates a dictionary of vehicle attributes by parsing the video XML file
        :param setid: The set id
        :param vid: The video id
        :return: A dictionary of vehicle attributes (obd sensor recording)
        """
        path_to_file = join(self._annotation_vehicle_path, setid, vid + '_obd.xml')
        tree = ET.parse(path_to_file)

        veh_attributes = {}
        frames = tree.findall("./frame")

        for f in frames:
            dict_vals = {k: float(v) for k, v in f.attrib.items() if k != 'id'}
            veh_attributes[int(f.get('id'))] = dict_vals

        return veh_attributes

    def generate_database(self):
        """
        Generates and saves a database of the pie dataset by integrating all annotations
        Dictionary structure:
        'set_id'(str): {
            'vid_id'(str): {
                'num_frames': int
                'width': int
                'height': int
                'traffic_annotations'(str): {
                    'obj_id'(str): {
                        'frames': list(int)
                        'occlusion': list(int)
                        'bbox': list([x1, y1, x2, y2]) (float)
                        'obj_class': str,
                        'obj_type': str,    # only for traffic lights, vehicles, signs
                        'state': list(int)  # only for traffic lights
                'ped_annotations'(str): {
                    'ped_id'(str): {
                        'frames': list(int)
                        'occlusion': list(int)
                        'bbox': list([x1, y1, x2, y2]) (float)
                        'behavior'(str): {
                            'action': list(int)
                            'gesture': list(int)
                            'cross': list(int)
                            'look': list(int)
                        'attributes'(str): {
                             'age': int
                             'id': str
                             'num_lanes': int
                             'crossing': int
                             'gender': int
                             'crossing_point': int
                             'critical_point': int
                             'exp_start_point': int
                             'intersection': int
                             'designated': int
                             'signalized': int
                             'traffic_direction': int
                             'group_size': int
                             'motion_direction': int
                'vehicle_annotations'(str){
                    'frame_id'(int){'longitude': float
                          'yaw': float
                          'pitch': float
                          'roll': float
                          'OBD_speed': float
                          'GPS_speed': float
                          'latitude': float
                          'longitude': float
                          'heading_angle': float
                          'accX': float
                          'accY': float
                          'accZ: float
                          'gyroX': float
                          'gyroY': float
                          'gyroZ': float

        :return: A database dictionary
        """

        print('---------------------------------------------------------')
        print("Generating database for pie")

        cache_file = join(self.cache_path, 'pie_database.pkl')
        if isfile(cache_file) and not self._regen_database:
            with open(cache_file, 'rb') as fid:
                try:
                    database = pickle.load(fid)
                except:
                    database = pickle.load(fid, encoding='bytes')
            print('pie annotations loaded from {}'.format(cache_file))
            return database

        # Path to the folder annotations
        set_ids = [f for f in sorted(listdir(self._annotation_path))]

        # Read the content of set folders
        database = {}
        for setid in set_ids:
            video_ids = [v.split('_annt.xml')[0] for v in sorted(listdir(join(self._annotation_path,
                                                                              setid))) if v.endswith("annt.xml")]
            database[setid] = {}
            for vid in video_ids:
                print('Getting annotations for %s, %s' % (setid, vid))
                database[setid][vid] = self._get_annotations(setid, vid)
                vid_attributes = self._get_ped_attributes(setid, vid)
                database[setid][vid]['vehicle_annotations'] = self._get_vehicle_attributes(setid, vid)
                for ped in database[setid][vid]['ped_annotations']:
                    database[setid][vid]['ped_annotations'][ped]['attributes'] = vid_attributes[ped]

        with open(cache_file, 'wb') as fid:
            pickle.dump(database, fid, pickle.HIGHEST_PROTOCOL)
        print('The database is written to {}'.format(cache_file))

        return database

    def get_data_stats(self):
        """
        Generates statistics for the dataset
        """
        annotations = self.generate_database()

        set_count = len(annotations.keys())

        ped_count = 0
        ped_box_count = 0
        video_count = 0
        total_frames = 0
        age = {'child': 0, 'adult': 0, 'senior': 0}
        gender = {'male': 0, 'female': 0}
        signalized = {'n/a': 0, 'C': 0, 'S': 0, 'CS': 0}
        traffic_direction = {'OW': 0, 'TW': 0}
        intersection = {'midblock': 0, 'T': 0, 'T-right': 0, 'T-left': 0, 'four-way': 0}
        crossing = {'crossing': 0, 'not-crossing': 0, 'irrelevant': 0}

        traffic_obj_types = {'vehicle': {'car': 0, 'truck': 0, 'bus': 0, 'train': 0, 'bicycle': 0, 'bike': 0},
                             'sign': {'ped_blue': 0, 'ped_yellow': 0, 'ped_white': 0, 'ped_text': 0, 'stop_sign': 0,
                                      'bus_stop': 0, 'train_stop': 0, 'construction': 0, 'other': 0},
                             'traffic_light': {'regular': 0, 'transit': 0, 'pedestrian': 0},
                             'crosswalk': 0,
                             'transit_station': 0}
        traffic_box_count = {'vehicle': 0, 'traffic_light': 0, 'sign': 0, 'crosswalk': 0, 'transit_station': 0}
        for sid, vids in annotations.items():
            video_count += len(vids)
            for vid, annots in vids.items():
                total_frames += annots['num_frames']
                for trf_ids, trf_annots in annots['traffic_annotations'].items():
                    obj_class = trf_annots['obj_class']
                    traffic_box_count[obj_class] += len(trf_annots['frames'])
                    if obj_class in ['traffic_light', 'vehicle', 'sign']:
                        obj_type = trf_annots['obj_type']
                        traffic_obj_types[obj_class][self._map_scalar_to_text(obj_class, obj_type)] += 1
                    else:
                        traffic_obj_types[obj_class] += 1
                for ped_ids, ped_annots in annots['ped_annotations'].items():
                    ped_count += 1
                    ped_box_count += len(ped_annots['frames'])
                    age[self._map_scalar_to_text('age', ped_annots['attributes']['age'])] += 1
                    if self._map_scalar_to_text('crossing', ped_annots['attributes']['crossing']) == 'crossing':
                        crossing[self._map_scalar_to_text('crossing', ped_annots['attributes']['crossing'])] += 1
                    else:
                        if ped_annots['attributes']['intention_prob'] > 0.5:
                            crossing['not-crossing'] += 1
                        else:
                            crossing['irrelevant'] += 1                    
                    intersection[
                        self._map_scalar_to_text('intersection', ped_annots['attributes']['intersection'])] += 1
                    traffic_direction[self._map_scalar_to_text('traffic_direction',
                                                               ped_annots['attributes']['traffic_direction'])] += 1
                    signalized[self._map_scalar_to_text('signalized', ped_annots['attributes']['signalized'])] += 1
                    gender[self._map_scalar_to_text('gender', ped_annots['attributes']['gender'])] += 1

        print('---------------------------------------------------------')
        print("Number of sets: %d" % set_count)
        print("Number of videos: %d" % video_count)
        print("Number of annotated frames: %d" % total_frames)
        print("Number of pedestrians %d" % ped_count)
        print("age:\n", '\n '.join('{}: {}'.format(tag, cnt) for tag, cnt in sorted(age.items())))
        print("gender:\n", '\n '.join('{}: {}'.format(tag, cnt) for tag, cnt in sorted(gender.items())))
        print("signal:\n", '\n '.join('{}: {}'.format(tag, cnt) for tag, cnt in sorted(signalized.items())))
        print("traffic direction:\n",
              '\n '.join('{}: {}'.format(tag, cnt) for tag, cnt in sorted(traffic_direction.items())))
        print("crossing:\n", '\n '.join('{}: {}'.format(tag, cnt) for tag, cnt in sorted(crossing.items())))
        print("intersection:\n", '\n '.join('{}: {}'.format(tag, cnt) for tag, cnt in sorted(intersection.items())))
        print("Number of pedestrian bounding boxes: %d" % ped_box_count)
        print("Number of traffic objects")
        for trf_obj, values in sorted(traffic_obj_types.items()):
            if isinstance(values, dict):
                print(trf_obj + ':\n', '\n '.join('{}: {}'.format(k, v) for k, v in sorted(values.items())),
                      '\n total: ', sum(values.values()))
            else:
                print(trf_obj + ': %d' % values)
        print("Number of pedestrian bounding boxes:\n",
              '\n '.join('{}: {}'.format(tag, cnt) for tag, cnt in sorted(traffic_box_count.items())),
              '\n total: ', sum(traffic_box_count.values()))

    def balance_samples_count(self, seq_data, label_type, random_seed=42):
        """
        Balances the number of positive and negative samples by randomly sampling
        from the more represented samples. Only works for binary classes.
        :param seq_data: The sequence data to be balanced.
        :param label_type: The lable type based on which the balancing takes place.
        The label values must be binary, i.e. only 0, 1.
        :param random_seed: The seed for random number generator.
        :return: Balanced data sequence.
        """
        for lbl in seq_data[label_type]:
            for i in lbl:
                if i[0] not in [0, 1]:
                    raise Exception("The label values used for balancing must be"
                                    " either 0 or 1")

        # balances the number of positive and negative samples
        print('---------------------------------------------------------')
        print("Balancing the number of positive and negative intention samples")

        gt_labels = [gt[0] for gt in seq_data[label_type]]
        num_pos_samples = np.count_nonzero(np.array(gt_labels))
        num_neg_samples = len(gt_labels) - num_pos_samples

        new_seq_data = {}
        # finds the indices of the samples with larger quantity
        if num_neg_samples == num_pos_samples:
            print('Positive and negative samples are already balanced')
            return seq_data
        else:
            print('Unbalanced: \t Positive: {} \t Negative: {}'.format(num_pos_samples, num_neg_samples))
            if num_neg_samples > num_pos_samples:
                rm_index = np.where(np.array(gt_labels) == 0)[0]
            else:
                rm_index = np.where(np.array(gt_labels) == 1)[0]

            # Calculate the difference of sample counts
            dif_samples = abs(num_neg_samples - num_pos_samples)
            # shuffle the indices
            np.random.seed(random_seed)
            np.random.shuffle(rm_index)
            # reduce the number of indices to the difference
            rm_index = rm_index[0:dif_samples]
            # update the data
            for k in seq_data:
                seq_data_k = seq_data[k]
                if not isinstance(seq_data[k], list):
                    new_seq_data[k] = seq_data[k]
                else:
                    new_seq_data[k] = [seq_data_k[i] for i in range(0, len(seq_data_k)) if i not in rm_index]

            new_gt_labels = [gt[0] for gt in new_seq_data[label_type]]
            num_pos_samples = np.count_nonzero(np.array(new_gt_labels))
            print('Balanced:\t Positive: %d  \t Negative: %d\n'
                  % (num_pos_samples, len(new_seq_data[label_type]) - num_pos_samples))
        return new_seq_data

    # Process pedestrian ids
    def _get_pedestrian_ids(self):
        """
        Returns all pedestrian ids
        :return: A list of pedestrian ids
        """
        annotations = self.generate_database()
        pids = []
        for sid in sorted(annotations):
            for vid in sorted(annotations[sid]):
                pids.extend(annotations[sid][vid]['ped_annotations'].keys())
        return pids

    def _get_random_pedestrian_ids(self, image_set, ratios=None, val_data=True, regen_data=False):
        """
        Generates and saves a random pedestrian ids
        :param image_set: The data split to return
        :param ratios: The ratios to split the data. There should be 2 ratios (or 3 if val_data is true)
        and they should sum to 1. e.g. [0.4, 0.6], [0.3, 0.5, 0.2]
        :param val_data: Whether to generate validation data
        :param regen_data: Whether to overwrite the existing data, i.e. regenerate splits
        :return: The random sample split
        """

        assert image_set in ['train', 'test', 'val']
        # Generates a list of behavioral xml file names for  videos
        cache_file = join(self.cache_path, "random_samples.pkl")
        if isfile(cache_file) and not regen_data:
            print("Random sample currently exists.\n Loading from %s" % cache_file)
            with open(cache_file, 'rb') as fid:
                try:
                    rand_samples = pickle.load(fid)
                except:
                    rand_samples = pickle.load(fid, encoding='bytes')
                assert image_set in rand_samples, "%s does not exist in random samples\n" \
                                                  "Please try again by setting regen_data = True" % image_set
                if val_data:
                    assert len(rand_samples['ratios']) == 3, "The existing random samples " \
                                                             "does not have validation data.\n" \
                                                             "Please try again by setting regen_data = True"
                if ratios is not None:
                    assert ratios == rand_samples['ratios'], "Specified ratios {} does not match the ones in existing file {}.\n\
                                                              Perform one of the following options:\
                                                              1- Set ratios to None\
                                                              2- Set ratios to the same values \
                                                              3- Regenerate data".format(ratios, rand_samples['ratios'])

                print('The ratios are {}'.format(rand_samples['ratios']))
                print("Number of %s tracks %d" % (image_set, len(rand_samples[image_set])))
                return rand_samples[image_set]

        if ratios is None:
            if val_data:
                ratios = [0.5, 0.4, 0.1]
            else:
                ratios = [0.5, 0.5]

        assert sum(ratios) > 0.999999, "Ratios {} do not sum to 1".format(ratios)
        if val_data:
            assert len(ratios) == 3, "To generate validation data three ratios should be selected"
        else:
            assert len(ratios) == 2, "With no validation only two ratios should be selected"

        print("################ Generating Random training/testing data ################")
        ped_ids = self._get_pedestrian_ids()
        print("Toral number of tracks %d" % len(ped_ids))
        print('The ratios are {}'.format(ratios))
        sample_split = {'ratios': ratios}
        train_samples, test_samples = train_test_split(ped_ids, train_size=ratios[0])
        print("Number of train tracks %d" % len(train_samples))

        if val_data:
            test_samples, val_samples = train_test_split(test_samples, train_size=ratios[1] / sum(ratios[1:]))
            print("Number of val tracks %d" % len(val_samples))
            sample_split['val'] = val_samples

        print("Number of test tracks %d" % len(test_samples))
        sample_split['train'] = train_samples
        sample_split['test'] = test_samples

        cache_file = join(self.cache_path, "random_samples.pkl")
        with open(cache_file, 'wb') as fid:
            pickle.dump(sample_split, fid, pickle.HIGHEST_PROTOCOL)
            print('pie {} samples written to {}'.format('random', cache_file))
        return sample_split[image_set]

    def _get_kfold_pedestrian_ids(self, image_set, num_folds=5, fold=1):
        """
        Generates kfold pedestrian ids
        :param image_set: Image set split
        :param num_folds: Number of folds
        :param fold: The given fold
        :return: List of pedestrian ids for the given fold
        """
        assert image_set in ['train', 'test'], "Image set should be either \"train\" or \"test\""
        assert fold <= num_folds, "Fold number should be smaller than number of folds"
        print("################ Generating %d fold data ################" % num_folds)
        cache_file = join(self.cache_path, "%d_fold_samples.pkl" % num_folds)

        if isfile(cache_file):
            print("Loading %d-fold data from %s" % (num_folds, cache_file))
            with open(cache_file, 'rb') as fid:
                try:
                    fold_idx = pickle.load(fid)
                except:
                    fold_idx = pickle.load(fid, encoding='bytes')
        else:
            ped_ids = self._get_pedestrian_ids()
            kf = KFold(n_splits=num_folds, shuffle=True)
            fold_idx = {'pid': ped_ids}
            count = 1
            for train_index, test_index in kf.split(ped_ids):
                fold_idx[count] = {'train': train_index.tolist(), 'test': test_index.tolist()}
                count += 1
            with open(cache_file, 'wb') as fid:
                pickle.dump(fold_idx, fid, pickle.HIGHEST_PROTOCOL)
                print('pie {}-fold samples written to {}'.format(num_folds, cache_file))
        print("Number of %s tracks %d" % (image_set, len(fold_idx[fold][image_set])))
        kfold_ids = [fold_idx['pid'][i] for i in range(len(fold_idx['pid'])) if i in fold_idx[fold][image_set]]
        return kfold_ids

    # Trajectory data generation
    def _get_data_ids(self, image_set, params):
        """
        Generates set ids and ped ids (if needed) for processing
        :param image_set: Image-set to generate data
        :param params: Data generation params
        :return: Set and pedestrian ids
        """
        _pids = None
        if params['data_split_type'] == 'default':
            set_ids = self._get_image_set_ids(image_set)
        else:
            set_ids = self._get_image_set_ids('all')
        if params['data_split_type'] == 'random':
            _pids = self._get_random_pedestrian_ids(image_set, **params['random_params'])
        elif params['data_split_type'] == 'kfold':
            _pids = self._get_kfold_pedestrian_ids(image_set, **params['kfold_params'])

        return set_ids, _pids

    def _squarify(self, bbox, ratio, img_width):
        """
        Changes the ratio of bounding boxes to a fixed ratio
        :param bbox: Bounding box
        :param ratio: Ratio to be changed to
        :param img_width: Image width
        :return: Squarified boduning box
        """
        width = abs(bbox[0] - bbox[2])
        height = abs(bbox[1] - bbox[3])
        width_change = height * ratio - width

        bbox[0] = bbox[0] - width_change / 2
        bbox[2] = bbox[2] + width_change / 2

        if bbox[0] < 0:
            bbox[0] = 0

        # check whether the new bounding box goes beyond image boarders
        # If this is the case, the bounding box is shifted back
        if bbox[2] > img_width:
            bbox[0] = bbox[0] - bbox[2] + img_width
            bbox[2] = img_width
        return bbox

    def _height_check(self, height_rng, frame_ids, boxes, images, occlusion):
        """
        Checks whether the bounding boxes are within a given height limit. If not, it
        will adjust the length of bounding boxes in data sequences accordingly
        :param height_rng: Height limit [lower, higher]
        :param frame_ids: List of frame ids
        :param boxes: List of bounding boxes
        :param images: List of images
        :param occlusion: List of occlusions
        :return: The adjusted data sequences
        """
        imgs, box, frames, occ = [], [], [], []
        for i, b in enumerate(boxes):
            bbox_height = abs(b[1] - b[3])
            if height_rng[0] <= bbox_height <= height_rng[1]:
                box.append(b)
                imgs.append(images[i])
                frames.append(frame_ids[i])
                occ.append(occlusion[i])
        return imgs, box, frames, occ

    def _get_center(self, box):
        """
        Calculates the center coordinate of a bounding box
        :param box: Bounding box coordinates
        :return: The center coordinate
        """
        return [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]

    def generate_data_trajectory_sequence(self, image_set, **opts):
        """
        Generates pedestrian tracks
        :param image_set: the split set to produce for. Options are train, test, val.
        :param opts:
                'fstride': Frequency of sampling from the data.
                'height_rng': The height range of pedestrians to use.
                'squarify_ratio': The width/height ratio of bounding boxes. A value between (0,1]. 0 the original
                                        ratio is used.
                'data_split_type': How to split the data. Options: 'default', predefined sets, 'random', randomly split the data,
                                        and 'kfold', k-fold data split (NOTE: only train/test splits).
                'seq_type': Sequence type to generate. Options: 'trajectory', generates tracks, 'crossing', generates
                                  tracks up to 'crossing_point', 'intention' generates tracks similar to human experiments
                'min_track_size': Min track length allowable.
                'random_params: Parameters for random data split generation. (see _get_random_pedestrian_ids)
                'kfold_params: Parameters for kfold split generation. (see _get_kfold_pedestrian_ids)
        :return: Sequence data
        """
        params = {'fstride': 1,
                  'sample_type': 'all',  # 'beh'
                  'height_rng': [0, float('inf')],
                  'squarify_ratio': 0,
                  'data_split_type': 'default',  # kfold, random, default
                  'seq_type': 'intention',
                  'min_track_size': 15,
                  'random_params': {'ratios': None,
                                    'val_data': True,
                                    'regen_data': False},
                  'kfold_params': {'num_folds': 5, 'fold': 1}}

        for i in opts.keys():
            params[i] = opts[i]

        print('---------------------------------------------------------')
        print("Generating trajectory sequence data")
        self._print_dict(params)
        annot_database = self.generate_database()
        if params['seq_type'] == 'trajectory':
            sequence_data = self._get_trajectories(image_set, annot_database, **params)
        elif params['seq_type'] == 'crossing':
            sequence_data = self._get_crossing(image_set, annot_database, **params)
        elif params['seq_type'] == 'intention':
            sequence_data = self._get_intention(image_set, annot_database, **params)

        return sequence_data

    def _get_trajectories(self, image_set, annotations, **params):
        """
        Generates trajectory data.
        :param image_set: Data split to use
        :param annotations: Annotations database
        :param params: Parameters to generate data (see generade_database)
        :return: A dictionary of trajectories
        """
        print('---------------------------------------------------------')
        print("Generating trajectory data")

        num_pedestrians = 0
        seq_stride = params['fstride']
        sq_ratio = params['squarify_ratio']
        height_rng = params['height_rng']

        image_seq, pids_seq = [], []
        box_seq, center_seq, occ_seq = [], [], []
        intent_seq = []
        obds_seq, gpss_seq, head_ang_seq, gpsc_seq, yrp_seq = [], [], [], [], []

        set_ids, _pids = self._get_data_ids(image_set, params)

        for sid in set_ids:
            for vid in sorted(annotations[sid]):
                img_width = annotations[sid][vid]['width']
                pid_annots = annotations[sid][vid]['ped_annotations']
                vid_annots = annotations[sid][vid]['vehicle_annotations']
                for pid in sorted(pid_annots):
                    if params['data_split_type'] != 'default' and pid not in _pids:
                        continue
                    num_pedestrians += 1
                    frame_ids = pid_annots[pid]['frames']
                    boxes = pid_annots[pid]['bbox']
                    images = [self._get_image_path(sid, vid, f) for f in frame_ids]
                    occlusions = pid_annots[pid]['occlusion']

                    if height_rng[0] > 0 or height_rng[1] < float('inf'):
                        images, boxes, frame_ids, occlusions = self._height_check(height_rng,
                                                                                  frame_ids, boxes,
                                                                                  images, occlusions)

                    if len(boxes) / seq_stride < params['min_track_size']:  # max_obs_size: #90 + 45
                        continue

                    if sq_ratio:
                        boxes = [self._squarify(b, sq_ratio, img_width) for b in boxes]

                    image_seq.append(images[::seq_stride])
                    box_seq.append(boxes[::seq_stride])
                    center_seq.append([self._get_center(b) for b in boxes][::seq_stride])
                    occ_seq.append(occlusions[::seq_stride])

                    ped_ids = [[pid]] * len(boxes)
                    pids_seq.append(ped_ids[::seq_stride])

                    intent = [[pid_annots[pid]['attributes']['intention_prob']]] * len(boxes)
                    intent_seq.append(intent[::seq_stride])

                    gpsc_seq.append([(vid_annots[i]['latitude'], vid_annots[i]['longitude'])
                                     for i in frame_ids][::seq_stride])
                    obds_seq.append([[vid_annots[i]['OBD_speed']] for i in frame_ids][::seq_stride])
                    gpss_seq.append([[vid_annots[i]['GPS_speed']] for i in frame_ids][::seq_stride])
                    head_ang_seq.append([[vid_annots[i]['heading_angle']] for i in frame_ids][::seq_stride])
                    yrp_seq.append([(vid_annots[i]['yaw'], vid_annots[i]['roll'], vid_annots[i]['pitch'])
                                    for i in frame_ids][::seq_stride])

        print('Subset: %s' % image_set)
        print('Number of pedestrians: %d ' % num_pedestrians)
        print('Total number of samples: %d ' % len(image_seq))

        return {'image': image_seq,
                'pid': pids_seq,
                'bbox': box_seq,
                'center': center_seq,
                'occlusion': occ_seq,
                'obd_speed': obds_seq,
                'gps_speed': gpss_seq,
                'heading_angle': head_ang_seq,
                'gps_coord': gpsc_seq,
                'yrp': yrp_seq,
                'intention_prob': intent_seq}

    def _get_crossing(self, image_set, annotations, **params):
        """
        Generates crossing data.
        :param image_set: Data split to use
        :param annotations: Annotations database
        :param params: Parameters to generate data (see generade_database)
        :return: A dictionary of trajectories
        """

        print('---------------------------------------------------------')
        print("Generating crossing data")

        num_pedestrians = 0
        seq_stride = params['fstride']
        sq_ratio = params['squarify_ratio']
        height_rng = params['height_rng']

        image_seq, pids_seq = [], []
        box_seq, center_seq, occ_seq = [], [], []
        intent_seq = []
        obds_seq, gpss_seq, head_ang_seq, gpsc_seq, yrp_seq = [], [], [], [], []
        activities = []

        set_ids, _pids = self._get_data_ids(image_set, params)

        for sid in set_ids:
            for vid in sorted(annotations[sid]):
                img_width = annotations[sid][vid]['width']
                pid_annots = annotations[sid][vid]['ped_annotations']
                vid_annots = annotations[sid][vid]['vehicle_annotations']
                for pid in sorted(pid_annots):
                    if params['data_split_type'] != 'default' and pid not in _pids:
                        continue
                    num_pedestrians += 1

                    frame_ids = pid_annots[pid]['frames']
                    event_frame = pid_annots[pid]['attributes']['crossing_point']

                    end_idx = frame_ids.index(event_frame)
                    boxes = pid_annots[pid]['bbox'][:end_idx + 1]
                    frame_ids = frame_ids[: end_idx + 1]
                    images = [self._get_image_path(sid, vid, f) for f in frame_ids]
                    occlusions = pid_annots[pid]['occlusion'][:end_idx + 1]

                    if height_rng[0] > 0 or height_rng[1] < float('inf'):
                        images, boxes, frame_ids, occlusions = self._height_check(height_rng,
                                                                                  frame_ids, boxes,
                                                                                  images, occlusions)

                    if len(boxes) / seq_stride < params['min_track_size']:
                        continue

                    if sq_ratio:
                        boxes = [self._squarify(b, sq_ratio, img_width) for b in boxes]

                    image_seq.append(images[::seq_stride])
                    box_seq.append(boxes[::seq_stride])
                    center_seq.append([self._get_center(b) for b in boxes][::seq_stride])
                    occ_seq.append(occlusions[::seq_stride])

                    ped_ids = [[pid]] * len(boxes)
                    pids_seq.append(ped_ids[::seq_stride])

                    intent = [[pid_annots[pid]['attributes']['intention_prob']]] * len(boxes)
                    intent_seq.append(intent[::seq_stride])

                    acts = [[int(pid_annots[pid]['attributes']['crossing'] > 0)]] * len(boxes)
                    activities.append(acts[::seq_stride])

                    gpsc_seq.append([[(vid_annots[i]['latitude'], vid_annots[i]['longitude'])]
                                     for i in frame_ids][::seq_stride])
                    obds_seq.append([[vid_annots[i]['OBD_speed']] for i in frame_ids][::seq_stride])
                    gpss_seq.append([[vid_annots[i]['GPS_speed']] for i in frame_ids][::seq_stride])
                    head_ang_seq.append([[vid_annots[i]['heading_angle']] for i in frame_ids][::seq_stride])
                    yrp_seq.append([[(vid_annots[i]['yaw'], vid_annots[i]['roll'], vid_annots[i]['pitch'])]
                                    for i in frame_ids][::seq_stride])

        print('Subset: %s' % image_set)
        print('Number of pedestrians: %d ' % num_pedestrians)
        print('Total number of samples: %d ' % len(image_seq))

        return {'image': image_seq,
                'pid': pids_seq,
                'bbox': box_seq,
                'center': center_seq,
                'occlusion': occ_seq,
                'obd_speed': obds_seq,
                'gps_speed': gpss_seq,
                'heading_angle': head_ang_seq,
                'gps_coord': gpsc_seq,
                'yrp': yrp_seq,
                'intention_prob': intent_seq,
                'activities': activities,
                'image_dimension': self._get_dim()}

    def _get_intention(self, image_set, annotations, **params):
        """
        Generates intention data.
        :param image_set: Data split to use
        :param annotations: Annotations database
        :param params: Parameters to generate data (see generade_database)
        :return: A dictionary of trajectories
        """
        print('---------------------------------------------------------')
        print("Generating intention data")

        num_pedestrians = 0
        seq_stride = params['fstride']
        sq_ratio = params['squarify_ratio']
        height_rng = params['height_rng']

        intention_prob, intention_binary = [], []
        image_seq, pids_seq = [], []
        box_seq, center_seq, occ_seq = [], [], []
        set_ids, _pids = self._get_data_ids(image_set, params)

        for sid in set_ids:
            for vid in sorted(annotations[sid]):
                img_width = annotations[sid][vid]['width']
                pid_annots = annotations[sid][vid]['ped_annotations']
                for pid in sorted(pid_annots):
                    if params['data_split_type'] != 'default' and pid not in _pids:
                        continue
                    num_pedestrians += 1
                    exp_start_frame = pid_annots[pid]['attributes']['exp_start_point']
                    critical_frame = pid_annots[pid]['attributes']['critical_point']
                    frames = pid_annots[pid]['frames']

                    start_idx = frames.index(exp_start_frame)
                    end_idx = frames.index(critical_frame)

                    boxes = pid_annots[pid]['bbox'][start_idx:end_idx + 1]
                    frame_ids = frames[start_idx:end_idx + 1]
                    images = [self._get_image_path(sid, vid, f) for f in frame_ids]
                    occlusions = pid_annots[pid]['occlusion'][start_idx:end_idx + 1]

                    if height_rng[0] > 0 or height_rng[1] < float('inf'):
                        images, boxes, frame_ids, occlusions = self._height_check(height_rng,
                                                                                  frame_ids, boxes,
                                                                                  images, occlusions)
                    if len(boxes) / seq_stride < params['min_track_size']:
                        continue

                    if sq_ratio:
                        boxes = [self._squarify(b, sq_ratio, img_width) for b in boxes]

                    int_prob = [[pid_annots[pid]['attributes']['intention_prob']]] * len(boxes)
                    int_bin = [[int(pid_annots[pid]['attributes']['intention_prob'] > 0.5)]] * len(boxes)

                    image_seq.append(images[::seq_stride])
                    box_seq.append(boxes[::seq_stride])
                    occ_seq.append(occlusions[::seq_stride])

                    intention_prob.append(int_prob[::seq_stride])
                    intention_binary.append(int_bin[::seq_stride])

                    ped_ids = [[pid]] * len(boxes)
                    pids_seq.append(ped_ids[::seq_stride])

        print('Subset: %s' % image_set)
        print('Number of pedestrians: %d ' % num_pedestrians)
        print('Total number of samples: %d ' % len(image_seq))

        return {'image': image_seq,
                'bbox': box_seq,
                'occlusion': occ_seq,
                'intention_prob': intention_prob,
                'intention_binary': intention_binary,
                'ped_id': pids_seq}
