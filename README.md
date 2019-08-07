# PIE annotations
<p align="center">
<img src="pie_annotations.png" alt="pie_annotations" align="middle" width="600"/>
</p>
<br/><br/>

This repository contains  annotations for the Pedestrian Intetnion Estimation ([PIE](http://data.nvision2.eecs.yorku.ca/PIE_dataset/)) dataset. The annotations are in XML format and are accompanied with a python interface for processing.

### Table of contents
* [Annotations](#annotations)
* [Video clips](#clips)
* [Interface](#interface)
	* [Dependencies](#dependencies)
	* [Extracting images](#extracting)
	* [Using the interface](#usage)
		* [Parameters](#parameters)
		* [Sequence analysis](#sequence)
* [Citation](#citation)
* [Authors](#authors)
* [License](#license)

<a name="annotations"></a>
# Annotations
PIE annotations are organized according to sets and video clip names. There are ## types of labels has a unique id in the form of `<set_id>_<video_id>_< pedestrian_number>`. TODO: update for the types of labels

All samples are annotated with bounding boxes using two-point coordinates (top-left, bottom-right) `[x1, y1, x2, y2]`. The bounding boxes have corresponding occlusion tags. The occlusion values are either 0 (no occlusion), 1 (partial occlusion >25%) or 2 (full occlusion >75%).

 According to their types, the annotations are divided into 3 groups:<br/>
**Annotations**: These include objects bounding box coordinates, occlusion information and pedestrian activities (e.g. walking, looking).  These annotations are one per frame per label.<br/>
**Attributes**: These include information regarding pedestrians' demographics, crossing point, crossing characteristics, etc. These annotations are one per pedestrian.<br/>
**Vehicle**: These are vehicle sensor data, e.g. speed, heading angle, yaw, etc.<br/>

<a name="clips"></a>
# Video clips
PIE contains 6 sets 53 video clips. The clips in each set are continuous, meaning that they belong to a single recording that is divided into chunks. Each video is approximately 10 min long. These clips should be downloaded and placed in `PIE_clips` folder as follows:
```
PIE_clips/set01/video_0001.mp4
PIE_clips/set01/video_0002.mp4
...
```
To download the videos, either run script `download_clips.sh` or manually download the clips from [here](http://data.nvision2.eecs.yorku.ca/PIE_dataset/data/PIE_clips.zip) and extract the zip archive.

<a name="interface"></a>
# Interface

<a name="dependencies"></a>
## Dependencies
The interface is written and tested using python 3.5. The interface also requires
the following external libraries:<br/>
* opencv-python
* numpy
* scikit-learn

<a name="extracting"></a>
## Extracting images
In order to use the data, first, the video clips should be converted into images. This can be done using script `split_clips_to_frames.sh` or via interface as follows:
```
from pie_data import PIE
pie_path = <path_to_the_root_folder>
imdb = PIE(data_path=pie_path)
imdb.extract_and_save_images()
```

Using either of the methods will create a folder called `images` and save the extracted images grouped by corresponding video ids in the folder.
```
images/set01/video_0001/
								00000.png
								00001.png
								...
images/set01/video_0002/
								00000.png
								00001.png
								...		
...
```

<a name="usage"></a>
## Using the interface
<a name="parameters"></a>
Upon using any methods to extract data, the interface first generates a database (by calling `generate_database()`) of all annotations in the form of a dictionary and saves it as a `.pkl` file in the cache directory (the default path is `PIE/data_cache`). For more details regarding the structure of the database dictionary see comments in the `pie_data.py` for function `generate_database()`.

### Parameters
The interface has the following configuration parameters:
```
data_opts = {'fstride': 1,
             'data_split_type': 'default',
             'seq_type': 'trajectory',
	     			 'height_rng': [0, float('inf')],
	     			 'squarify_ratio': 0,
             'min_track_size': 0,
             'random_params': {'ratios': None,
                               'val_data': True,
                               'regen_data': True},
             'kfold_params': {'num_folds': 5, 'fold': 1}}
```
*'fstride'*.  This is used for sequence data. The stride specifies the sampling resolution, i.e. every nth frame is used
for processing.<br/>
*'data_split_type'*. The PIE data can be split into train/test or val in three different ways. `default` uses the predefined train/val/test split specified in `_get_image_set_ids()` method in `pie_data.py`. `random` randomly divides pedestrian ids into train/test (or val) subsets depending on `random_params` (see  method `_get_random_pedestrian_ids()` for more information). `kfold` divides the data into k sets for cross-validation depending on `kfold_params` (see  method `_get_kfold_pedestrian_ids()` for more information).<br/>
*'seq_type'*. Type of sequence data to generate (see [Sequence analysis](#sequence)).
*'height_rng'*. These parameters specify the range of pedestrian scales (in pixels) to be used. For example  `height_rng': [10, 50]` only uses pedestrians within the range of 10 to 50 pixels in height.<br/>
*'squarify_ratio'*. This parameter can be used to fix the aspect ratio (width/height) of bounding boxes. `0` the original bounding boxes are returned.<br/>
*'min_track_size'*. The minimum allowable sequence length in frames. Shorter sequences will not be used.


<a name="sequence"></a>
### Sequence analysis
There are three built-in sequence data generators accessed via `generate_data_trajectory_sequence()`. The type of sequences generated are `trajectory`, `intention` and `crossing`. To create a custom data generator, follow a similar structure and add a function call to `generate_data_trajectory_sequence()` in the interface.

<a name="citation"></a>
# Citation
If you use our dataset, please cite:
```
@inproceedings{rasouli2017they,
  title={PIE: A Large-Scale Dataset and Models for Pedestrian Intention Estimation and Trajectory Prediction},
  author={Rasouli, Amir and Kotseruba, Iuliia and Kunic, Toni and Tsotsos, John K},
  booktitle={ICCV},
  year={2019}
}

```
<a name="authors"></a>
## Authors

* **[Amir Rasouli](http://www.cse.yorku.ca/~aras/index.html)**
* **[Iuliia Kotseruba](http://www.cse.yorku.ca/~yulia_k/)**

Please send email to yulia_k@eecs.yorku.ca or aras@eecs.yorku.ca if there are any problems with downloading or using the data.

<a name="license"></a>
## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details
