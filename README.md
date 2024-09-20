# Pedestrian Intention Estimation (PIE)
<p align="center">
<img src="demos/pie_annotations.png" alt="pie_annotations" align="middle" width="600"/>
</p>
<br/><br/>

This repository contains code and annotations for the Pedestrian Intention Estimation ([PIE](http://data.nvision2.eecs.yorku.ca/PIE_dataset/)) dataset:
[A. Rasouli, I. Kotseruba, T. Kunic, J.K. Tsotsos, PIE: A Large-Scale Dataset and Models for Pedestrian Intention Estimation, ICCV, 2019](https://openaccess.thecvf.com/content_ICCV_2019/papers/Rasouli_PIE_A_Large-Scale_Dataset_and_Models_for_Pedestrian_Intention_Estimation_ICCV_2019_paper.pdf) and
a series of scripts for visualization and scenario evaluation.

### Download videos
Videos are grouped into 6 sets corresponding to different routes driven in Toronto, Canada. 

The total size of all video clips is approx. **74GB**.

**Download links** [YorkU server](http://data.nvision2.eecs.yorku.ca/PIE_dataset/PIE_clips/) 
[Google Drive](https://drive.google.com/drive/folders/180MXX1z3aicZMwYu2pCM0TamzUKT0L16?usp=drive_link)

### Content
* [annotations](annotations/README.md#top): Contains the annotations and a script for extracting images from raw videos
* [scenarioEval](scenarioEval/README.md#top): Contains code for scenario generation and metrics for trajectory and action prediction
* [visualization](visualization/README.md#top): Contains a series of scripts for visualizing data samples and illustrating trajectory and action prediction modules
* [model_outputs](model_outputs/README.md#top): Contains models' outputs for behavior prediction to be used for evaluation and visualization
* [utilities](utilities/README.md#top): Contains configuration file for evaluation and visualization, dataset interfaces, and other utility functions
* [camera_params](camera_params): Contains camera parameters for the PIE dataset

<a name="citation"></a>
### Citation
If you use our dataset, please cite:
```
@InProceedings{Rasouli_2019_ICCV,
author = {Rasouli, Amir and Kotseruba, Iuliia and Kunic, Toni and Tsotsos, John K.},
title = {PIE: A Large-Scale Dataset and Models for Pedestrian Intention Estimation and Trajectory Prediction},
booktitle = {International Conference on Computer Vision (ICCV)},
year = {2019}}
```
<a name="authors"></a>
### Authors

* **[Amir Rasouli](https://aras62.github.io/)**
* **[Iuliia Kotseruba](http://www.cse.yorku.ca/~yulia_k/)**

Please send an email to yulia_k@eecs.yorku.ca or arasouli.ai@gmail.com if there are any problems with downloading or using the data.

<a name="license"></a>
### License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details
