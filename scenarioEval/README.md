# Scenario Evaluation
This folder contains the code for extracting scenarios from [PIE](http://data.nvision2.eecs.yorku.ca/PIE_dataset/) and
[JAAD](https://data.nvision2.eecs.yorku.ca/JAAD_dataset/) datasets and computing metrics as proposed in the following papers:
* A. Rasouli, "[A Novel Benchmarking Paradigm and a Scale- and Motion-Aware Model for Egocentric Pedestrian Trajectory Prediction](https://arxiv.org/pdf/2310.10424)",
ICRA, 2024
* A. Rasouli, I. Kotseruba, "[Diving Deeper Into Pedestrian Behavior Understanding: Intention Estimation, Action Prediction, and Event Risk Assessment](https://arxiv.org/pdf/2407.00446)",
IV, 2024


## Data
For evaluation, [PIE](../annotations/README.md#top) and [JAAD](https://github.com/ykotseruba/JAAD) annotations are required. 
Note that the path to the datasets are provided as environment variables, e.g. ```os.environ['PIE_PATH']``` the value of which should be replaced by
the path to the corresponding datasets. Note that for evaluation, the images/videos are not required. Simply create a folder for 
the dataset, e.g. ```PIE``` and unzip the annotations from ```../annotations/*``` in that folder. 

## Evaluate
A sample script ```scenario_scripts.py``` is provided to replicate the results as reported in the papers. All 
configurations for metrics and scenario-based analysis are provided in ```../utilities/configs.yaml```. See the comments 
for the parameters within the configuration file for more information.

<a name="trajectory"></a>
### Trajectory prediction
To replicate the results of ENCORE-D on PIE as in [A Novel Benchmarking Paradigm and a Scale- and Motion-Aware Model for Egocentric Pedestrian Trajectory Prediction](https://arxiv.org/pdf/2310.10424)
use the following function: 
```
from utilities.data_gen_utils import get_tracks, get_dataset
from utilities.utils import get_config, get_predictions
from scenarioEval.trajectory_evaluate import evaluate_trajectory_scenario

evaluate_scenario_trajectory(config_path=```../utilities/configs.yaml```,
                             pred_path=```../model_outputs/ec_traj_pie.csv```)
```
For [PedFormer](https://arxiv.org/pdf/2210.07886), change the prediction path to ```../model_outputs/pf_traj_pie.csv```.

To generate the results of single factor analysis (as reported in **Tables IV** and **V** ) 
for all scenarios as well as overall performance (as in **Table VI**), default parameters should be used. In this case
we use observation and prediction lengths (```obs_len/pred-len```) of 15 and 45 respectively with samples being extracted
from pedestrian sequence with %50 ```overlap```. For evaluation, we use single factor scenario specifications 
```configs['scenarios']['traj_tf']```. For additional parameters see the configuration file ```../utilities/configs.yaml```.

Executing this function with default parameters will generate a file ```scenario_results/eval_res.csv``` containing the results.
If ```save_file_path``` in ```../utilities/configs.yaml``` is not changed after the initial execution, subsequent files are saved as 
```scenario_results/eval_res_#.csv```  where ```#``` will be the largest existing file number + 1. For example, after second execution
results are written to ```scenario_results/eval_res_1.csv``` and after fifth execution to ```scenario_results/eval_res_4.csv```.

The results file is a csv file while columns defined as follows: 
```
sceanrio, sub-scen, count, B_mse_15, B_mse_30, ..., sCF_mse
```
The first two columns correspond to the scenario (e.g. ```ped_scale``` for pedestrian scale) and sub-scenario
(e.g. ```0-50```). The third column, ```count``` shows the total number of samples in the sub-scenario and the 
remaining columns correspond to different metrics. For scenario-analysis, in the main tables IV and V, ```B_mse/sB_mse``` from
columns 6 and 14 are reported. The overall results computed by averaging over the entire test set are reported in the 
last row, ```all, all```.

For two factor scenario-analysis simply use ```configs['scenarios']['traj_tf']``` as follows: 

```
def evaluate_scenario_trajectory(...):
    ...
    evaluate_trajectory_scenario(predictions, gt,
                              configs=configs, data=data,
                              scenarios=configs['scenarios']['traj_tf'],
                              verbose=True)
```

The results file structure is similar to single factor ones. Here, under ```scenario``` and ```sub-scen``` columns, values 
are separated by ```;```s. For example, ```ped_scale;veh_speed``` means pedestrian scale vs vehicle speed.  

Note that additional dimensions for scenario analysis can be defined, e.g. ```['ped_scale','veh_speed', 'ped_scale'] ```. 
However, as the number of factor dimensions increases, the sub-scenarios more likely become empty given the limited size of 
the datasets. In such cases, the values are set to ```nan```. Lastly, single and multiple factor scenarios can be defined in a 
single configuration and passed for processing at once.

### Action prediction
To replicate the results in [Diving Deeper Into Pedestrian Behavior Understanding: Intention Estimation, Action Prediction, and Event Risk Assessment](https://arxiv.org/pdf/2407.00446)
use 
```
from utilities.data_gen_utils import get_tracks, get_dataset
from utilities.utils import get_config, get_predictions
from scenarioEval.action_evaluate import evaluate_action_scenario

evaluate_scenario_action(config_path='../utilities/configs.yaml')
```
The overall process of generating the results is similar to [trajectory prediction](#trajectory) and 
results are saved to ```scenario_results/eval_res.csv``` structured as follows:

```
scenario, sub-scen, count, count_0, ..., count_N, <task>_Acc, <task>_bAcc,...
```
where ```N``` is the total number of classes in the given ```task```, which is either **act**ion, **intetion** or **risk** assessment. 
As shown, the main difference is that besides the overall  ```count``` of samples, 
the number of samples in each class is also reported. For instance, for action ```count_0``` and ```count_1``` correspond to
the number of samples in non-crossing and crossing categories, respectively.

Executing the function with default parameters generates the scenario analysis results reported in **Table IV** paper (see column ```x_mAP``` 
where x is either **action** or **intention**) and overall results in tables
II, III, and VI. To generate results for different tasks simply change ```dset``` to **action**, **intention**, and **risk** as follows:

```
def evaluate_scenario_action(...):
    ...
    dset = 'risk' # to generate results for risk assessment task
```

As for other parameters, we set time to event (TTE) to 1s-3s (or 30-90 frames given that our datasets are 30 fps). The tracks are
also sampled at 30% overall (as opposed to 50% for trajectory) to ensure that first and last tracks sampled from the sequences
are matching to the TTE boundaries. 

We also set the prediction len ```pred_len``` to 1s (or 30 frames). This is the prediction horizon of [PedFormer](https://arxiv.org/pdf/2210.07886)
which is equal to the lower value of TTE. For all models, including action prediction models, such as [SFGRU](https://arxiv.org/pdf/2005.06582)
that do not generate future trajectories, we generate the evaluation data for scenario analysis the same way. This is because 
even though the number of samples and observation tracks do not change when setting prediction length to 0,
scenario analysis can change significantly. This is due to the fact that scenario statistics are generated according to full tracks (obs+pred),
NOT only the observation portion. Hence, the distribution of data in different categories changes based on the length of the tracks.




### Citation
If you use the code for scenario extraction and evaluation, please cite the following papers:
```
@InProceedings{Rasouli_2024_ICRA,
author = {Rasouli, Amir},
title = {A Novel Benchmarking Paradigm and a Scale- and Motion-Aware Model for Egocentric Pedestrian Trajectory Prediction},
booktitle = {International Conference on Robotics and Automation (ICRA)},
year = {2024}
}

@InProceedings{Rasouli_2024_IV,
author = {Rasouli, Amir and Kotseruba, Iuliia},
title = {Diving Deeper Into Pedestrian Behavior Understanding: Intention Estimation, Action Prediction, and Event Risk Assessment},
booktitle = {Intelligent Vehicles Symposium (IV)},
year = {2024}
}
```

#### Disclaimer
This code has been tested on the **PIE** and **JAAD** datasets with the provided **model outputs** and **architectures evaluated** in the paper.
If you encounter any issues, please report for a solution. **Note that any changes to the code or configuration for evaluation on different datasets or
models are at the user's discretion and no support will be provided.**
