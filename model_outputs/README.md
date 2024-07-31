# Model Output Samples
This folder contains the output of state-of-the-art algorithms, [ENCORE](https://arxiv.org/pdf/2310.10424) and [PedFormer](https://arxiv.org/pdf/2210.07886) for trajectory and action prediction, intention estimation, and risk assessment tasks.  

Files are formatted as ```<model_abbrev>_<task>_<dataset>.csv```. To load the model output files use ```from utilities.utils import get_predictions```.
See [scenario script](../scenarioEval/scneario_scripts.py) for use-case example. The outputs can be used to replicate the 
results in the following papers:


<a name="citation"></a>
### Citation
If you use the model outputs for evaluation, please cite the corresponding paper(s).

For **ENCORE** trajectory prediction:
```
@InProceedings{Rasouli_2024_ICRA,
author = {Rasouli, Amir},
title = {A Novel Benchmarking Paradigm and a Scale- and Motion-Aware Model for Egocentric Pedestrian Trajectory Prediction},
booktitle = {International Conference on Robotics and Automation (ICRA)},
year = {2024}
}
```
For **PedFormer** behavior prediction and estimation:

```
@InProceedings{Rasouli_2024_IV,
author = {Rasouli, Amir and Kotseruba, Iuliia},
title = {Diving Deeper Into Pedestrian Behavior Understanding: Intention Estimation, Action Prediction, and Event Risk Assessment},
booktitle = {Intelligent Vehicles Symposium (IV)},
year = {2024}
}

@InProceedings{Rasouli_2023_ICRA,
author = {Rasouli, Amir and Kotseruba, Iuliia},
title = {PedFormer: Pedestrian Behavior Prediction via Cross-Modal Attention Modulation and Gated Multitask Learning},
booktitle = {International Conference on Robotics and Automation (ICRA)},
year = {2023}
}
```
