# GNNPN-SC
The opensource for a graph neural network and pointer network-based approach for QoS-aware service composition

## Datasets
Download data from: 
[URL](https://bit.ly/3zxOSD9).

Extract the zipped file into ```<work_dir>/data``` after downloaded.

## Pre-trained models
Download pre-trained models from:
[URL](https://bit.ly/3NlamqC).

Extract the zipped file into ```<work_dir>/solutions/pretrained``` after downloaded.

## Example use cases
The framework is divided into three parts.
### Candidate Service Reduction
```angular2html
python main.py [dataset] ML

*[dataset] can be replaced by QWS or Normal.

For example:
python main.py QWS ML

The results are available after two or three epochs.
```
The model and the results are stored in ```<work_dir>/solutions/ML/<dataset>```.

### Initial Service Solution Construction
Two layers of pointer networks need to be trained separately.
```angular2html
python main.py [dataset] PNLow (MLEpoch)
python main.py [dataset] PNHigh (PNLowEpoch) (MLEpoch) 

*[dataset] can be replaced by QWS or Normal.

*(MLEpoch) means the number of epoch of ML's result you will use for training. 
-1 or no input indicate that you want to use the pre-trained model.

*(PNLowEpoch) means the number of epoch of PNLow's result you will use for training.
-1 or no input indicate that you want to use the pre-trained model.
if PNLowEpoch=-1, ML will automatically use the pre-trained model, which means that the training set needs to be the same for both PN layers.

For example:
python main.py QWS PNLow -1
python main.py QWS PNHigh 2 -1

The results for PNLow are available after 10-30 epochs.
The results for PNHigh are available after 30-40 epochs.
```
The model and the results are stored in ```<work_dir>/solutions/PNLow(PNHigh)/<dataset>```.

To get the performance of ML+2PN, please input:
```
python main.py [dataset] ML+2PN (PNHighEpoch)

*[dataset] can be replaced by QWS or Normal.

*(PNHighEpoch) means the number of epoch of PNHigh's result you will use.
-1 or no input indicate that you want to use the pre-trained model.

For example:
python main.py QWS ML+2PN -1
```

### Service Solution Fine-tuning
```angular2html
python main.py [dataset] WOA (PNHighEpoch)

*[dataset] can be replaced by QWS or Normal.

*(PNHighEpoch) means the number of epoch of PNHigh's result you will use.
-1 or no input indicate that you want to use the pre-trained model.

For example:
python main.py Normal WOA -1
```
The results are stored in ```<work_dir>/solutions/WOA/<dataset>/ML+2PN+WOA.txt```.

## Other baselines
```
python main.py [dataset] [Baseline] (MLEpoch)

*[dataset] can be replaced by QWS or Normal.

* [Baseline]: The project provides baselines for comparison, including:
** ML+ESWOA
** ML+DAAGA
** ESWOA
** DAAGA
** SDFGA
** DPKSD
** ML+PDDQN

* (MLEpoch) means the number of epoch of ML's result you will use.
-1 or no input indicate that you want to use the pre-trained model.
Only baselines with ML need this 

For example:
python main.py Normal ML+ESWOA -1
```
