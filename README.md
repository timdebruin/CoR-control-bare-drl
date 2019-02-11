# CoR-control-bare-drl
This repository is meant as a very minimal implementation of DRL methods to be used as a starting point for further experimentation. If you have fixes, or changes to the default parameters that will make the methods work better, please create a pull request or tell me. To use this code as a starting point, either fork the repo or download a local copy.

This code makes use of the [CoR control benchmarks](https://github.com/timdebruin/CoR-control-benchmarks), a set of control problems where the state of a dynamical system should be driven towards a reference (goal) state. Some of these benchmarks have physical counterparts within the [Cognitive Robotics](https://www.tudelft.nl/en/3me/departments/cognitive-robotics-cor/) department of the [Delft University of Technology](https://www.tudelft.nl/en/).

## Installation
 The code requires python 3.6+ with either tensorflow or tensorflow-gpu installed:
 
 For the CPU version of tensorflow:
 ```bash
pip install tensorflow
``` 
For the GPU version: 

 ```bash
pip install tensorflow-gpu
``` 

The remaining requirements can be installed by running the following command from the main repository directory (after downloading / cloning):
```bash
pip install -r requirements.txt
```

## Get going
The main directory contains the [simple_run.py](simple_run.py) file which currently includes a basic (and not overly effective) implementation of the [Normalized Advantage Functions](https://arxiv.org/abs/1603.00748) algorithm on the [Magman benchmark](https://github.com/timdebruin/CoR-control-benchmarks#magnetic-manipulator). It does include comments on things that can be improved, so good luck!
