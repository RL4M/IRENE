<img src="logo.png" width="200" height="200" align=right />

# Code for IRENE

This repository provides the code for IRENE. Based on the code, you can easily train your own IRENE by configuring your own dataset and modifying the training details (such as optimizer, learning rate, etc).

## Overview
As illustrated by the right figure [1], IRENE a new Transformer-based multi-modal medical diagnosis and prognosis paradigm. Different from the current deep learning powered diagnosis systems that mostly lean upon a non-unified way to fuse information from multiple sources, IRENE has the ability to learn holistic multi-modal representations progressively by treating input data in different modalities in a uniform way as sequences of tokens, simultaneously incorporating entire medical knowledge graph information.

## Setup the Environment
This software was implemented a system running `Ubuntu 16.04.4 LTS`, with `Python 3.7.6`, `PyTorch 1.8.1`, and `CUDA 11.4`. We have tried to reduced the number of dependencies for running the code. Nonetheless, you still need to install some necessary packages, such as `sklearn`, `PIL`, `apex (from NVIDIA)`, `matplotlib`, and `skimage`.

You can adjust the batch size to adapt to your own hardware environment. Personally, we recommend the use of four NVIDIA GPUs.

## Code Description
The main architecture of IRENE lies in the `models/` folder. The `modeling_irene.py` is the main backbone, while the rest necessary modules are distributed into different files based on their own functions, i.e., `attention.py`, `block.py`, `configs.py`, `embed.py`, `encoder.py`, and `mlp.py`. Please refer to each file to acquire more implementation details. 

`run.sh` includes the running script, which is:
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python -W ignore irene.py --CLS 8 --BSZ 64 --DATA_DIR ./data --SET_TYPE xxx.pkl
```
**Parameter description**:

`--CLS`: number of diseases.

`--BSZ`: batch size.

`--DATA_DIR`: location of the imaging data.

`--SET_TYPE`: file name of the clinical textual data (`***.pkl`).

Note that `xxx.pkl` is a dictionary that stores the clinical textual data in the format of `key-value`. Here is a short piece of code showing how to organize the `***.pkl`:
```python
>>> import pickle
>>> f = open('***.pkl', 'rb')
>>> subset = pickle.load(f)
>>> f.close()
>>> list(subset.keys())[0:10] # display top 10 case ids
>>> key = list(subset.keys())[0] # select the 1st key
>>> subset[key] # display the clinical data
>>> subset[key]['pdesc'] # the chief complaint feature
>>> subset[key]['bics'] # the demographics information (age and sex)
>>> subset[key]['bts'] #  the laboratory test results
>>> subset[key]['label'] # the disease labels
```

[1]: This figure is generated using Stable Diffusion, where the prompt is ``A Transformer that exploits multi-modal clinical information for medical diagnosis.''