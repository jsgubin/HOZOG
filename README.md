# [Start HOZOG]

HOZOG is a new hyperparameter optimization technique  with zeroth-order hyper-gradients. The experimental results demonstrate the benefits of  HOZOG in terms of simplicity, scalability, flexibility, effectiveness and efficiency compared with the state-of-the-art hyperparameter optimization methods.

## Installation

- Clone the repo: `git clone https://github.com/hozog.git`
- Create conda environment by `conda env create -f {HOZOG}.yaml`

## Quick Start

### An example

Run test_mp.py for a try on data cleaning task.

![alt text](https://github.com/jsgubin/HOZOG/blob/main/HOZO/results/results.png)

### Build a model class in Python

Build a model for task, the model must have two function: *\_\_int\_\_* and *train_valid*. The first parameter of *\_\_int\_\_* must be the hyper-parameters. Other parameters for *\_\_int\_\_* can be a *init_model_dict* defined in main function. *train_valid* must receive data (can be a data_dict defined in main function) and return a validation loss.

### Create a HOZOG instance

Create a HOZOG instance by 

```python
hozo_example=hozo.HOZO(model=model_class_you_define, max_iter=2000, eta=40, q=5, mu=1e-3). 
```

eta is hyper step size (hyper leraning rate). q and mu are parameters defined in HOZOG paper.

### Define dicts you need

For example:

```python
init_model_dict = 'num gpus':num gpus,'T':T,'lr':lr, 'times':times
data_dict = 'data':data
kw = 'init model dict':init_model_dict, 'data dict':data_dict
```

### Fit with HOZOG

Run HOZOG algorithm with parallel multi-processes as follows:

```python
hozo_example.fit(lmd0=lambda0, **kw) 
```

lmd0 denotes the init value of hyperparameters.

## Tips

A sigmoid function for hyper-parameters may make it run better.

You can change process_num in HOZOG class to make it faster.

## License

Code released under the [MIT](https://github.com/) license.

## Reference
Gu, B., Liu, G., Zhang, Y., Geng, X. and Huang, H., 2021. Optimizing Large-Scale Hyperparameters via Automated Learning Algorithm. arXiv preprint arXiv:2102.09026.
