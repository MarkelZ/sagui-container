# Sagui Container

A Docker container for replicating experiments from [Reinforcement Learning by Guided Safe Exploration](https://arxiv.org/abs/2307.14316) using adapted versions of OpenAI's [safety-gym](https://github.com/openai/safety-gym) and [SaGui](https://github.com/qisong-yang/SaGui).


## Installation

### Step 1: Clone the repository

```sh
git clone https://github.com/MarkelZ/sagui-container.git
cd sagui-container
```

### Step 2: Download Mujoco 2.0

This code relies on `mujoco_py 2.0.2.5` which requires Mujoco 2.0.

Create a `mujoco/` directory within `sagui-container/` to store Mujoco-related files:

```sh
mkdir mujoco
```

Download Mujoco 2.0 (`mujoco200.tar.gz`) from [here](https://www.roboti.us/download.html) and extract it into the `mujoco/` directory.

Obtain the activation from [here](https://www.roboti.us/license.html) and save the downloaded file as `mjkey.txt` in the `mujoco/` directory.

Your file structure should look like this:

```
sagui-container 
└── mujoco/
    ├── mujoco200/
    └── mjkey.txt
```

### Step 3: Build the docker container

```sh
docker build -t sagui-container .
```


## Replicating the experiments

Start the docker container with:

```sh
docker run -it sagui-container
```

This should open a shell.

To train the guide, run:
```
cd /path/to/sagui

python train-guide.py --env GuideENV -s SEED --cost_lim d --logger_kwargs_str '{"output_dir": "./guide"}'
```

If you want to test the two versions of SaGui with a well-trained guide, run:
```
cd /path/to/sagui

python sagui-cs.py /path/to/guide --env StudentENV -s SEED --cost_lim d --logger_kwargs_str '{"output_dir": "./xxx"}'

python sagui-ld.py /path/to/guide --env StudentENV -s SEED --cost_lim d --logger_kwargs_str '{"output_dir": "./xxx"}'
```

where we should have the guide in `/path/to/guide`.

`SEED` is the random seed (we use 0, 10, ..., 90 in the paper experiments), `d` is the real-world safety threshold, and `'{"output_dir": "./xxx"}'` indicates where to store the data. 

As to hyperparameters and experimental setup, you can refer to the paper and its Appendix.


## Licenses and Ownership

This project simplifies the installation and execution of code from [safety-gym](https://github.com/openai/safety-gym) and [SaGui](https://github.com/qisong-yang/SaGui), which is licensed under the MIT license.
