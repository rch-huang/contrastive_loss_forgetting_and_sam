# lifelong-learning-pretraining-and-sam

Code for the paper "Mehta, S. V., Patil, D., Chandar, S., & Strubell, E. (2023). [An Empirical Investigation of the Role of Pre-training in Lifelong Learning](https://jmlr.org/papers/v24/22-0496.html). The Journal of Machine Learning Research 24 (2023)"
 
## Installation

### Requirements
Python 3.6, PyTorch 1.7.0, transformers 2.9.0


### Setting up a virtual environment

[Conda](https://conda.io/) can be used to set up a virtual environment
with Python 3.6 in which you can
sandbox dependencies required for our implementation:

1.  [Download and install Conda](https://conda.io/docs/download.html).

2.  Create a Conda environment with Python 3.6

    ```
    conda create -n lll python=3.6
    ```

3.  Activate the Conda environment.  (You will need to activate the Conda environment in each terminal in which you want to run our implementation).

    ```
    conda activate lll
    ```

### Setting up our environment

1. Visit http://pytorch.org/ and install the PyTorch 1.7.0 package for your system.

    ```
    conda install pytorch==1.7.0 cudatoolkit=11.0 -c pytorch
    ```

2. Install other requirements

   ```
   pip install -r requirements.txt
   ```

That's it! You're now ready to reproduce our results.

# Running Vision & NLP Experiments

## 0. Setting up datasets

1. First create the data directory:
    ```
    mkdir data
    ```
2. To download the data for <b>Split CIFAR-100</b> and <b>Split CIFAR-50</b> experiments, run:
    ```
    ./scripts/download_vision_data.sh cifar100
    ```
3. To download data for <b>5-dataset</b> experiments, run:
    ```
    ./scripts/download_vision_data.sh 5data
    ```
 
## 1. Running Lifelong Learning Experiments

#### A. Vision

To run the vision experiments and create the necessary model checkpoints for random initialization, run:
```
./scripts/run_vision.sh \ 
    {DATASET} \ 
    {METHOD} \
    ./data \
    ./output/{DATASET}/{METHOD}/random/run_1 \
     1 \
     random \
     5 \
     {CUDA_DEVICE} \
     rndminit_ckpt \
     no_sam
```
where `{DATASET}` is one of `"5data", "cifar50", "cifar100"`, and `{METHOD}` is 
one of `"sgd", "er", "ewc"`.

Similarly, to run and create the necessary model checkpoints for pre-trained initialization, run:
```
./scripts/run_vision.sh \
    {DATASET} \
    {METHOD} \
    ./data \
    ./output/{DATASET}/{METHOD}/pt/run_1 \
    1 \
    pt \
    5 \
    {CUDA_DEVICE} \
    imagenetinit_ckpt \
    no_sam
```
where `{DATASET}` is one of `"5data", "cifar50", "cifar100"`, and `{METHOD}` is 
one of `"sgd", "er", "ewc"`.

The above run commands will create a folder called `output` with all of the relevant data for the 
run as well as the model checkpoints. In our experiments, we run this with 5 different random seeds. The data in `Table 5` for vision experiments is generated based on the `log.json` files in each run folder.

## 2. Running the analysis

### I) Sharpness
Create the folders:
```
mkdir -p results/analysis/sharpness/{DATASET}/random
mkdir -p results/analysis/sharpness/{DATASET}/pt
```
for each dataset of interest (`5data, cifar50, cifar100`).

#### A. Vision

To run the sharpness analysis, we run the following command:
```
./scripts/run_vision_analysis.sh \
    {DATASET} \
    ./data \
    ./output/{DATASET}/random/run_1 \
    ./results/analysis/sharpness/{DATASET}/random/run_1.json \
    sharpness \
    0
```

Run a similar command for the pre-trained models.

## Citation
If you use our code in your research, please cite: [An Empirical Investigation of the Role of Pre-training in Lifelong Learning](https://jmlr.org/papers/v24/22-0496.html)

```
@article{JMLR:v24:22-0496,
  author  = {Sanket Vaibhav Mehta and Darshan Patil and Sarath Chandar and Emma Strubell},
  title   = {An Empirical Investigation of the Role of Pre-training in Lifelong Learning},
  journal = {Journal of Machine Learning Research},
  year    = {2023},
  volume  = {24},
  number  = {214},
  pages   = {1--50},
  url     = {http://jmlr.org/papers/v24/22-0496.html}
}
```
