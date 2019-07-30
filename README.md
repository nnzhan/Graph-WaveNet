# Graph WaveNet for Deep Spatial-Temporal Graph Modeling

This is the original pytorch implementation of Graph WaveNet in the following paper: 
[Graph WaveNet for Deep Spatial-Temporal Graph Modeling, IJCAI 2019] (https://arxiv.org/abs/1906.00121).

<p align="center">
  <img width="350" height="400" src=./fig/model.png>
</p>

## Requirements
- python 3
- pytorch
- scipy
- numpy
- pandas
- pyaml


## Data Preparation

### Step1: Download METR-LA and PEMS-BAY data from [Google Drive](https://drive.google.com/open?id=10FOTa6HXPqX8Pf5WRoRwcFnW9BrNZEIX) or [Baidu Yun](https://pan.baidu.com/s/14Yy9isAIZYdU__OYEQGa_g) links provided by [DCRNN](https://github.com/liyaguang/DCRNN).

### Step2: Follow [DCRNN](https://github.com/liyaguang/DCRNN)'s scripts to preprocess data.

```
# Create data directories
mkdir -p data/{METR-LA,PEMS-BAY}

# METR-LA
python -m scripts.generate_training_data --output_dir=data/METR-LA --traffic_df_filename=data/metr-la.h5

# PEMS-BAY
python -m scripts.generate_training_data --output_dir=data/PEMS-BAY --traffic_df_filename=data/pems-bay.h5

```
## Experiments
Train models configured in Table 3 of the paper.

```
ep=100
dv=cuda:0
mkdir experiment
mkdir experiment/metr

#identity
expid=1
python train.py --device $dv --gcn_bool --adjtype identity  --epoch $ep --expid $expid  --save ./experiment/metr/metr > ./experiment/metr/train-$expid.log
rm ./experiment/metr/metr_epoch*

#forward-only
expid=2
python train.py --device $dv --gcn_bool --adjtype transition --epoch $ep --expid $expid  --save ./experiment/metr/metr > ./experiment/metr/train-$expid.log
rm ./experiment/metr/metr_epoch*

#adaptive-only
expid=3
python train.py --device $dv --gcn_bool --adjtype transition --aptonly  --addaptadj --randomadj --epoch $ep --expid $expid  --save ./experiment/metr/metr > ./experiment/metr/train-$expid.log
rm ./experiment/metr/metr_epoch*

#forward-backward
expid=4
python train.py --device $dv --gcn_bool --adjtype doubletransition  --epoch $ep --expid $expid  --save ./experiment/metr/metr > ./experiment/metr/train-$expid.log
rm ./experiment/metr/metr_epoch*

#forward-backward-adaptive
expid=5
python train.py --device $dv --gcn_bool --adjtype doubletransition --addaptadj  --randomadj  --epoch $ep --expid $expid  --save ./experiment/metr/metr > ./experiment/metr/train-$expid.log
rm ./experiment/metr/metr_epoch*

```


