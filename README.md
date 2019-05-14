# GwNet

## Data

### Download METR-LA and PEMS-BAY data from [DCRNN](https://github.com/liyaguang/DCRNN)

### Follow DCRNN scripts to preprocess data.

## Run experiments

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

