# Graph WaveNet for Deep Spatial-Temporal Graph Modeling

This is the original pytorch implementation of Graph WaveNet in the following paper: 
[Graph WaveNet for Deep Spatial-Temporal Graph Modeling, IJCAI 2019] (https://arxiv.org/abs/1906.00121),
with modifications presented in [Incrementally Improving Graph WaveNet Performance on Traffic Prediction] (https://arxiv.org/abs/1912.07390):


<p align="center">
  <img width="350" height="400" src=./fig/model.png>
</p>

## Requirements
- python 3
- see `requirements.txt`


## Data Preparation

1) Download METR-LA and PEMS-BAY data from [Google Drive](https://drive.google.com/open?id=10FOTa6HXPqX8Pf5WRoRwcFnW9BrNZEIX) or [Baidu Yun](https://pan.baidu.com/s/14Yy9isAIZYdU__OYEQGa_g) links provided by [DCRNN](https://github.com/liyaguang/DCRNN).

2)

```
# Create data directories
mkdir -p data/{METR-LA,PEMS-BAY}

# METR-LA
python generate_training_data.py --output_dir=data/METR-LA --traffic_df_filename=data/metr-la.h5

# PEMS-BAY
python generate_training_data.py --output_dir=data/PEMS-BAY --traffic_df_filename=data/pems-bay.h5

```

## Train Commands
Note: train.py saves metrics to a directory specified by the `--save` arg in metrics.csv and test_metrics.csv

Model that gets (3.00 - 3.02 Test MAE, ~2.73 Validation MAE)
```
python train.py --cat_feat_gc --fill_zeroes --do_graph_conv --addaptadj  --randomadj --es_patience 20 --save logs/baseline_v2
```

Finetuning (2.99 - 3.00 MAE)
```
python generate_training_data.py --seq_length_y 6 --output_dir data/METR-LA_12_6
python train.py --data  data/METR-LA_12_6 --cat_feat_gc --fill_zeroes --do_graph_conv --addaptadj  --randomadj --es_patience 20 --save logs/front_6
python train.py --checkpoint  logs/front_6/best_model.pth --cat_feat_gc --fill_zeroes --do_graph_conv --addaptadj  --randomadj --es_patience 20 --save logs/finetuned

```
Original Graph Wavenet Model (3.04-3.07 MAE)
```
python train.py --clip 5 --lr_decay_rate 1. --nhid 32 --do_graph_conv --addaptadj  --randomadj --save logs/baseline
```

You can also train from a jupyter notebook with
```{python}
from train import main
from durbango import pickle_load
args = pickle_load('baseline_args.pkl') # manipulate these in python
args.lr_decay_rate = .97
args.clip = 3
args.save = 'logs/from_jupyter'
main(args) # takes roughly an hour depending on nhid, and early_stopping
```

Train models configured in Table 3 of the original GraphWavenet paper by using the `--adjtype, --addaptadj, --aptonly` command line argument.
These flags are (somewhat) documented in util.py.

Run unitests with `pytest`

### Possible Improvements
* move redundant `.transpose(1,3)` to dataloader or `load_dataset`
