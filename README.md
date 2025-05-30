## Requirements
- Python version 3.11

## Installation
Install the dependencies:
``` sh
python -m venv myenv
source myenv/bin/activate
pip install -r requirements.txt
```

## Start training
To start training, use the following command:

1. Movielens-100K
``` sh
python train.py --lr 0.1 --reg 0.005 --cl_t 0.1 --item_cluster 30 --dataset 100k
```

2. Movielens-1M
``` sh
python train.py --lr 0.1 --reg 0.005 --cl_t 0.5 --item_cluster 45 --dataset ml-1m
```

3. Lastfm-2K
``` sh
python train.py --lr 0.05 --reg 0.001 --cl_t 0.5 --item_cluster 500 --dataset lastfm-2k
```

4. HetRec2011
``` sh
python train.py --lr 0.1 --reg 0.005 --cl_t 0.5 --item_cluster 45 --dataset hetrec
```