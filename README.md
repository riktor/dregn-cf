# Density Ratio Based Personalised Ranking from Implicit Feedback

This repository is an example pytorch implementation of DREGN-CF ([arXiv](https://arxiv.org/abs/2011.05061)):

> Density Ratio Based Personalised Ranking from Implicit Feedback
Riku Togashi, Masahiro Kato, Mayu Otani, Shin'ichi Satoh.  
In Proceedings of the Web Conference 2021 (WWW '21).

DREGN-CF is a method based on semi-supervised density ratio estimation.
It minimises a risk derived from the weighted Bregman divergence.
The default implementation of DREGN-CF uses the risk estimator without importance sampling.
The codes are mainly based on the pytorch implementation of LightGCN by authors ([here](https://github.com/gusye1234/LightGCN-PyTorch)).

### Files

- `data/`
  - `gowalla/`    
    - need train.txt and test.txt here;
  - `yelp2018/`
    - need train.txt and test.txt here;
  - `amazonbook/`
    - need train.txt and test.txt here;
- `conf/`:
  - `dataset`: "dataset_name.yaml" contains the path configurations;
  - `model`: "model_name.yaml" contains the model configurations;
  - `config.yaml`: default config file;
- `code/`:
  - `main.py`: ;
  - `train.py`: the implementation of training process;
  - `model.py`: the implementation of DREGN-CF;
  - `dataset.py`: the implementation of the mini-batch sampler;
  - `logger.py`: logging utilities;
  - `evaluator/`: the code of C++ evaluator.

### Set up directories
Need to prepare directories to save logs, codes (including confs), and models.
The configuration for the directory paths is in "config.yaml".
```
  $ mkdir -p runs/log runs/code_backup runs/model_backup
```

### Set up for C++ evaluator
This repository includes the efficient C++ evaluator implemented in the original repository of [LightGCN](https://github.com/kuandeng/LightGCN).
```
  $ python setup.py build_ext --inplace
```
  
### Running the code
- AmazonBook
  ```
  $ python code/main.py experiment_name=amazon_book_dregncf model=dregncf dataset=amazon_book optimize.batch_size=2500 optimize.lr=0.01 optimize.train_epochs=100 reg_weight=5e-2 dr_upper_bound=80
  ```

- Yelp2018
  ```
  $ python code/main.py experiment_name=yelp2018_dregncf model=dregncf dataset=yelp2018 optimize.batch_size=2500 optimize.lr=0.01 optimize.train_epochs=150 reg_weight=5e-2 dr_upper_bound=70
  ```

- Gowalla  
  ```
  $ python code/main.py experiment_name=gowalla_dregncf model=dregncf dataset=gowalla optimize.batch_size=2500 optimize.lr=0.01 optimize.train_epochs=60 reg_weight=6e-2  dr_upper_bound=70
  ```
