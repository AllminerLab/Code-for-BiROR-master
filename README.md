### Environment

```
python = 3.8
torch = 1.12.0
cuda = 10.2.89
```

### Directory

- code

  data_process.py: Data preprocessing code before model training.

  main.py: Main function.

  model.py: Model function.

  train.py: Train function.

- data

  data_info: Used to save information related to the dataset.

  data_ix: Used to save the datasets after id mapping.

  ix_mapping: Used to save id and index mapping files.

  kg: Used to save knowledge graph datasets.

  raw_data: Used to save the original dataset.

- result

  Used to save experimental results.

### Run the Codes

- Run data preprocessing code

```sh
cd ./code
python data_process.py
```

- Run the model code

```
cd ./code
python main.py --entity_dim 128 --pro_dim 128 --word_dim 32 --hidden_dim 32 --n_hop 4 --lr 0.01 --ls1_weight 1 --ls2_weight 1 --lc1_weight 1 --lc2_weight 1 --lc3_weight 2 --l2_reg 0.
001 --batch_size 256 --epochs 10 --n_memory 64 --re_get_ripple_set True
```
