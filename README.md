# README

## Arguments

```
args
--data data directory, used when 2D features are added
--feature whether to include feature, default false
--reward whether to include reward, default true
--input_size default 24 (39 with 2D feature)
--f reward function, default 'sparse' (sparse, smooth, both)
--name dataset name, default 'IEDB_Jespersen'
--log output log name, default 'train', overwrite mode
--maxlen length used in data standardization, default 2000
--seed torch.manual_seed(), default 1
--out output directory, default can be set in consts.py
```

## Run

Default run: sparse reward, output train.log to the output directory set in consts.py

```python
python lstm.py
```

Baseline run: no reward

```python
python lstm.py --reward False --log 'baseline'
```

Sample run: smooth reward, output to specific directory

```python
python lstm.py --f 'smooth' --log 'smooth' --seed 2 --out './output/'
```

