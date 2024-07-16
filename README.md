# Training / Finetuning a GLiNER Model

## Step 1: Install the Gliner Package
First, install the `gliner` package using pip:
```bash
pip install gliner==0.2.2
```

# Step 2: clone this branch
```bash
git clone -b training https://github.com/urchade/GLiNER.git
```


## Step 3: Train from scratch
```bash
python train.py --config config_span.yaml
```

## Step 3b: Finetuning an existing model
```bash
python train.py --config config_finetune.yaml
```
