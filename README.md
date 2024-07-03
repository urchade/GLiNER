Sure, here's an improved version of your documentation in Markdown format:

# Training a Gliner Model

## Step 1: Install the Gliner Package
First, install the `gliner` package using pip:
```bash
pip install gliner==0.2.2
```

## Step 2: Train the Model
Next, train the model using the provided configuration file:
```bash
python train.py --config config_span.yaml
```
By default:
- **Training Dataset:** Pilener data located at `data/pilene_train.json`
- **Evaluation Dataset:** Cross-NER and MIT data located at `data/NER_datasets`
