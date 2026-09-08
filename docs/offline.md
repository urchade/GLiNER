# Loading models offline

Prepare a complete model directory on a machine with internet access:

```python
from gliner import GLiNER

model = GLiNER.from_pretrained("urchade/gliner_multi-v2.1")
model.save_pretrained("gliner-offline", safe_serialization=True)
```

Copy the entire `gliner-offline` directory to the offline machine, then load it:

```python
from gliner import GLiNER

model = GLiNER.from_pretrained("gliner-offline", local_files_only=True)
```

`save_pretrained` includes model weights, the resolved backbone configuration,
and tokenizers. Models with a separate label encoder or generative decoder also
save their auxiliary tokenizer in `labels_tokenizer/` or `decoder_tokenizer/`.
Keep these subdirectories with the model when copying it.

Older Hub checkpoints may contain a GLiNER configuration that only names the
backbone, such as `microsoft/mdeberta-v3-base`, without embedding its configuration.
Downloading that checkpoint's files alone may therefore be insufficient. Loading
and saving it with the code above resolves and packages those dependencies.

`local_files_only=True` restricts loading to local files and the Hugging Face
cache; it does not download missing dependencies. An incomplete legacy checkpoint
still requires its missing backbone configuration or tokenizer to be cached or
provided locally. If the backbone files are in a separate local directory,
`model_name` in `gliner_config.json` can point to that directory. Missing files
raise an error without attempting a network connection.
