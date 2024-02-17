import torch
from model import GLiNER


def save_model(current_model, path):
    config = current_model.config
    dict_save = {"model_weights": current_model.state_dict(), "config": config}
    torch.save(dict_save, path)


def load_model(path, model_name=None, device=None):
    dict_load = torch.load(path, map_location=torch.device('cpu'))
    config = dict_load["config"]

    if model_name is not None:
        config.model_name = model_name

    loaded_model = GLiNER(config)
    loaded_model.load_state_dict(dict_load["model_weights"])
    return loaded_model.to(device) if device is not None else loaded_model
