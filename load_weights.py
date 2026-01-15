import torch
from gen import RRDBNet 

def load_model(weight_path=r"path to the weights", device="cuda"):
    model = RRDBNet().to(device)

    state_dict = torch.load(weight_path, map_location=device)
    new_state_dict = {}
    if 'params_ema' in state_dict:
        loaded_weights = state_dict['params_ema']
    elif 'params' in state_dict:
        loaded_weights = state_dict['params']
    else:
        loaded_weights = state_dict
    for k, v in loaded_weights.items():
        if not k.startswith('model.'):
            new_state_dict['model.' + k] = v
        else:
            new_state_dict[k] = v
    model.load_state_dict(new_state_dict, strict=True)
    model.eval()
    return model
