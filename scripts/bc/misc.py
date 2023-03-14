import json
import random
import torch
import numpy as np

def control_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)

def tensorize(var, device='cpu'):
    """
    Convert input to torch.Tensor on desired device
    :param var: type either torch.Tensor or np.ndarray
    :param device: desired device for output (e.g. cpu, cuda)
    :return: torch.Tensor mapped to the device
    """
    if type(var) == torch.Tensor:
        return var.to(device)
    elif type(var) == np.ndarray:
        return torch.from_numpy(var).float().to(device)
    elif type(var) == float:
        return torch.tensor(var).float()
    else:
        print("Variable type not compatible with function.")
        return None

