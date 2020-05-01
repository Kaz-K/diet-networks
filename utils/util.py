import os
import json
import random
import collections
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def load_json(path):
    def _json_object_hook(d):
        for k, v in d.items():
            d[k] = None if v is False else v
        return collections.namedtuple('X', d.keys())(*d.values())
    def _json_to_obj(data):
        return json.loads(data, object_hook=_json_object_hook)
    return _json_to_obj(open(path).read())


def check_manual_seed(seed):
    seed = seed or random.randint(1, 10000)
    random.seed(seed)
    torch.manual_seed(seed)
    return seed


def load_model(model, save_path):
    model_state = torch.load(save_path)
    if isinstance(model, nn.DataParallel):
        model_state = model_state.module.state_dict()
    else:
        model_state = model_state.state_dict()
    model.load_state_dict(model_state['model'])


def get_output_dir_path(save_config, study_name):
    study_time = datetime.datetime.now().strftime('%b%d_%H-%M-%S')
    dir_name = study_name + '_' + study_time
    output_dir_path = os.path.join(
        save_config.output_root_dir, dir_name
    )
    os.makedirs(output_dir_path, exist_ok=True)
    return output_dir_path
