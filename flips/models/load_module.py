# Copyright (c) 2025 Max Planck Institute for Polymer Research
# Licensed under the MIT license.

import importlib

def load_module(object):
    module, object = object.rsplit(".", 1)
    # NOTE: would be good to rename gafl_flex to flips completely eveywhere...
    # backwards compatibility:
    if module == "models.gafl.flow_model_final":
        # module = "flips.models.gafl.flow_model_old"
        module = "flips.models.gafl.flow_model"
    elif not module.startswith("flips."):
        module = "flips." + module

    module = importlib.import_module(module)
    fn = getattr(module, object)
    return fn