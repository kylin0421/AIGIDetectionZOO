from .CNNDetection.cnnspot_api import CNNSPOT_API
from .UnivFD.univfd_api import UNIVFD_API
from .PatchCraft.patchcraft_api import PATCHCRAFT_API
from .LGrad.lgrad_api import LGRAD_API

import warnings
warnings.simplefilter("ignore", FutureWarning)



__version__ = "0.1.1.dev2"


AVAILABLE_MODELS=['cnnspot','univfd','patchcraft','lgrad']

def available_models():
    return AVAILABLE_MODELS



def load_detector(name: str, **kwargs):                 

    assert name in AVAILABLE_MODELS,f"Unknown detector: {name}, currently supported: {AVAILABLE_MODELS}"

    if name == "cnnspot":
        return CNNSPOT_API(**kwargs)
    elif name == "univfd":
        return UNIVFD_API(**kwargs)
    elif name == "patchcraft":
        return PATCHCRAFT_API(**kwargs)
    elif name == 'lgrad':
        return LGRAD_API(**kwargs)
    else:
        raise ValueError(f"Unknown detector: {name}.")