from .nonadaptivea3c_train import nonadaptivea3c_train
from .nonadaptivea3c_val import nonadaptivea3c_val
from .savn_train import savn_train
from .savn_val import savn_val

trainers = [ 
    'vanilla_train',
    'learned_train',
]

testers = [
    'vanilla_val',
    'learned_val',
]

variables = locals()