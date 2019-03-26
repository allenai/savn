from .basic_episode import BasicEpisode
from .test_val_episode import TestValEpisode
__all__ = [
    'BasicEpisode',
    'TestValEpisode',
]

# All models should inherit from BasicEpisode
variables = locals()