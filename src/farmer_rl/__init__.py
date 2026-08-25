"""RL foundations for the Kaggriculture simulation competition.

The core package deliberately imports no optional training dependency.  PyTorch,
Gymnasium, Ray and kaggle-environments are loaded only by the components which
need them, so schema and feature tests remain cheap and deterministic.
"""

from .actions import ActionCandidate, CandidateGenerator, CandidateSet, JointActionCodec
from .collector import collect_episode
from .environment import KaggricultureEnv, StepResult
from .opponents import OpponentPool, OpponentSpec
from .tokenizer import ObservationTokenizer, TokenBatch
from .trajectory import EpisodeTrajectory, Transition

__all__ = [
    "ActionCandidate",
    "CandidateGenerator",
    "CandidateSet",
    "EpisodeTrajectory",
    "JointActionCodec",
    "KaggricultureEnv",
    "ObservationTokenizer",
    "OpponentPool",
    "OpponentSpec",
    "StepResult",
    "TokenBatch",
    "Transition",
    "collect_episode",
]
