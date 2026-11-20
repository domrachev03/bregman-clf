"""cvxlbs: Bregman control Lyapunov functions as rows of an inverse-dynamics QP.

A `Controller` assembles a QP over the joint acceleration from `Certificate`s (constraint rows and
cost terms written as CasADi expressions of the state) and solves it at every control step.
"""

from .controller import Controller, IDController, StepInfo
from .params import Params
from .problem import Problem, QPData

__all__ = ["Controller", "IDController", "Params", "Problem", "QPData", "StepInfo"]
