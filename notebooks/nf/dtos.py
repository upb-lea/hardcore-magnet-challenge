from dataclasses import dataclass


@dataclass
class MaterialParameters:
    """
    Material parameters according to dissertation erika stenglein
    This dataclass type contains the frequency independent material parameters k and gamma.
    """
    param_k: float
    param_gamma: float
    frequency_max: float
    b_sat: float

@dataclass
class Polynomials:
    """
    Polynomial parameters according to dissertation erika stenglein
    The parameters are used to fit the W_hyst curve.
    """
    a_1: float
    a_2: float
    a_3: float
    a_4: float
    b_1: float
    b_2: float
    beta: float