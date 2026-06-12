import numpy as np

import element as el


def generate_random_case(P: int = 10, seed=None) -> el.Element:
    """
    Builds an Element with a randomized shock: random left/right values,
    shock position and shock steepness. Used as a test case for the
    mesh-clustering / RBF-shape optimization.
    """
    rng = np.random.default_rng(seed)

    uL, uR = rng.uniform(-5, 5, size=2)
    shock_pos = rng.uniform(-0.7, 0.7)
    shock_intensity = rng.uniform(10, 100)

    return el.Element(P=P, uL=uL, uR=uR, shock_pos=shock_pos, shock_intensity=shock_intensity)
