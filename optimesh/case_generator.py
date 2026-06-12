import numpy as np

import element as el


def generate_random_case(P: int = 10, seed=None) -> el.Element:
    """
    Builds an Element with a randomized shock: random left/right values,
    shock position and shock steepness. Used as a test case for the
    mesh-clustering / RBF-shape optimization.
    """
    rng = np.random.default_rng(seed)

    Us = rng.normal(5,10), rng.normal(0,5)

    shock_pos = rng.uniform(-0.7, 0.7)
    shock_intensity = rng.uniform(10, 100)

    return el.Element(P=P, uL=Us[1], uR=Us[0], shock_pos=shock_pos, shock_intensity=shock_intensity)
