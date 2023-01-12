import numpy as np


def rbo_cumulative_weight(d: int, p: float) -> float:
    """Calculate the sum of weights of ranks 1 to `d` in RBO.

    Eq. 21 in the original paper.

    Parameters
    ----------
    d : int
        Rank position.
    p : float
        RBO persistence.

    Returns
    -------
    float
        The sum of weights.

    Examples
    --------
    rbo_cumulative_weight(10, 0.9) -> 0.8555854467473518
    rbo_cumulative_weight(20, 0.95) -> 0.8534071700394501
    rbo_cumulative_weight(50, 0.98) -> 0.8522339103194004
    """
    assert d > 0
    assert 0.0 < p < 1.0
    weights = (
        1
        - p ** (d - 1)
        + ((1 - p) / p)
        * d
        * (np.log(1 / (1 - p)) - sum(p**i / i for i in range(1, d)))
    )
    return weights


def rbo_discrete_weight(d: int, p: float) -> float:
    """Calculate the weight of rank `d` in RBO.

    Eq. 19 in the original paper.

    Parameters
    ----------
    d : int
        Rank position.
    p : float
        RBO persistence.

    Returns
    -------
    float
        The weight.

    Examples
    --------
    rbo_discrete_weight(1, 0.9) -> 0.25584278811044947
    rbo_discrete_weight(1, 0.95) -> 0.15767011966073646
    rbo_discrete_weight(1, 0.98) -> 0.07983720419241119
    """
    assert d > 0
    assert 0.0 < p < 1.0
    weight = ((1 - p) / p) * (
        np.log(1 / (1 - p)) - sum(p**i / i for i in range(1, d))
    )
    return weight
