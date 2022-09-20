# Gelin Eguinosa Rosique
# 2022

import numpy as np
from numpy.linalg import norm


def cos_sim(a: np.ndarray, b: np.ndarray):
    """
    Calculate the cosine similarity between the vectors 'a' and 'b'.

    Args:
        a: Numpy.ndarray containing one of the vectors embeddings.
        b: Numpy.ndarray containing one of the vectors embeddings.

    Returns:
        Float with the cosine similarity between the two vectors.
    """
    # Use Numpy.
    result = np.dot(a, b) / (norm(a) * norm(b))
    # Transform from float32 to float (float32 is not JSON serializable)
    result = float(result)
    return result
