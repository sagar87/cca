import numpy as np
import scipy.spatial as spatial
from scipy.optimize import linear_sum_assignment


def match_vectors(XA, XB, fill_array=False):
    dist_matrix = spatial.distance.cdist(XA, XB, inverted_cosine_similarity)
    ridx, cidx = linear_sum_assignment(dist_matrix)
    distances = dist_matrix[ridx, cidx]
    if fill_array:
        cidx = np.append(
            cidx, np.array([i for i in range(XB.shape[0]) if i not in cidx])
        )
    return ridx, cidx, distances


def inverted_cosine_similarity(vector1, vector2):
    """
    When vectors point in the same direction, cosine similarity is 1;
    when vectors are perpendicular, cosine similarity is 0;
    and when vectors point in opposite directions, cosine similarity is -1.
    """
    abs_cosine_similarity = np.abs(1 - spatial.distance.cosine(vector1, vector2))
    return np.abs(abs_cosine_similarity - 1)


def dist_inv_cos(XA, XB):
    return spatial.distance.cdist(XA, XB, inverted_cosine_similarity)
