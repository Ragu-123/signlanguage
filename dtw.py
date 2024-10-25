from numpy import array, zeros, full, argmin, inf, isinf
from scipy.spatial.distance import euclidean
from sklearn.metrics.pairwise import manhattan_distances, euclidean_distances
from nltk.metrics.distance import edit_distance
import numpy as np

def dtw(x, y, dist_func=euclidean, warp=1, w=inf, s=1.0):
    """
    Computes Dynamic Time Warping (DTW) for sequences of 2D or 3D keypoints.

    :param array x: N1*M array of keypoints (N1 frames, M dimensions per keypoint)
    :param array y: N2*M array of keypoints (N2 frames, M dimensions per keypoint)
    :param func dist_func: Distance function used to calculate cost (default: Euclidean)
    :param int warp: Number of shifts allowed for the alignment path.
    :param int w: Window size to limit maximal distance between indices of matched entries |i, j|.
    :param float s: Weight applied on off-diagonal moves of the path. Higher values bias the path towards the diagonal.
    :return: Minimum distance, cost matrix, accumulated cost matrix, and wrap path.
    """
    assert len(x) and len(y)
    assert isinf(w) or (w >= abs(len(x) - len(y)))
    assert s > 0

    r, c = len(x), len(y)
    D0 = full((r + 1, c + 1), inf)
    D0[0, 0] = 0
    C = zeros((r, c))

    for i in range(r):
        for j in range(c):
            if isinf(w) or (max(0, i - w) <= j <= min(c, i + w)):
                C[i, j] = dist_func([x[i]], [y[j]])[0, 0] if callable(dist_func) else dist_func(x[i], y[j])
                D0[i + 1, j + 1] = C[i, j]

    D1 = D0[1:, 1:]
    for i in range(r):
        for j in range(c):
            min_list = [D0[i, j]]
            for k in range(1, warp + 1):
                if i + k < r + 1:
                    min_list.append(D0[i + k, j] * s)
                if j + k < c + 1:
                    min_list.append(D0[i, j + k] * s)
            D1[i, j] += min(min_list)

    path = _traceback(D0)
    return D1[-1, -1], C, D1, path

def _traceback(D):
    """Traceback to find the optimal alignment path."""
    i, j = array(D.shape) - 2
    p, q = [i], [j]
    while i > 0 or j > 0:
        tb = argmin((D[i, j], D[i, j + 1], D[i + 1, j]))
        if tb == 0:
            i -= 1
            j -= 1
        elif tb == 1:
            i -= 1
        else:
            j -= 1
        p.insert(0, i)
        q.insert(0, j)
    return array(p), array(q)

# Testing and visualization
if __name__ == '__main__':
    w = inf
    s = 1.0
    if 1:  # 1-D numeric example with Manhattan distance
        x = [0, 0, 1, 1, 2, 4, 2, 1, 2, 0]
        y = [1, 1, 1, 2, 2, 2, 2, 3, 2, 0]
        dist_func = manhattan_distances
        w = 1
    elif 0:  # 2-D numeric example with Euclidean distance
        x = [[0, 0], [0, 1], [1, 1], [1, 2], [2, 2], [4, 3], [2, 3], [1, 1], [2, 2], [0, 1]]
        y = [[1, 0], [1, 1], [1, 1], [2, 1], [4, 3], [4, 3], [2, 3], [3, 1], [1, 2], [1, 0]]
        dist_func = euclidean_distances
    else:  # 1-D list of strings with edit distance
        x = ['i', 'soon', 'found', 'myself', 'muttering', 'to', 'the', 'walls']
        y = ['see', 'drown', 'himself']
        dist_func = edit_distance

    # Run DTW
    dist, cost, acc, path = dtw(x, y, dist_func, w=w, s=s)

    # Visualization (2D example only)
    if isinstance(cost, np.ndarray):
        import matplotlib.pyplot as plt
        plt.imshow(cost.T, origin='lower', cmap=plt.cm.Reds, interpolation='nearest')
        plt.plot(path[0], path[1], '-o')
        plt.xlabel('x sequence')
        plt.ylabel('y sequence')
        plt.title(f'Minimum distance: {dist}')
        plt.show()
