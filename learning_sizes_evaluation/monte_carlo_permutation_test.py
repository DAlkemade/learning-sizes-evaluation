import numpy as np

R = 5000  # Number of permutations


def single_permute(results_a, results_b):
    """
    Perform a single permutation test on two result vectors.
    """
    vector_length = len(results_a)
    swap_vector = np.random.randint(0, 2, vector_length)
    permuted_a = results_a.copy()
    permuted_b = results_b.copy()
    for i in range(vector_length):
        if swap_vector[i] == 1:
            permuted_a[i] = results_b[i]
            permuted_b[i] = results_a[i]
    original_difference = abs(np.mean(results_a) - np.mean(results_b))
    permuted_difference = abs(np.mean(permuted_a) - np.mean(permuted_b))
    return permuted_difference >= original_difference


def permutation_test(results_a, results_b):
    """
    Perform a single permutation test on two result vectors. It returns the probability that the null hypothesis is true
    """
    s = 0
    for i in range(R):
        if single_permute(results_a, results_b):
            s += 1
    p = (s + 1) / (R + 1)
    return p


if __name__ == '__main__':
    a = np.ones((10, 1))
    b = np.zeros((10, 1))
    print(permutation_test(a, b))
