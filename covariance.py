def mean(values):
    """Calculate the arithmetic mean of a list of numbers."""
    if not values:
        raise ValueError("List cannot be empty.")
    return sum(values) / len(values)


def sample_covariance(x, y):
    """
    Calculate the sample covariance between two lists from scratch.
    Uses Bessel's correction (n - 1 degrees of freedom).
    """
    if len(x) != len(y):
        raise ValueError("Lists x and y must have the same length.")
    if len(x) < 2:
        raise ValueError("At least two data points are required for sample covariance.")

    n = len(x)
    mean_x = mean(x)
    mean_y = mean(y)

    cov = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n)) / (n - 1)
    return cov


def covariance_matrix(data):
    """
    Compute the covariance matrix for a 2D dataset.
    data: list of columns/features, e.g., [feature_1, feature_2, ...]
    """
    num_features = len(data)
    matrix = [[0.0] * num_features for _ in range(num_features)]

    for i in range(num_features):
        for j in range(num_features):
            matrix[i][j] = sample_covariance(data[i], data[j])

    return matrix


if __name__ == "__main__":
    # Sample input variables
    x = [10, 20, 30, 40, 50]
    y = [12, 24, 33, 45, 53]
    z = [50, 40, 31, 20, 11]

    # 1. Single pair covariance
    cov_xy = sample_covariance(x, y)
    cov_xz = sample_covariance(x, z)

    print(f"Sample Covariance (X, Y): {cov_xy:.2f}")
    print(f"Sample Covariance (X, Z): {cov_xz:.2f}")

    # 2. Covariance matrix for multi-variable data [X, Y, Z]
    dataset = [x, y, z]
    cov_mat = covariance_matrix(dataset)

    print("\nCovariance Matrix [[X, Y, Z]]:")
    for row in cov_mat:
        print([round(val, 2) for val in row])