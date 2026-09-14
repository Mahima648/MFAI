import math

def compute_mean(values):
    if not values:
        raise ValueError("Cannot calculate mean of an empty list.")
    return sum(values) / len(values)


def sample_covariance(x, y):
    if len(x) != len(y):
        raise ValueError("Feature lists must have the same length.")
    if len(x) < 2:
        raise ValueError("At least 2 data points required for sample covariance.")

    n = len(x)
    mean_x = compute_mean(x)
    mean_y = compute_mean(y)

    cov = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n)) / (n - 1)
    return cov


def covariance_matrix(data):
    p = len(data)
    cov_mat = [[0.0] * p for _ in range(p)]
    for i in range(p):
        for j in range(p):
            cov_mat[i][j] = sample_covariance(data[i], data[j])
    return cov_mat

def is_symmetric(matrix, tol=1e-8):
    n = len(matrix)
    for i in range(n):
        for j in range(i + 1, n):
            if abs(matrix[i][j] - matrix[j][i]) > tol:
                return False
    return True


def max_off_diagonal(matrix):
    n = len(matrix)
    max_val = 0.0
    p, q = 0, 1
    for i in range(n):
        for j in range(i + 1, n):
            if abs(matrix[i][j]) > max_val:
                max_val = abs(matrix[i][j])
                p, q = i, j
    return max_val, p, q


def compute_eigen(matrix, max_iter=200, tol=1e-10):
    n = len(matrix)
    if any(len(row) != n for row in matrix):
        raise ValueError("Matrix must be square.")
    if not is_symmetric(matrix):
        raise ValueError("Jacobi method requires a real symmetric matrix.")

    a = [row[:] for row in matrix]
    v = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]

    for _ in range(max_iter):
        max_val, p, q = max_off_diagonal(a)
        if max_val < tol:
            break

        app, aqq, apq = a[p][p], a[q][q], a[p][q]

        if abs(app - aqq) < 1e-15:
            theta = math.pi / 4.0 if apq > 0 else -math.pi / 4.0
        else:
            theta = 0.5 * math.atan2(2.0 * apq, aqq - app)

        c = math.cos(theta)
        s = math.sin(theta)

        new_app = c * c * app - 2.0 * s * c * apq + s * s * aqq
        new_aqq = s * s * app + 2.0 * s * c * apq + c * c * aqq
        a[p][q] = 0.0
        a[q][p] = 0.0
        a[p][p] = new_app
        a[q][q] = new_aqq

        for i in range(n):
            if i != p and i != q:
                a_ip = a[i][p]
                a_iq = a[i][q]
                a[i][p] = c * a_ip - s * a_iq
                a[p][i] = a[i][p]
                a[i][q] = s * a_ip + c * a_iq
                a[q][i] = a[i][q]
        for i in range(n):
            v_ip = v[i][p]
            v_iq = v[i][q]
            v[i][p] = c * v_ip - s * v_iq
            v[i][q] = s * v_ip + c * v_iq
    eigenvalues = [a[i][i] for i in range(n)]
    eigenvectors = [[v[i][col] for i in range(n)] for col in range(n)]

    paired = sorted(zip(eigenvalues, eigenvectors), key=lambda x: x[0], reverse=True)
    eigenvalues = [pair[0] for pair in paired]
    eigenvectors = [pair[1] for pair in paired]

    return eigenvalues, eigenvectors


def mat_vec_multiply(matrix, vector):
    return [sum(row[j] * vector[j] for j in range(len(vector))) for row in matrix]


def verify_eigen_identity(matrix, val, vec, tol=1e-6):
    av = mat_vec_multiply(matrix, vec)
    lv = [val * x for x in vec]
    return all(abs(av[i] - lv[i]) < tol for i in range(len(vec)))


if __name__ == "__main__":
    print("=" * 65)
    print("MATHEMATICAL FOUNDATIONS FOR AI: COVARIANCE & SPECTRAL ANALYSIS")
    print("=" * 65)

    feature_1 = [2.5, 0.5, 2.2, 1.9, 3.1]
    feature_2 = [2.4, 0.7, 2.9, 2.2, 3.0]
    feature_3 = [1.0, 0.2, 1.1, 0.9, 1.5]

    dataset = [feature_1, feature_2, feature_3]

    print("\n[1] Sample Covariance Matrix (A):")
    cov_mat = covariance_matrix(dataset)
    for row in cov_mat:
        print("  " + str([round(val, 4) for val in row]))

    print("\n[2] Spectral Decomposition (Jacobi Algorithm):")
    eigenvalues, eigenvectors = compute_eigen(cov_mat)

    for idx, (val, vec) in enumerate(zip(eigenvalues, eigenvectors), start=1):
        print(f"\nComponent {idx}:")
        print(f"  Eigenvalue  (λ_{idx}) : {val:.6f}")
        print(f"  Eigenvector (v_{idx}) : {[round(x, 6) for x in vec]}")

        # Verification
        passed = verify_eigen_identity(cov_mat, val, vec)
        print(f"  Check (A * v = λ * v) : {'PASSED (Valid Eigenpair)' if passed else 'FAILED'}")

    print("\n" + "=" * 65)