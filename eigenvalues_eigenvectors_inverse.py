import math

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


def jacobi_eigen(matrix, max_iter=200, tol=1e-10):
    n = len(matrix)
    if any(len(row) != n for row in matrix):
        raise ValueError("Matrix must be square.")
    if not is_symmetric(matrix):
        raise ValueError("Jacobi method requires a symmetric matrix.")

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

    return eigenvalues, eigenvectors


def matrix_inverse_from_eigen(eigenvalues, eigenvectors, tol=1e-9):
    n = len(eigenvalues)
    inv_matrix = [[0.0] * n for _ in range(n)]

    for val, vec in zip(eigenvalues, eigenvectors):
        if abs(val) < tol:
            raise ValueError(f"Matrix is singular or near-singular (eigenvalue {val} ≈ 0). Inversion impossible.")

        # Outer product: v_i * v_i^T scaled by 1 / lambda_i
        inv_val = 1.0 / val
        for r in range(n):
            for c in range(n):
                inv_matrix[r][c] += inv_val * vec[r] * vec[c]

    return inv_matrix


def mat_mul(a, b):
    """Multiplies two square matrices: A * B."""
    n = len(a)
    return [[sum(a[i][k] * b[k][j] for k in range(n)) for j in range(n)] for i in range(n)]


if __name__ == "__main__":
    # Symmetric, invertible matrix
    A = [
        [4.0, 1.0, 1.0],
        [1.0, 3.0, 0.0],
        [1.0, 0.0, 2.0]
    ]

    # 1. Compute Eigenvalues & Eigenvectors
    vals, vecs = jacobi_eigen(A)

    print("--- Eigenvalues (λ) ---")
    print([round(x, 4) for x in vals])

    print("\n--- Eigenvectors (v) ---")
    for i, vec in enumerate(vecs, start=1):
        print(f"v_{i}: {[round(x, 4) for x in vec]}")

    # 2. Compute Inverse using A^(-1) = V * Λ^(-1) * V^T
    A_inv = matrix_inverse_from_eigen(vals, vecs)

    print("\n--- Matrix Inverse (A^(-1)) ---")
    for row in A_inv:
        print([round(val, 4) for val in row])

    # 3. Verification: A * A^(-1) ≈ Identity Matrix
    product = mat_mul(A, A_inv)
    print("\n--- Verification: A * A^(-1) (Should be Identity Matrix) ---")
    for row in product:
        print([round(val, 4) for val in row])
