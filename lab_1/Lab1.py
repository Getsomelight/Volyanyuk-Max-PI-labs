import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

np.random.seed(1000)


def qr_decompose(matrix_a):
    """
    Performs QR decomposition of a matrix using Gram-Schmidt process

    :param matrix_a: original matrix
    :return: orthogonal matrix Q and upper triangular matrix R
    """
    n, m = matrix_a.shape

    q = np.empty((n, n))
    u = np.empty((n, n))

    u[:, 0] = matrix_a[:, 0]
    q[:, 0] = u[:, 0] / np.linalg.norm(u[:, 0])

    for i in range(1, n):

        u[:, i] = matrix_a[:, i]
        for j in range(i):
            u[:, i] -= (matrix_a[:, i]@q[:, j]) * q[:, j]

        q[:, i] = u[:, i] / np.linalg.norm(u[:, i])

    r = np.zeros((n, m))
    for i in range(n):
        for j in range(i, m):
            r[i, j] = matrix_a[:, j] @ q[:, i]

    return q, r


def task_1() -> None:
    """
    Demonstrates QR decomposition for a random 3x3 matrix
    """
    n = 3
    matrix_a = np.random.uniform(-8, 8, (n, n))
    print("A = ", matrix_a)
    a_partial = matrix_a[:, :2].copy()
    q = np.zeros_like(a_partial, dtype=float)
    for i in range(a_partial.shape[1]):
        q_temp = a_partial[:, i].astype(float)
        for j in range(i):
            q_temp -= np.dot(q[:, j], a_partial[:, i]) * q[:, j]
        norm = np.linalg.norm(q)
        if norm > 1e-10:
            q[:, i] = q_temp / norm
        else:
            q[:, i] = q_temp

    print("Первые два ортонормированных столбца: ", q)
    print("Проверка на ортонормированность: ", q[:, 0] @ q[:, 1])

    q, r = qr_decompose(matrix_a)
    print("Q = \n", q)
    print("R = \n", r)

    q1, r1 = np.linalg.qr(matrix_a)
    print("Q(via linalg) = \n", q1)
    print("R(via linalg) = \n", r1)


def task_2() -> None:
    """
    Solves system of linear equations using QR decomposition
    """
    matrix_a = np.array(
        [
            [8.2, -3.2, 14.2, 14.8],
            [5.6, -12, 15, -6.4],
            [5.7, 3.6, -12.4, -2.3],
            [6.8, 13.2, -6.3, -8.7]
        ]
    )
    print("A = ", matrix_a)
    matrix_b = np.array([-8.4, 4.5, 3.3, 14.3])
    print("B = ", matrix_b)
    q, r = qr_decompose(matrix_a)
    print("Q = \n", q)
    print("R = \n", r)

    r_temp = matrix_b.copy()
    c = np.zeros(4)

    for k in range(4):
        c[k] = np.dot(r_temp, q[:, k])
        r_temp = r_temp - c[k]*q[:, k]

    x = np.zeros(4)
    for i in range(4 - 1, -1, -1):
        x[i] = (c[i]-np.dot(r[i, i + 1:], x[i + 1:])) / r[i, i]

    print("X = ", x)
    print("Check via np.solve", np.linalg.solve(matrix_a, matrix_b))


def check_diagonal_dominance(matrix_a):
    """
    Checks if matrix has strict diagonal dominance

    :param matrix_a: matrix to check
    :return: True if matrix has strict diagonal dominance
    """
    n = len(matrix_a)
    for i in range(n):
        diagonal = abs(matrix_a[i, i])
        row_sum = sum(abs(matrix_a[i, j]) for j in range(n) if j != i)
        if diagonal <= row_sum:
            return False
    return True


def rearrange_for_dominance(matrix_a, matrix_b):
    """
    Attempts to achieve diagonal dominance by row permutation

    :param matrix_a: coefficient matrix
    :param matrix_b: right-hand side vector
    :return: modified matrix and vector, success flag
    """
    n = len(matrix_a)
    a_new = matrix_a.copy()
    b_new = matrix_b.copy()

    available_rows = list(range(n))

    for col in range(n):
        max_val = 0
        max_row = -1

        for row in available_rows:
            if abs(matrix_a[row, col]) > max_val:
                max_val = abs(matrix_a[row, col])
                max_row = row

        if max_row == -1:
            return matrix_a, matrix_b, False

        if max_row != col and max_row in available_rows:
            a_new[[col, max_row]] = a_new[[max_row, col]]
            b_new[[col, max_row]] = b_new[[max_row, col]]
            available_rows.remove(max_row)

    return a_new, b_new, check_diagonal_dominance(a_new)


def task_3() -> None:
    """
    Solves system of linear equations using Jacobi iterative method
    """
    matrix_a = np.array(
        [
            [7.5, 3.8, 4.8],
            [1.9, 4.1, 2.1],
            [3.1, 2.8, 4.9]
        ]
    )
    print("A = ", matrix_a)
    matrix_b = np.array([0.2, 2.1, 5.6])
    print("B = ", matrix_b)
    has_dominance = check_diagonal_dominance(matrix_a)

    if not has_dominance:
        print("\nИсходная матрица не имеет диагонального преобладания.")

        matrix_a, matrix_b, success = rearrange_for_dominance(matrix_a, matrix_b)

        if success:
            print("Удалось достичь диагонального преобладания перестановкой строк.")
        else:
            print("Не удалось достичь диагонального преобладания перестановкой строк.")
            for i in range(3):
                if abs(matrix_a[i, i]) <= sum(abs(matrix_a[i, j]) for j in range(3) if j != i):
                    for j in range(3):
                        if i != j and matrix_a[j, i] != 0:
                            factor = matrix_a[j, i] / matrix_a[i, i] if matrix_a[i, i] != 0 else 1
                            matrix_a[i] += factor * matrix_a[j]
                            matrix_b[i] += factor * matrix_b[j]
                            break

            if check_diagonal_dominance(matrix_a):
                print("Удалось достичь диагонального преобладания преобразованиями.")
            else:
                print("Не удалось достичь строгого диагонального преобладания.")
                print("Сходимость метода не гарантирована.")
    else:
        print("\nМатрица имеет диагональное преобладание.")

    n = len(matrix_b)
    x = np.zeros(n)
    tol = 1e-3
    max_iter = 100
    history = []
    for k in range(max_iter):
        x_new = np.copy(x)
        for i in range(n):
            s1 = np.dot(matrix_a[i, :i], x_new[:i])
            s2 = np.dot(matrix_a[i, i + 1:], x[i + 1:])
            x_new[i] = (matrix_b[i]-s1-s2) / matrix_a[i, i]
        error = np.linalg.norm(x_new - x, np.inf)
        history.append([k + 1, *x_new, error])
        if error < tol:
            break
        if error > 1e+10:
            print("Метод не сходится")
            break
        x = x_new
    data_frame = pd.DataFrame(
        history,
        columns=["i", "x1", "x2", "x3", "err"]
    )
    print(data_frame)
    print(np.linalg.solve(matrix_a, matrix_b))


def f(x):
    """
    Original equation y = f(x)
    """
    return -1.38 * x**3 - 5.42 * x**2 + 2.57 * x + 10.95


def df(x):
    """
    Derivative of the equation y = f`(x)
    """
    return -4.14 * x**2 - 10.84 * x + 2.57


def bisection_method(
        function, left_bound, right_bound,
        tolerance=1e-3,
        max_iterations=100):
    """
    Implements bisection method for finding equation roots

    :param function: function to find root of
    :param left_bound: left interval boundary
    :param right_bound: right interval boundary
    :param tolerance: acceptable error margin
    :param max_iterations: maximum number of iterations
    :return: found root or None
    """
    if function(left_bound) * function(right_bound) >= 0:
        print(
            f"На интервале [{left_bound}, {right_bound}] \
            не выполняется условие f(left_bound)*f(right_bound) < 0"
        )
        return None

    print(f"Интервал: [{left_bound}, {right_bound}]")
    print(
        f"f({left_bound}) = {function(left_bound):.6f}, \
        f({right_bound}) = {function(right_bound):.6f}"
    )
    print("Итерационный процесс:")
    print("k\t left_bound\t\t right_bound\t\t c\t\t f(c)")
    print("-" * 70)

    for k in range(max_iterations):
        c = (left_bound+right_bound) / 2
        fc = function(c)

        print(f"{k}\t {left_bound:.6f}\t {right_bound:.6f}\t {c:.6f}\t {fc:.6f}")

        if abs(fc) < tolerance or (right_bound-left_bound) / 2 < tolerance:
            print(f"\nРешение найдено за {k} итераций")
            print(f"x = {c:.6f}, f(x) = {fc:.6f}")
            return c

        if function(left_bound) * fc < 0:
            right_bound = c
        else:
            left_bound = c

    print(f"Достигнуто максимальное число итераций ({max_iterations})")
    return (left_bound+right_bound) / 2


def combined_method(
        function, derived_function, left_bound, right_bound,
        tolerance=1e-5,
        max_iterations=100):
    """
    Combined method of chords and tangents for solving equations

    :param function: original function
    :param derived_function: derivative of function
    :param left_bound: left interval boundary
    :param right_bound: right interval boundary
    :param tolerance: acceptable error margin
    :param max_iterations: maximum number of iterations
    :return: found root or None
    """
    if function(left_bound) * function(right_bound) >= 0:
        print(
            f"На интервале [{left_bound}, {right_bound}] \
            не выполняется условие f(left_bound)*f(right_bound) < 0"
        )
        return None

    f2_left_bound = -8.28*left_bound - 10.84  # f''(left_bound)
    f2_right_bound = -8.28*right_bound - 10.84  # f''(right_bound)

    print(f"Интервал: [{left_bound}, {right_bound}]")
    print(
        f"f({left_bound}) = {function(left_bound):.8f}, \
        f({right_bound}) = {function(right_bound):.8f}"
    )
    print(f"f''({left_bound}) = {f2_left_bound:.2f}, f''({right_bound}) = {f2_right_bound:.2f}")
    print("Итерационный процесс:")
    print("k\t x_хорд\t\t x_кас\t\t разность")
    print("-" * 60)

    x_chord = left_bound
    x_tangent = right_bound

    for k in range(max_iterations):
        x_chord_new = (left_bound
                       - function(left_bound)
                       * (right_bound-left_bound)
                       / (function(right_bound)-function(left_bound)))

        x_tangent_new = x_tangent - function(x_tangent)/derived_function(x_tangent)

        difference = abs(x_chord_new - x_tangent_new)

        print(f"{k}\t {x_chord_new:.8f}\t {x_tangent_new:.8f}\t {difference:.2e}")

        if difference < tolerance:
            root = (x_chord_new + x_tangent_new) / 2
            print(f"\nРешение найдено за {k} итераций")
            print(f"x = {root:.8f}, f(x) = {f(root):.2e}")
            return root

        x_chord = x_chord_new
        x_tangent = x_tangent_new

        if function(left_bound) * function(x_chord_new) < 0:
            right_bound = x_chord_new
        else:
            left_bound = x_chord_new

    print(f"Достигнуто максимальное число итераций ({max_iterations})")
    return (x_chord+x_tangent) / 2


def task_4() -> None:
    """
    Finds roots of nonlinear equation using various methods
    """
    x = np.linspace(-5, 5, 1000)
    y = f(x)

    plt.figure(figsize=(12, 6))
    plt.plot(
        x,
        y,
        'b-',
        linewidth=2,
        label='f(x) = -1.38x³ - 5.42x² + 2.57x + 10.95'
    )
    plt.axhline(
        y=0,
        color='k',
        linestyle='--',
        alpha=0.7
    )
    plt.grid(True, alpha=0.3)
    plt.xlabel('x')
    plt.ylabel('f(x)')

    sign_changes = []
    for i in range(len(x) - 1):
        if y[i] * y[i + 1] <= 0:
            sign_changes.append((x[i], x[i + 1]))
            plt.axvline(
                x=x[i],
                color='r',
                linestyle='--',
                alpha=0.5
            )
            plt.axvline(
                x=x[i + 1],
                color='r',
                linestyle='--',
                alpha=0.5
            )

    plt.show()

    intervals = [(-4, -3), (-2, -1), (1, 2)]

    bisection_roots = []
    for i, (left_bound, right_bound) in enumerate(intervals):
        print(f"\nКорень {i + 1}:")
        root = bisection_method(
            f, left_bound, right_bound,
            tolerance=1e-3)
        if root is not None:
            bisection_roots.append(root)

    combined_roots = []
    for i, (left_bound, right_bound) in enumerate(intervals):
        print(f"\nКорень {i + 1}:")
        root = combined_method(
            f, df, left_bound, right_bound,
            tolerance=1e-5)
        if root is not None:
            combined_roots.append(root)


def newton_system(
        sys_equat, jac, x0,
        tol=1e-4,
        max_iter=100):
    """
    Implements Newton's method for solving systems of nonlinear equations

    :param sys_equat: system of equations
    :param jac: Jacobian matrix
    :param x0: initial guess
    :param tol: acceptable error margin
    :param max_iter: maximum number of iterations
    :return: solution and iteration count
    """
    x = x0
    for i in range(max_iter):
        sys_equat_val = sys_equat(x)
        jac_val = jac(x)
        delta = np.linalg.solve(jac_val, -sys_equat_val)
        x += delta
        if np.linalg.norm(delta) < tol:
            return x, i + 1
    return x, max_iter


def system_equations(v):
    """
    System of nonlinear equations F(x,y) = 0
    v: [x, y] - coordinates of the system
    """
    var_x, var_y = v
    return np.array(
        [
            np.sin(var_x) + 2*var_y - 2,
            2*var_x + np.cos(var_y - 1) - 0.7
        ]
    )


def jacobian_system(v):
    """
    Jacobian matrix for the system of equations
    v: [x, y] - coordinates of the system
    """
    var_x, var_y = v
    return np.array(
        [
            [np.cos(var_x), 2],
            [2, -np.sin(var_y - 1)]
        ]
    )


def task_5() -> None:
    """
    Solving a system of nonlinear equations by the Newton method
    """
    x0 = np.array([0.0, 1.0])
    sol, iters = newton_system(system_equations, jacobian_system, x0)
    print(
        f"Решение системы: x = {sol[0]:.5f}, \
        y = {sol[1]:.5f}, итераций: {iters}"
    )
    print("Проверка F(x) = 0:\n", system_equations(sol))


def main():
    """
    Main function that runs all tasks
    """
    print("Задание 1: QR-разложение")
    task_1()
    print("\n\n\nЗадание 2: Решение СЛАУ через QR-разложение")
    task_2()
    print("\n\n\nЗадание 3: Метод простых итераций для СЛАУ")
    task_3()
    print("\n\n\nЗадание 4: Решение нелинейного уравнения")
    task_4()
    print("\n\n\nЗадание 5: Решение системы нелинейных уравнений")
    task_5()


if __name__ == "__main__":
    main()
