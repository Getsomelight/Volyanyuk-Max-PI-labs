import math
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy import integrate
from scipy import linalg
import pandas as pd

np.random.seed(1000)

def QR_dec(A):
    n, m = A.shape

    Q = np.empty((n, n))
    u = np.empty((n, n))

    u[:, 0] = A[:, 0]
    Q[:, 0] = u[:, 0] / np.linalg.norm(u[:, 0])

    for i in range(1, n):

        u[:, i] = A[:, i]
        for j in range(i):
            u[:, i] -= (A[:, i] @ Q[:, j]) * Q[:, j]

        Q[:, i] = u[:, i] / np.linalg.norm(u[:, i])

    R = np.zeros((n, m))
    for i in range(n):
        for j in range(i, m):
            R[i, j] = A[:, j] @ Q[:, i]

    return Q, R

def task_1():
    n = 3
    A = np.random.uniform(-8, 8, (n, n))
    print("A = ", A)
    A_partial = A[:, :2].copy()
    Q = np.zeros_like(A_partial, dtype=float)
    for i in range(A_partial.shape[1]):
        q = A_partial[:, i].astype(float)
        for j in range(i):
            q -= np.dot(Q[:, j], A_partial[:, i]) * Q[:, j]
        norm = np.linalg.norm(q)
        if norm > 1e-10:
            Q[:, i] = q / norm
        else:
            Q[:, i] = q

    print("Первые два ортонормированных столбца: ", Q)
    print("Проверка на ортонормированность: ", Q[:, 0] @ Q[:, 1])

    Q,R = QR_dec(A)
    print("Q = \n", Q)
    print("R = \n", R)

    Q1, R1 = np.linalg.qr(A)
    print("Q(via linalg) = \n", Q1)
    print("R(via linalg) = \n", R1)


def task_2():
    A = np.array([[8.2, -3.2, 14.2, 14.8], [5.6, -12, 15, -6.4], [5.7, 3.6, -12.4, -2.3], [6.8, 13.2, -6.3, -8.7]])
    print("A = ", A)
    B = np.array([-8.4, 4.5, 3.3, 14.3])
    print("B = ", B)
    Q, R = QR_dec(A)
    print("Q = \n", Q)
    print("R = \n", R)

    r = B.copy()
    c = np.zeros(4)

    for k in range(4):
        c[k] = np.dot(r, Q[:, k])
        r = r - c[k] * Q[:, k]

    X = np.zeros(4)
    for i in range(4 - 1, -1, -1):
        X[i] = (c[i] - np.dot(R[i, i + 1:], X[i + 1:])) / R[i, i]

    print("X = ", X)
    print("Check via np.solve", np.linalg.solve(A, B))


def check_diagonal_dominance(A):
    n = len(A)
    for i in range(n):
        diagonal = abs(A[i, i])
        row_sum = sum(abs(A[i, j]) for j in range(n) if j != i)
        if diagonal <= row_sum:
            return False
    return True


def rearrange_for_dominance(A, B):
    n = len(A)
    A_new = A.copy()
    B_new = B.copy()

    available_rows = list(range(n))

    for col in range(n):
        max_val = 0
        max_row = -1

        for row in available_rows:
            if abs(A[row, col]) > max_val:
                max_val = abs(A[row, col])
                max_row = row

        if max_row == -1:
            return A, B, False

        if max_row != col and max_row in available_rows:
            A_new[[col, max_row]] = A_new[[max_row, col]]
            B_new[[col, max_row]] = B_new[[max_row, col]]
            available_rows.remove(max_row)

    return A_new, B_new, check_diagonal_dominance(A_new)


def task_3():
    A = np.array([[7.5, 3.8, 4.8], [1.9, 4.1, 2.1], [3.1, 2.8, 4.9]])
    print("A = ", A)
    B = np.array([0.2, 2.1, 5.6])
    print("B = ", B)
    has_dominance = check_diagonal_dominance(A)

    if not has_dominance:
        print("\nИсходная матрица не имеет диагонального преобладания.")

        A, B, success = rearrange_for_dominance(A, B)

        if success:
            print("Удалось достичь диагонального преобладания перестановкой строк.")
        else:
            print("Не удалось достичь диагонального преобладания перестановкой строк.")
            for i in range(3):
                if abs(A[i, i]) <= sum(abs(A[i, j]) for j in range(3) if j != i):
                    for j in range(3):
                        if i != j and A[j, i] != 0:
                            factor = A[j, i] / A[i, i] if A[i, i] != 0 else 1
                            A[i] += factor * A[j]
                            B[i] += factor * B[j]
                            break

            if check_diagonal_dominance(A):
                print("Удалось достичь диагонального преобладания преобразованиями.")
            else:
                print("Не удалось достичь строгого диагонального преобладания.")
                print("Сходимость метода не гарантирована.")
    else:
        print("\nМатрица имеет диагональное преобладание.")

    n = len(B)
    x = np.zeros(n)
    tol = 1e-3
    max_iter = 100
    history = []
    for k in range(max_iter):
        x_new = np.copy(x)
        for i in range(n):
            s1 = np.dot(A[i, :i], x_new[:i])
            s2 = np.dot(A[i, i + 1:], x[i + 1:])
            x_new[i] = (B[i] - s1 - s2) / A[i, i]
        error = np.linalg.norm(x_new - x, np.inf)
        history.append([k + 1, *x_new, error])
        if error < tol:
            break
        if error > 1e+10:
            print("Метод не сходится")
            break
        x = x_new
    df = pd.DataFrame(history, columns = ["i", "x1", "x2", "x3", "err"])
    print(df)
    print(np.linalg.solve(A, B))

def f(x):
    return -1.38*x**3 - 5.42*x**2 + 2.57*x + 10.95

def df(x):
    return -4.14*x**2 - 10.84*x + 2.57


def bisection_method(f, a, b, tolerance = 1e-3, max_iterations = 100):
    if f(a) * f(b) >= 0:
        print(f"На интервале [{a}, {b}] не выполняется условие f(a)*f(b) < 0")
        return None

    print(f"Интервал: [{a}, {b}]")
    print(f"f({a}) = {f(a):.6f}, f({b}) = {f(b):.6f}")
    print("Итерационный процесс:")
    print("k\t a\t\t b\t\t c\t\t f(c)")
    print("-" * 70)

    for k in range(max_iterations):
        c = (a + b) / 2
        fc = f(c)

        print(f"{k}\t {a:.6f}\t {b:.6f}\t {c:.6f}\t {fc:.6f}")

        if abs(fc) < tolerance or (b - a) / 2 < tolerance:
            print(f"\nРешение найдено за {k} итераций")
            print(f"x = {c:.6f}, f(x) = {fc:.6f}")
            return c

        if f(a) * fc < 0:
            b = c
        else:
            a = c

    print(f"Достигнуто максимальное число итераций ({max_iterations})")
    return (a + b) / 2


def combined_method(f, df, a, b, tolerance = 1e-5, max_iterations = 100):
    if f(a) * f(b) >= 0:
        print(f"На интервале [{a}, {b}] не выполняется условие f(a)*f(b) < 0")
        return None

    f2_a = -8.28 * a - 10.84  # f''(a)
    f2_b = -8.28 * b - 10.84  # f''(b)

    print(f"Интервал: [{a}, {b}]")
    print(f"f({a}) = {f(a):.8f}, f({b}) = {f(b):.8f}")
    print(f"f''({a}) = {f2_a:.2f}, f''({b}) = {f2_b:.2f}")
    print("Итерационный процесс:")
    print("k\t x_хорд\t\t x_кас\t\t разность")
    print("-" * 60)

    x_chord = a
    x_tangent = b

    for k in range(max_iterations):
        x_chord_new = a - f(a) * (b - a) / (f(b) - f(a))

        x_tangent_new = x_tangent - f(x_tangent) / df(x_tangent)

        difference = abs(x_chord_new - x_tangent_new)

        print(f"{k}\t {x_chord_new:.8f}\t {x_tangent_new:.8f}\t {difference:.2e}")

        if difference < tolerance:
            root = (x_chord_new + x_tangent_new) / 2
            print(f"\nРешение найдено за {k} итераций")
            print(f"x = {root:.8f}, f(x) = {f(root):.2e}")
            return root

        x_chord = x_chord_new
        x_tangent = x_tangent_new

        if f(a) * f(x_chord_new) < 0:
            b = x_chord_new
        else:
            a = x_chord_new

    print(f"Достигнуто максимальное число итераций ({max_iterations})")
    return (x_chord + x_tangent) / 2

def task_4():
    x = np.linspace(-5, 5, 1000)
    y = f(x)

    plt.figure(figsize=(12, 6))
    plt.plot(x, y, 'b-', linewidth=2, label='f(x) = -1.38x³ - 5.42x² + 2.57x + 10.95')
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.7)
    plt.grid(True, alpha=0.3)
    plt.xlabel('x')
    plt.ylabel('f(x)')

    sign_changes = []
    for i in range(len(x) - 1):
        if y[i] * y[i + 1] <= 0:
            sign_changes.append((x[i], x[i + 1]))
            plt.axvline(x=x[i], color='r', linestyle='--', alpha=0.5)
            plt.axvline(x=x[i + 1], color='r', linestyle='--', alpha=0.5)

    plt.show()

    intervals = [(-4, -3), (-2, -1), (1, 2)]

    bisection_roots = []
    for i, (a, b) in enumerate(intervals):
        print(f"\nКорень {i + 1}:")
        root = bisection_method(f, a, b, tolerance=1e-3)
        if root is not None:
            bisection_roots.append(root)

    combined_roots = []
    for i, (a, b) in enumerate(intervals):
        print(f"\nКорень {i + 1}:")
        root = combined_method(f, df, a, b, tolerance=1e-5)
        if root is not None:
            combined_roots.append(root)

def newton_system(F, J, x0, tol = 1e-4, max_iter = 100):
    x = x0
    for i in range(max_iter):
        F_val = F(x)
        J_val = J(x)
        delta = np.linalg.solve(J_val, -F_val)
        x += delta
        if np.linalg.norm(delta) < tol:
            return x, i+1
    return x, max_iter

def F(v):
    x, y = v
    return np.array(
        [
        np.sin(x) + 2 * y - 2,
        2 * x + np.cos(y - 1) - 0.7
        ]
    )

def J(v):
    x, y = v
    return np.array(
        [
            [np.cos(x), 2],
            [2, -np.sin(y-1)]
        ]
    )

def task_5():
    x0 = np.array([0.0, 1.0])
    sol, iters = newton_system(F, J, x0)
    print(f"Решение системы: x = {sol[0]:.5f}, y = {sol[1]:.5f}, итераций: {iters}")
    print("Проверка F(x) = 0:\n", F(sol))


def main():
    print("Задание 1: ")
    task_1()
    print("\n\n\nЗадание 2: ")
    task_2()
    print("\n\n\nЗадание 3: ")
    task_3()
    print("\n\n\nЗадание 4: ")
    task_4()
    print("\n\n\nЗадание 5: ")
    task_5()


if __name__ == "__main__":
    main()