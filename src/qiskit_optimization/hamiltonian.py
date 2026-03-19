from qiskit.quantum_info import Pauli, SparsePauliOp
import numpy as np
import quantum as q

X = SparsePauliOp("X")
Y = SparsePauliOp("Y")
Z = SparsePauliOp("Z")
I = SparsePauliOp("I")


def x_op(i, n):
    """Return an operator (I-Z)/2 at specified qubit index i, with n qubits in total"""
    candidates = [I] * n
    candidates[i] = (I - Z) / 2
    res = candidates[0]
    for i in range(1, n):
        res = res ^ candidates[i]
    return res


def I_op(n):
    """Return an identity operator over n qubits"""
    return SparsePauliOp(["I" * n], [1])


def indx(i, j, N):
    return N * i + j


def H_cost(d, M):
    H_cost = None
    N = len(d)
    for i in range(M - 1):
        for j in range(N):
            for k in range(N):
                if d[j, k] == 0:
                    continue
                ind1 = indx(i, j, N)
                ind2 = indx(i + 1, k, N)
                term = d[j, k] * x_op(ind1, N * M) @ x_op(ind2, N * M)
                H_cost = term if H_cost is None else H_cost + term
    return H_cost


def H_1(N, M):
    total = None
    for i in range(M):
        for j in range(N):
            term = -x_op(indx(i, j, N), N * M)
            total = term if total is None else total + term
        for j in range(N):
            for k in range(j + 2, N):
                term = 2 * x_op(indx(i, j, N), N * M) @ x_op(indx(i, k, N), N * M)
                total = term if total is None else total + term
    return total


def H_2(N, M):
    total = None
    for i in range(N):
        for j in range(M):
            term = -x_op(indx(i, j, N), N * M)
            total = term if total is None else total + term
        for j in range(M):
            for k in range(j + 2, M):
                term = 2 * x_op(indx(i, j, N), N * M) @ x_op(indx(k, j, N), N * M)
                total = term if total is None else total + term
    return total


def H_3(t, N, M):
    total = None
    for i in range(M - 1):
        for j in range(N):
            for k in range(N):
                term = (
                    t[j, k]
                    * x_op(indx(i, j, N), N * M)
                    @ x_op(indx(i + 1, k, N), N * M)
                )
                total = term if total is None else total + term
    return total


def H_4(a, b, N, M):
    total = None
    for i in range(N):
        term = -x_op(indx(0, a, N), N * M) - x_op(indx(M - 1, b, N), N * M)
        total = term if total is None else total + term
    return total


def H_5(deltah, N, M):
    total = None
    for i in range(M - 1):
        for j in range(N):
            for k in range(N):
                term = (
                    deltah[j, k]
                    * x_op(indx(i, j, N), N * M)
                    @ x_op(indx(i + 1, k, N), N * M)
                )
                total = term if total is None else total + term
    return total


def H_total(lambdas, deltah, t, N, M, alpha=0.05):
    d = t + alpha * deltah
    H = [H_1(N, M), H_2(N, M), H_3(t, N, M), H_4(N, M), H_5(deltah, N, M)]
    total = H_cost(d, M)
    for i in range(len(H)):
        total += lambdas[i] * H[i]
    return total
