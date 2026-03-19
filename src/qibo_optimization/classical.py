import numpy as np
from qibo.hamiltonians import SymbolicHamiltonian
from qibo.symbols import I, X, Y, Z


def x_op(q):
    """Operador (I - Z)/2 sobre el qubit q."""
    return (1 - Z(q)) / 2


def I_op(n):
    """Identidad sobre n qubits."""
    op = 1
    for q in range(n):
        op *= I(q)
    return op


def indx(i, j, N):
    """Mapeo (i, j) -> índice lineal."""
    return N * i + j


def H_cost(d, N, M):
    total = 0
    for i in range(M - 1):
        for j in range(N):
            for k in range(N):
                if d[j, k] == 0:
                    continue
                ind1 = indx(i, j, N)
                ind2 = indx(i + 1, k, N)
                total += d[j, k] * x_op(ind1) * x_op(ind2)
    return total


def H_1(N, M):
    total = 0
    for i in range(M):
        for j in range(N):
            total -= x_op(indx(i, j, N))
        for j in range(N):
            for k in range(j + 1, N):
                total += 2 * x_op(indx(i, j, N)) * x_op(indx(i, k, N))
    return total


def H_2(N, M):
    total = 0
    for j in range(N):
        for i in range(M):
            total -= x_op(indx(i, j, N))
        for i in range(M):
            for k in range(i + 1, M):
                total += 2 * x_op(indx(i, j, N)) * x_op(indx(k, j, N))
    return total


def H_3(t, N, M):
    total = 0
    for i in range(M - 1):
        for j in range(N):
            for k in range(N):
                total += t[j, k] * x_op(indx(i, j, N)) * x_op(indx(i + 1, k, N))
    return total


def H_4(N, M):
    total = 0
    for j in range(N):
        q1 = indx(0, j, N)
        q2 = indx(M - 1, j, N)
        total += x_op(q1) + x_op(q2) - 2 * x_op(q1) * x_op(q2)
    return total


def H_5(deltah, N, M):
    total = 0
    for i in range(M - 1):
        for j in range(N):
            for k in range(N):
                total += deltah[j, k] * x_op(indx(i, j, N)) * x_op(indx(i + 1, k, N))
    return total


def H_total(lambdas, deltah, t, N, M, alpha=0.05, as_symbolic_hamiltonian=True):
    d = t + alpha * deltah

    terms = [
        H_1(N, M),
        H_2(N, M),
        H_3(t, N, M),
        H_4(N, M),
        H_5(deltah, N, M),
    ]

    total = H_cost(d, N, M)
    for i, term in enumerate(terms):
        total += lambdas[i] * term

    if as_symbolic_hamiltonian:
        return SymbolicHamiltonian(total)
    return total