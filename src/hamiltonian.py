from qiskit.quantum_info import Pauli, SparsePauliOp

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
    H_cost = 0
    N = len(d)
    for i in range(M):
        for j in range(N):
            for k in range(N):
                if d[j, k] == 0:
                    continue
                ind1 = indx(i, j)
                ind2 = indx(i + 1, k)
                H_cost += d(j, k) * x_op(ind1, N) @ x_op(ind2, N)
    return H_cost


def H_1(N, M):
    total = 0
    for i in range(M):
        for j in range(N):
            total -= x_op(indx(i, j))
        for j in range(N):
            for k in range(j + 2, N):
                total += 2 * x_op(indx(i, j)) @ x_op(indx(i, k))
    return total


def H_2(N, M):
    total = 0
    for i in range(N):
        for j in range(M):
            total -= x_op(indx(i, j))
        for j in range(M):
            for k in range(j + 2, M):
                total += 2 * x_op(indx(i, j)) @ x_op(indx(k, j))
    return total


def H_3(t, N, M):
    total = 0
    for i in range(M):
        for j in range(N):
            for k in range(N):
                total += t(j, k) * x_op(indx(i, j)) @ x_op(indx(i + 1, k))
    return total


def H_4(N, M):
    total = 0
    for i in range(N):
        total += (
            x_op(indx(1, i))
            + x_op(indx(M, i))
            - 2 * x_op(indx(1, i)) @ x_op(indx(M, i))
        )
    return total


def H_5(deltah, N, M):
    total = 0
    for i in range(M):
        for j in range(N):
            for k in range(N):
                total += deltah(j, k) * x_op(indx(i, j)) @ x_op(indx(i + 1, k))
    return total


def H_total(lambdas, deltah, t, N, M, alpha=0.05):
    d = t + alpha * deltah
    H = [H_1(N, M), H_2(N, M), H_3(t, N, M), H_4(N, M), H_5(deltah, N, M)]
    total = H_cost(d, M)
    for i in range(len(H)):
        total += lambdas[i] * H[i]
    return total
