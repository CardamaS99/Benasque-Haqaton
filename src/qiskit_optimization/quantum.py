from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import Parameter


def create_qaoa_circ(theta, J, h, p=1):
    nqubits = len(h)
    qc = QuantumCircuit(nqubits)

    beta = theta[:p]
    gamma = theta[p:]

    # estado inicial
    for i in range(0, nqubits):
        qc.h(i)

    for irep in range(0, p):

        # Hamiltoniano problema
        for i in range(nqubits):
            for j in range(nqubits):
                qc.rzz(2 * J[i, j] * gamma[irep], i, j)

        for i in range(nqubits):
            qc.rz(2 * h(i) * gamma[irep], i)
        # Hamiltoniano mestura
        for i in range(0, nqubits):
            qc.rx(2 * beta[irep], i)

    qc.measure_all()

    return qc


def ising_value(J, h, x):
    value = 0
    nqubits = len(h)
    for i in range(nqubits):
        for j in range(nqubits):
            value -= J[i, j] * x[i] * x[j]

    for i in range(nqubits):
        value -= h[i] * x[i]

    return value


def compute_expectation(counts, J, h):
    avg = 0
    sum_count = 0

    for bitstring, count in counts.items():

        obj = ising_value(J, h, bitstring)
        avg += obj * count
        sum_count += count

    return avg / sum_count


def get_expectation(backend, theta, J, h, shots=512):

    qc = create_qaoa_circ(theta)
    counts = backend.run(qc, nshots=shots).result().get_counts()
    return compute_expectation(counts, J, h)
