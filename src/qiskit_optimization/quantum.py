from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import Parameter


def get_J_h(H):
    nqubits = H.num_qubits
    J = [[0] * nqubits for _ in range(nqubits)]
    h = [0] * nqubits

    for pauli, coeff in zip(H.paulis, H.coeffs):
        pauli = str(pauli)
        if pauli.count('Z') == 0:
            continue
        elif pauli.count('Z') == 1:
            i = pauli.find("Z")
            h[i] += coeff
        elif pauli.count('Z') == 2:
            i = pauli.find("Z")
            j = pauli.find("Z", i + 1)
            J[i][j] += coeff
            

    return J, h


def create_qaoa_circ(J, h, p=1):
    nqubits = len(h)
    qc = QuantumCircuit(nqubits)

    # estado inicial
    for i in range(0, nqubits):
        qc.h(i)

    for irep in range(0, p):

        # Hamiltoniano problema
        param = Parameter(f"gamma{irep}")
        for i in range(nqubits):
            for j in range(nqubits):
                if i != j:
                    qc.rzz(2 * J[i][j] * param, i, j)
            qc.rz(2 * h[i] * param, i)
        # Hamiltoniano mestura
        param = Parameter(f"beta{irep}")
        for i in range(0, nqubits):
            qc.rx(2 * param, i)

    qc.measure_all()

    return qc


def ising_value(J, h, x):
    value = 0
    nqubits = len(h)
    for i in range(nqubits):
        for j in range(nqubits):
            value -= J[i][j] * x[i] * x[j]

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


def get_expectation(qc, backend, J, h, shots=100):
    counts = backend.run(qc, nshots=shots, repetition_period = 400e-4).result().get_counts()
    return compute_expectation(counts, J, h)


