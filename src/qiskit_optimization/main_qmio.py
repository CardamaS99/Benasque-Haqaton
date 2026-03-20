from hamiltonian import H_total
from quantum import get_expectation, get_J_h, create_qaoa_circ
import numpy as np
from scipy.optimize import minimize
import pandas as pd
from qmiotools.integrations.qiskitqmio import QmioBackend


df_t1 = pd.read_csv('../../data/tiempo_caso1.csv')
tiempos = df_t1.to_numpy()

df_d1 = pd.read_csv('../../data/distancia_caso1.csv')
deltah = df_d1.to_numpy()

print("About to calculate hamiltonian")
hamil = H_total(np.ones(5), deltah, tiempos,0,0, len(tiempos), len(tiempos))

print("getting J,h")
J, h = get_J_h(hamil)

ansatz = create_qaoa_circ(J, h)
backend = QmioBackend(reservation_name="Benasque_QPU", tunnel_time_limit="00:10:00")

from qiskit import transpile

ansatz = transpile(ansatz, backend)

def cost_qmio(params):
    circuit = ansatz.assign_parameters(params)
    return get_expectation(circuit,backend, J, h)


init_params = np.random.uniform(-np.pi, np.pi, 2)

cost_qmio(init_params)