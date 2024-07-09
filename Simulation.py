#%%
import numpy as np
import matplotlib.pyplot as plt
from numba import njit
from tqdm import tqdm
import math
import pandas as pd
from tqdm import tqdm
#%% Definimos los parametros que vamos a usar en la simulacion completa
global A, a, b, k, alpha, beta, gamma_ARNm, K_protein, gamma_protein, K_signal, Hill

A = 1.55/60
a = 1.04
b = 3/2
B = 0.19
k = 0.7
alpha = 0.617
beta = 6.17
gamma_ARNm = 1/5
K_protein = 50
gamma_protein = np.log(2)/1200
K_signal = 1000
Hill = 2

def Signal_distribution(K_signal_distribution, R_signal_distribution, P_signal_distribution):
    return ((math.factorial(K_signal_distribution+R_signal_distribution-1))/(math.factorial(R_signal_distribution-1)*math.factorial(K_signal_distribution)))*((1-P_signal_distribution)**K_signal_distribution)*((P_signal_distribution)**R_signal_distribution)

@njit
def num_polimerasa(Growth_Rate):
    return A*(1-np.e**(-a*(Growth_Rate**b)))
@njit
def num_gen(Growth_Rate):
    return 2**(k*Growth_Rate) + B
@njit
def Hill_Function(enviromental_signal, activation = True):

    if activation:
        return (enviromental_signal**Hill)/(K_signal**Hill + enviromental_signal**Hill)
    else:
        return (K_signal**Hill)/(K_signal**Hill + enviromental_signal**Hill)
@njit
def ARNm_production(enviromental_signal,Growth_Rate):
    return num_polimerasa(Growth_Rate)*num_gen(Growth_Rate)*(alpha + beta*Hill_Function(enviromental_signal, activation = True))
@njit
def ARNm_degradation(cantidad_ARNm):
    return gamma_ARNm*cantidad_ARNm
@njit
def protein_production(cantidad_ARNm):
    return K_protein*cantidad_ARNm
@njit
def protein_degradation(Growth_Rate, cantidad_protein):
    return (gamma_protein + (Growth_Rate/60)*np.log(2))*cantidad_protein

@njit
def modelo_constitutivo(enviromental_signal, Growth_Rate, cantidad_ARNm, cantidad_protein):

    propensidad_creacion_ARNm = ARNm_production(enviromental_signal,Growth_Rate)
    propensidad_creacion_proteina = protein_production(cantidad_ARNm)

    propensidad_degradacion_ARNm = ARNm_degradation(cantidad_ARNm)
    propensidad_degradacion_proteina = protein_degradation(Growth_Rate, cantidad_protein)

    return propensidad_creacion_ARNm, propensidad_creacion_proteina, propensidad_degradacion_ARNm, propensidad_degradacion_proteina
@njit
def Gillespie(trp0,tmax):
    """
    Esta funcion se emplea solamente para hacer la evolución de un paso individual en la celula. Evoluciona no un paso temporal, 
    pero si temporalmente la cantidad de veces que pueda evolucionar antes del tmax en una corrida
    """
    
    time,enviromental_signal, Growth_Rate, cantidad_ARNm, cantidad_protein =trp0 
    
    while time < tmax:
        s_1, s_2, s_3, s_4  = modelo_constitutivo(enviromental_signal, Growth_Rate, cantidad_ARNm, cantidad_protein)
        S_T = s_1 + s_2 + s_3 + s_4 
        
        τ = (-1/S_T)*np.log(np.random.rand())
        time+=τ

        if time < tmax:

            x = np.random.rand()

            if x <= (s_1)/S_T:
                cantidad_ARNm += 1

            elif x<= (s_1 + s_2)/S_T:
                cantidad_protein += 1
            
            elif x <= (s_1 + s_2 + s_3)/S_T :
                cantidad_ARNm-=1
            
            elif x <= (s_1 + s_2 + s_3 + s_4)/S_T :
                cantidad_protein-=1

            time+=τ
    return np.array([time,enviromental_signal, Growth_Rate, cantidad_ARNm, cantidad_protein]) 
@njit
def Estado_celula(X0,tiempos):
    X = np.zeros((len(tiempos),len(X0)))
    X[0] = X0
    for i in range(1,len(tiempos)):
        X[i] = Gillespie(X[i-1],tiempos[i])
    return X
#%%
num_cel = 100000 #número de células
Enviroment_Signal = np.random.negative_binomial(2000000, 0.5, size=num_cel)
Cells_Simulation = []
for S_Signal in tqdm(Enviroment_Signal):
    x0 = np.array([0., S_Signal, 3.33, 0., 0.])
    celula_individual = Estado_celula(x0,np.arange(0.,700.,1.))
    Cells_Simulation.append(celula_individual)
Cells_Simulation = np.array(Cells_Simulation)
#%%

#%%
from sklearn.feature_selection import mutual_info_classif
#%%
A = mutual_info_classif(Enviroment_Signal.reshape(-1,1), Cells_Simulation[:,-10,4].reshape(-1,1))
#%%
print(A)
#%%
plt.hist(Enviroment_Signal)
#%%
informacion_Lista = []
for i in tqdm(range(0,700)):

    data_C1 = {'S': Cells_Simulation[:,i,4],
            'Z': Cells_Simulation[:,i,4]}
    Cov_matrix_C1 = np.array(pd.DataFrame.cov(pd.DataFrame(data_C1)))

    Informacion = (1/2)*np.log2((Cov_matrix_C1[0][0]* Cov_matrix_C1[1][1])/(Cov_matrix_C1[0][0]* Cov_matrix_C1[1][1] - (Cov_matrix_C1[0][1])**2))
    informacion_Lista.append(Informacion)
#%%
print(Cov_matrix_C1)
#%%

Cells_Simulation.shape

#%%
Distribucion_proteina = Cells_Simulation[:,-10,4]
plt.hist(Distribucion_proteina)
#%% Estados Estacionarios de especies
def Stationary_State_ARNm(Growth_Rate):
    return (num_polimerasa(Growth_Rate)*num_gen(Growth_Rate)*(alpha + beta))/(gamma_ARNm)

def Stationary_State_Protein(Growth_Rate):
    return (K_protein*Stationary_State_ARNm(Growth_Rate))/(gamma_protein + (Growth_Rate/60)*np.log(2))

Stationary_States_ARNm = []
Stationary_States_Protein = []
for Growth_Rate in [0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0]:
    Stationary_States_ARNm.append(Stationary_State_ARNm(Growth_Rate))
    Stationary_States_Protein.append(Stationary_State_Protein(Growth_Rate))

# Graficas Estacionario
plt.figure(figsize=(8,5))
plt.title(r"Stationary States vs Growth Rates", fontsize = 14)
plt.scatter([0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0], Stationary_States_ARNm, label = r"$ARNm_{ss}$")
plt.scatter([0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0], Stationary_States_Protein, label = r"$Protein_{ss}$")
plt.legend()
#%%

plt.plot(Cells_Simulation[0,:,4])
# %%
Enviroment_Signal = np.random.negative_binomial(10, 0.2, size=num_cel)
plt.hist(Enviroment_Signal, bins = 50)
# %%
