#%%
import numpy as np
import matplotlib.pyplot as plt
#%%
global A, a, b, k, alpha, beta, gamma_ARNm, K_protein, gamma_protein

A = 1
a = 1
b = 1
k = 1
B = 1 
alpha = 1
beta = 1
gamma_ARNm = 1
K_protein = 1
gamma_protein = 1

@njit
def num_polimerasa(Growth_Rate):
    return A*(1-np.exp**(-a*(Growth_Rate**b)))
@njit
def num_gen(Growth_Rate):
    return 2**(k*Growth_Rate) + B
@njit
def Hill_Function(enviromental_signal, activation = True):

    if activation:
        return None
    else:
        return None
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
    return (gamma_protein + Growth_Rate*np.log(2))*cantidad_protein

@njit
def modelo_constitutivo(enviromental_signal, Growth_Rate, cantidad_ARNm, cantidad_protein):

    propensidad_creacion_ARNm = ARNm_production(enviromental_signal,Growth_Rate)
    propensidad_creacion_proteina = protein_production(cantidad_ARNm)

    propensidad_degradacion_ARNm = ARNm_degradation(cantidad_ARNm)
    propensidad_degradacion_proteina = protein_degradation(Growth_Rate, cantidad_protein)


    return propensidad_creacion_ARNm, propensidad_creacion_proteina, propensidad_degradacion_ARNm, propensidad_degradacion_proteina

@njit('f8[:](f8[:],f8)')
def Gillespie(trp0,tmax):
    """
    Esta funcion se emplea solamente para hacer la evolución de un paso individual en la celula. Evoluciona no un paso temporal, 
    pero si temporalmente la cantidad de veces que pueda evolucionar antes del tmax en una corrida
    """
    
    t,enviromental_signal, Growth_Rate, cantidad_ARNm, cantidad_protein =trp0 

    while t < tmax:
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

            t+=τ
    return np.array([t,enviromental_signal, Growth_Rate, cantidad_ARNm, cantidad_protein]) 
        
@njit('f8[:,:](f8[:],f8[:])')
def Estado_celula(X0,tiempos):

    X = np.zeros((len(tiempos),len(X0)))
    X[0] = X0
    
    for i in range(1,len(tiempos)):
        X[i] = Gillespie(X[i-1],tiempos[i])
    
    return X

x0 = np.array([0., 0., 0., 0., 0.])

num_cel = 2000 #número de células 
celulas = np.array([Estado_celula(x0,np.arange(0.,700.,2.)) for i in tqdm(range(num_cel))])