#%%
import numpy as np
from scipy.spatial import cKDTree


def discrete_continuous_info_fast(d, c, k=3, base=np.exp(1)):
    num_d_symbols = 0
    first_symbol = []
    symbol_IDs = np.zeros(len(d), dtype=int)
    c_split = {i: [] for i in range(len(d))}  # Initialize with all possible keys
    cs_indices = {i: [] for i in range(len(d))}  # Initialize with all possible keys

    # Sort lists by continuous variable 'c'
    c_idx = np.argsort(c)
    c = c[c_idx]
    d = d[c_idx]

    # Bin continuous data 'c' according to discrete symbols 'd'
    for c1 in range(len(d)):
        symbol_IDs[c1] = num_d_symbols + 1
        for c2 in range(num_d_symbols):
            if d[c1] == d[first_symbol[c2]]:
                symbol_IDs[c1] = c2
                break
        if symbol_IDs[c1] > num_d_symbols:
            num_d_symbols = num_d_symbols + 1
            first_symbol.append(c1)
            c_split[num_d_symbols] = []
            cs_indices[num_d_symbols] = []
        c_split[symbol_IDs[c1]].append(c[c1])
        cs_indices[symbol_IDs[c1]].append(c1)

    # Compute neighbor statistic for each data pair (c, d)
    m_tot = 0
    av_psi_Nd = 0
    V = np.zeros(len(d))
    psi_ks = 0

    for c_bin in range(1, num_d_symbols + 1):
        one_k = min(k, len(c_split[c_bin]) - 1)
        if one_k > 0:
            tree = cKDTree(np.array(c_split[c_bin]).reshape(-1, 1))
            for pivot in range(len(c_split[c_bin])):
                one_c = c_split[c_bin][pivot]
                _, indices = tree.query(one_c, k=one_k + 1)
                the_neighbor = indices[-1]  # the furthest neighbor

                distance_to_neighbor = abs(c_split[c_bin][the_neighbor] - one_c)

                m = max(one_k, indices.size - 1)
                m_tot = m_tot + psi(m)
                V[cs_indices[c_bin][pivot]] = 2 * distance_to_neighbor
        else:
            m_tot = m_tot + psi(num_d_symbols * 2)
            V[cs_indices[c_bin][0]] = 2 * (c[-1] - c[0])

        p_d = len(c_split[c_bin]) / len(d)
        av_psi_Nd = av_psi_Nd + p_d * psi(p_d * len(d))
        psi_ks = psi_ks + p_d * psi(max(one_k, 1))

    f = (psi(len(d)) - av_psi_Nd + psi_ks - m_tot / len(d)) / np.log(base)

    return f, V

def psi(x):
    return np.log(x)

# Example usage:
# Assuming d and c are numpy arrays of discrete and continuous variables respectively
# f, V = discrete_continuous_info_fast(d, c)
#%%
A = np.random.randint(0,2,size = 10000)
B = np.random.normal(0,20,size = 10000)

information, _ = discrete_continuous_info_fast(B,A, k = 1)
print(information)


# %%
