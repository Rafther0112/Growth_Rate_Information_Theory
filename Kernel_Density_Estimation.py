#%%
import numpy as np
from scipy import stats
#%%
def measure(n):
    "Measurement model, return two coupled measurements."
    m1 = np.random.normal(size=n)
    m2 = np.random.normal(scale=0.5, size=n)
    return m1+m2, m1-m2

m1, m2 = measure(200000)
xmin = m1.min()
xmax = m1.max()
ymin = m2.min()
ymax = m2.max()
# %%
X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
positions = np.vstack([X.ravel(), Y.ravel()])
values = np.vstack([m1, m2])
kernel = stats.gaussian_kde(values)
Z = np.reshape(kernel(positions).T, X.shape)
# %%
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.imshow(np.rot90(Z), cmap=plt.cm.gist_earth_r,
          extent=[xmin, xmax, ymin, ymax])
ax.plot(m1, m2, 'k.', markersize=2)
ax.set_xlim([xmin, xmax])
ax.set_ylim([ymin, ymax])
plt.show()
# %%
# %%
from scipy.stats import gaussian_kde
from sklearn.feature_selection import mutual_info_regression
import numpy as np
#%%
#generate some data
x1 = m1
x2 = m2
dataset = np.stack((x1,x2))

#kde of x1,x2 and joint x1,x2
kd_x1 = gaussian_kde(dataset[0,:])
kd_x2 = gaussian_kde(dataset[1,:])
kd_x1_x2 = gaussian_kde(dataset)
#%%

#%%
#entropy calculations
h_x1_x2 = -np.mean(np.log2(kd_x1_x2(dataset)))
h_x1 = -np.mean(np.log2(kd_x1(dataset[0,:])))
h_x2 = -np.mean(np.log2(kd_x2(dataset[1,:])))
#%%
#Information calculations
I_x1_x2_kde = (h_x1 + h_x2 - h_x1_x2)
I_x1_x2_sklearn = mutual_info_regression(x2.reshape(-1,1),x1)

print(f"Mutual information using kde: {I_x1_x2_kde}")
print(f"Mutual information using sklear (nearest neighbors): {I_x1_x2_sklearn}")
# %%
from sklearn.datasets import make_regression
from sklearn.feature_selection import mutual_info_regression

X = np.random.normal(0,2,100000).reshape(-1,1)
Y = np.random.normal(0,2,100000).reshape(-1,1)

print(mutual_info_regression(X, Y))
#%%
import numpy as np

# Media y covarianza de la distribución bivariada
mean = [0, 3]  # Media de ambas variables
cov = [[1, 0.5], [0.5, 1]]  # Matriz de covarianza

# Generar datos bivariados
data = np.random.multivariate_normal(mean, cov, 1000000).T

# Obtener las dos listas de datos
data1 = data[0]  # Primera lista
data2 = data[1]  # Segunda lista

print(mutual_info_regression(data1.reshape(-1,1), data2.reshape(-1,1)))
# %%
np.linalg.det(cov)
# %%
print((1/2)*np.log((1*0.5)/(np.linalg.det(cov))))
# %%
from scipy.stats import entropy
from collections import Counter

data = np.random.multivariate_normal(mean, cov, 100000).T

# Obtener las dos listas de datos
data1 = data[0]  # Primera lista
data2 = data[1]  # Segunda lista

contador = Counter(data1)
total = len(data1)
probabilidades_array = [frecuencia / total for frecuencia in contador.values()]

qk = np.array(probabilidades_array)  # biased coin
entropy(qk, base = 2)
# %%
