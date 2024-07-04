#%%
import numpy as np

def Freedman_Diaconis_Rule(data):
    """
    Funcion para estimar la cantidad de bins necesarias para 
    hacer la representacion de las distribucion de probabilidad 
    de una serie de datos
    """
    
    q75, q25 = np.percentile(data, [75 ,25])
    iqr = q75 - q25
    N = len(data)
    bin_width_fd = 2 * iqr * N ** (-1/3)
    num_bins_fd = int((data.max() - data.min()) / bin_width_fd)

    return num_bins_fd

# %%

mean = 0  # Mean of the distribution
std_dev = 1  # Standard deviation of the distribution
num_points = 1000000  # Number of data points

# Generate data points from a normal distribution
data = np.random.normal(mean, std_dev, num_points)

# Plot the histogram of the data
plt.hist(data, bins=Freedman_Diaconis_Rule(data), edgecolor='k', alpha=0.7)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram of Normally Distributed Data')
plt.show()
#%%
import numpy as np
from sklearn.metrics import mutual_info_score
import matplotlib.pyplot as plt

# Parameters for the normal distributions
mean1 = 0  # Mean of the first distribution
std_dev1 = 1  # Standard deviation of the first distribution

mean2 = 0  # Mean of the second distribution
std_dev2 = 2  # Standard deviation of the second distribution

num_points = 1000000  # Number of data points

# Generate data points from two normal distributions
data1 = np.random.normal(mean1, std_dev1, num_points)
data2 = np.random.normal(mean2, std_dev2, num_points)

# Discretize the data using 100 bins
num_bins = Freedman_Diaconis_Rule(data1)
data1_discrete = np.digitize(data1, bins=np.linspace(data1.min(), data1.max(), num_bins))
data2_discrete = np.digitize(data2, bins=np.linspace(data2.min(), data2.max(), num_bins))

# Calculate mutual information
mi = mutual_info_score(data1_discrete, data2_discrete)
print(f'Mutual Information: {mi}')
#%%
# Plot the histograms of the two distributions
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].hist(data1, bins=num_bins, edgecolor='k', alpha=0.7)
axes[0].set_title('Histogram of Data1')
axes[0].set_xlabel('Value')
axes[0].set_ylabel('Frequency')

axes[1].hist(data2, bins=num_bins, edgecolor='k', alpha=0.7)
axes[1].set_title('Histogram of Data2')
axes[1].set_xlabel('Value')
axes[1].set_ylabel('Frequency')

plt.tight_layout()
plt.show()
# %%
