import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st

offset = False

# Generate and Sort Noisy Fake Data
N = 1000
x = np.sort(np.random.randn(N))
if offset:
    y = np.sort(np.random.randn(N)) - 0.25

# Compute CDF and peaked CDF (pCDF)
# The peaked CDF flips over at the median
CDF  = np.arange(1,len(x)+1)/(len(x))
pCDF = np.minimum(CDF, 1-CDF)
actualpCDF = np.minimum(st.norm.cdf(x), 1 - st.norm.cdf(x))


# Plot vs histogram
fig, axes = plt.subplots(1,2, figsize=(12,5))
axes[0].set_title('Traditional Density Estimation')
axes[0].hist(x, bins=100, density=True, alpha=0.5, label='hist')
axes[0].plot(x, st.norm.pdf(x), label='normal PDF')
if offset:
    axes[0].hist(y, bins=100, density=True, alpha=0.5, label='hist offset')
axes[0].set_xlabel('Random Variable Values (Outcomes)')
axes[0].set_ylabel('Probability Density')
axes[0].legend()

axes[1].set_title('Peaked CDFs')
axes[1].plot(x, pCDF, label='Measured pCDF')
axes[1].plot(x, actualpCDF, label='normal pCDF')
if offset:
    axes[1].plot(y, pCDF, label='Measured offset pCDF')
axes[1].set_xlabel('Random Variable Values (Outcomes)')
axes[1].set_ylabel('Peaked CDF')
axes[1].legend()


plt.show()


