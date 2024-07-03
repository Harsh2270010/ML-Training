import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns 
from scipy.stats import poisson,binom
n=300 ; p=8.5
values=np.arange(1,n+1)
pmf=binom.pmf(values,n,p)
#print(pmf)
plt.bar(values,pmf)
plt.xlabel("data dictionary")
plt.ylabel("probability data")
plt.show()
