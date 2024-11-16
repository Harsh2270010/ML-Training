import matplotlib.pyplot as plt  
import numpy as np  

# Generate random data  
x = np.random.normal(5, 1, 1000)  
y = np.random.normal(10, 2, 1000)  

# Create scatter plot  
plt.scatter(x, y, alpha=0.5)  
plt.title('Scatter Plot with Random Data')  
plt.xlabel('X-axis')  
plt.ylabel('Y-axis')  
plt.show()  