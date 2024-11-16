import matplotlib.pyplot as plt
import numpy as np

# Generate random data
np.random.seed(10)  # For reproducibility
data = [np.random.normal(0, std, 100) for std in range(1, 4)]  # Three different distributions

# Create a box plot
plt.boxplot(data, vert=True, patch_artist=True, labels=['Group 1', 'Group 2', 'Group 3'])

# Add title and labels
plt.title('Box Plot Example')
plt.xlabel('Groups')
plt.ylabel('Values')

# Show the plot
plt.grid(axis='y')  # Optional: Add grid lines for better readability
plt.show()