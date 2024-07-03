import matplotlib.pyplot as plt 
import numpy as np
e=plt.axes(projection="3d")



x1=np.random.randint(0,50,50)
y1=np.random.randint(0,40,50)
z1=np.random.randint(0,30,50)
e.scatter(x1,y1,z1,marker='d',label='dataset1')

x2=np.random.randint(0,50,50)
y2=np.random.randint(0,40,50)
z2=np.random.randint(0,30,50)
e.scatter(x2,y2,z2,marker='o',label='dataset2')

x3=np.random.randint(0,50,50)
y3=np.random.randint(0,40,50)
z3=np.random.randint(0,30,50)
e.scatter(x3,y3,z3,marker='^',label='dataset3')

e.legend()
#  e.scatter(xdata1,ydata2,zdata3)
plt.show()