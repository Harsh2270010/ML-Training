#Anova and its principles

from scipy.stats import f_oneway
from scipy.stats import *
population1=[10,20,30,40,50]
population2=[70,80,90,40,50]
population3=[10,20,30,40,40]
population4=[40,50,60,30,70]

e=f_oneway(population1,population2,population3,population4)
print(e)

list1=["12","232","abf"]
list2=[]
for i in  list1:
    if(i.isdigit()):
        list2.append(i)
print(list2)