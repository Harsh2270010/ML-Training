#import statistics as st 

#list=[1000,100,200,300]
#median=st.median(list)
#print(median)

#from sklearn.preprocessing import MinMaxScaler
#data=[(10,20),(20,30)]
#sc=MinMaxScaler()
#model =sc.fit(data)
#e=model.transform(data)
#print(e)


# print (median)

from sklearn.preprocessing import StandardScaler

data1=[(10,20),(20,30),(40,50)]
data2=[(1,2),(3,4),(5,6)]
sc=StandardScaler()
model=sc.fit(data1,data2)
e=model.transform(data1,data2)
print(e)