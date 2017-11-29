import numpy as np
from numpy import genfromtxt
import time
from scipy.spatial import distance
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn.cluster import MiniBatchKMeans
import random
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from numpy import binary_repr
import pyclust

train="/home/surabhi/Desktop/matlab files/Cortex_coords.csv"
test="/home/surabhi/Desktop/matlab files/Output.csv"
bkmcenters=[]

traindata = genfromtxt(train, delimiter=',')[:100]
#print traindata.shape

testpotentials=genfromtxt(test,delimiter=',')[:10,:]

bkm = pyclust.BisectKMeans(n_clusters=10)
bkm.fit(traindata)

for (key,value) in (bkm.centers_.items()):
    bkmcenters.append(value)
bkmcenters=np.asarray(bkmcenters)




#np.savetxt('/home/surabhi/MATLAB2017/bin/centroidsBKM.csv', (bkmcenters[:,:1],bkmcenters[:,1:2],bkmcenters[:,2:]), delimiter=',')
#the distance matrix containing distances between electrodes and clusters is created in Matlab and saved as csv
#open the distance matrix file
distancesBKM=genfromtxt('/home/surabhi/Desktop/matlab files/10clusters/DistanceMatrixBKM.csv', delimiter=',')

def findmapping(test,given):
    x=np.array(test).T
    y=np.array(given).T
    arr=[]
    mapper={}
    k=0
    
    for i in range(len(x)):
        arr=[]
        for j in range(len(y[i])):
            #print y[i]
            #print ('calculating distance between ',x[i],' &',y[i][j])
            arr.append(distance.euclidean(x[i],y[i][j]))
        minindex=arr.index(min(arr))
        mapper[k]=minindex
        k+=1
    return mapper



#testpotentials=testpotentials[:50,:]
#print testpotentials[0].reshape(50,1)
sourcematrix=distancesBKM.reshape(10,25)

distancesBKM=distancesBKM.reshape(10,25)
print testpotentials


labels = np.random.randint(2, size=(10,)).reshape(10,1)
#to test single active electrode
singleactive=binary_repr(8,width=10)
singleactive=np.array(list(singleactive), dtype=int).reshape(10,1)

active1=sourcematrix*labels
active2=sourcematrix*singleactive

active1=active1[~np.all(active1 == 0, axis=1)]
active2=active2[~np.all(active2 == 0, axis=1)]

activeloc1=bkmcenters*labels
activeloc2=bkmcenters*singleactive

#activeloc1=activeloc1[~(activeloc1==0)]
#activeloc2=activeloc2[~(activeloc2==0)]

first=findmapping(testpotentials,active1)
#second=findmapping(testpotentials,active2)

activeloc1=activeloc1[~np.all(activeloc1 == 0, axis=1)]
activeloc2=activeloc2[~np.all(activeloc2 == 0, axis=1)]
#print first

testlocations=genfromtxt("/home/surabhi/Desktop/thesis-August/Electrode_locations.csv",delimiter=',')

Xt=testlocations[:,:1]
Yt=testlocations[:,1:2]
Zt=testlocations[:,2:]
Xt= Xt[~np.isnan(Xt)]
Yt= Yt[~np.isnan(Yt)]
Zt= Zt[~np.isnan(Zt)]
#activeloc1= activeloc1[np.all(activeloc1, axis=1)]
Xbkm=activeloc1[:,:1]
Ybkm=activeloc1[:,1:2]
Zbkm=activeloc1[:,2:]
Xbkm= Xbkm[~np.isnan(Xbkm)]
Ybkm= Ybkm[~np.isnan(Ybkm)]
Zbkm= Zbkm[~np.isnan(Zbkm)]

fig = plt.figure()
ax = Axes3D(fig)

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
for i in range(len(Xbkm)): #plot each point + it's index as text above
 ax.scatter(Xbkm[i],Ybkm[i],Zbkm[i],color='b') 
 #ax.text(active1[i],0,0,  '%s' % (str(i)), size=20, zorder=1,  color='k')
 ax.text(Xbkm[i],Ybkm[i],Zbkm[i],  '%s' % (str(i)), size=20, zorder=1,  
 color='green') 

for i in range(len(Xt)): #plot each point + it's index as text above
 ax.scatter(Xt[i],Yt[i],Zt[i],color='r',marker='+') 
 #ax.text(active1[i],0,0,  '%s' % (str(i)), size=20, zorder=1,  color='k')
 ax.text(Xt[i],Yt[i],Zt[i],  '%s' % (str(i)), size=10, zorder=1,  
 color='black') 

plt.show()