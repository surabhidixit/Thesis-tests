import os
import boto3
import io
from numpy import array
import time
from pyspark.mllib.clustering import GaussianMixture, GaussianMixtureModel
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from mpl_toolkits.mplot3d import Axes3D
from sklearn import mixture

outdata=sc.textFile("/home/surabhi/Desktop/thesis-August/train.csv")

import numpy as np
def parse(data):
    list=[]
    for i in range(len(data)):
        value=float(data[i][1:-1])
        list.append(value)
    return (list)
parsedata=outdata.map(lambda line:line.encode('utf-8').split(",")).map(lambda l:parse(l))


start_time = time.time()
gmm = GaussianMixture.train(parsedata, 80)
gmm.fit(parsedata)
print time.time()-start_time



#testing Gaussian mixture model for python
start_time = time.time()
#print sample1

gmix = mixture.GMM(n_components=90, covariance_type='full')
gmix.fit(parsedata)
#gmix.predict(parsedInSample1)
end_time = time.time()
gmpython= end_time-start_time
print gmpython

def parse1(data):
    sampleParse=[]
    for i in range(len(data)):
        sampleParse.append(parse2(data[i]))
    return np.array(sampleParse)
def parse2(data):
    data= (data.split(" "))
    list=[]
    for i in range(len(data)):
        if(data[i]!=''):
            list.append(float(data[i]))
    return np.array(list)
    
sample1=np.array(outdataRDD.takeSample(False,50))
parsedSample1=parse1(sample1)
#convert to spark dataframe for use in the pipeline
sample1RDD=sc.parallelize(parsedSample1)
print (sample1.shape)