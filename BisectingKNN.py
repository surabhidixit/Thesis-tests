from numpy import array

from pyspark.mllib.clustering import BisectingKMeans, BisectingKMeansModel
from pyspark.mllib.clustering import KMeans, KMeansModel

import os
from pyspark import SparkConf, SparkContext


# Configure the environment                                                     
if 'SPARK_HOME' not in os.environ:
    os.environ['SPARK_HOME'] = '/home/spark-2.0.2-bin-hadoop2.7'

conf = SparkConf().setAppName('test').setMaster('local[4]')
sc = SparkContext(conf=conf)
sc.setLogLevel("ERROR")

# Load and parse the data
data = sc.textFile("/home/surabhi/Desktop/CollegeStuff/Thesis/Output.csv")
parsedData = data.map(lambda line: array([float(x) for x in line.split(',')]))

# Build the model (cluster the data)
model = BisectingKMeans.train(parsedData, 25, maxIterations=15)

# Evaluate clustering
cost = model.computeCost(parsedData)
print("Bisecting K-means Cost = " + str(cost))