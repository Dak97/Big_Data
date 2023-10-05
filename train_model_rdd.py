from pyspark.sql import SparkSession
import pandas as pd
import numpy as np
import random
from shutil import rmtree
from os import path
from utils import *

TRAIN = False
TEST = True
LOAD = True

    
current_model = models.DECTREE

spark = SparkSession.builder.appName('TRAIN_MODEL').getOrCreate()

# carico il training e il test set
if LOAD:
    df_spark_tr = spark.createDataFrame(pd.read_csv(f'{PATH_DATASET}train.csv'))
    df_spark_te = spark.createDataFrame(pd.read_csv(f'{PATH_DATASET}test.csv'))
else:
    df = pd.read_csv('movie_model.csv')

    df = df.sample(frac=1)

    tr_num = int((df.shape[0] * 80)/100)

    df_tr = df.iloc[:tr_num,:]
    df_te = df.iloc[tr_num:,:]

    df_tr.to_csv(f'{PATH_DATASET}train.csv')
    df_te.to_csv(f'{PATH_DATASET}test.csv')

    df_spark_tr = spark.createDataFrame(df_tr)
    df_spark_te = spark.createDataFrame(df_te)

rdd_tr = df_spark_tr.rdd
rdd_te = df_spark_te.rdd

                                   
if TRAIN:
    rdd_train = extract_feature_from_text(rdd_tr, num_feat=2000)

    if path.isdir(f'{get_path(current_model)}data'):
        rmtree(f'{get_path(current_model)}data')
        rmtree(f'{get_path(current_model)}metadata')

    if current_model == models.DECTREE:
        model = get_model_class(current_model).trainClassifier(rdd_train, numClasses=2, categoricalFeaturesInfo={},
                                     impurity='gini', maxDepth=5, maxBins=32)
    else:
        # svm, logistic regression, naive bayes
        model = get_model_class(current_model).train(rdd_train)

    model.save(spark.sparkContext, f'{get_path(current_model)}')

    print('\nTrain ended.\n')

if TEST:
    model = get_model_class(current_model, load=True).load(spark.sparkContext, f'{get_path(current_model)}')

    rdd_test = extract_feature_from_text(rdd_te, train=False)
    if current_model == models.DECTREE:
        pred_and_labels = model.predict(rdd_test.map(lambda x: x))
        print(pred_and_labels.collect())
    else:
        pred_and_labels = rdd_test.map(lambda x: (x[0], model.predict(x[1])))
    
    error = pred_and_labels.filter(lambda x: x[0] == x[1]).count() / float(rdd_test.count())

    print('\Testing ended.\n')
    print('Accuracy:', error)
    
