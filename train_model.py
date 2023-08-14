from pyspark.sql import SparkSession
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.feature import IDF, HashingTF
from pyspark.mllib.classification import SVMWithSGD, SVMModel
import pandas as pd
import numpy as np
import random
from shutil import rmtree

TRAIN = False
LOAD = not TRAIN

spark = SparkSession.builder.appName('TRAIN_MODEL').getOrCreate()


if LOAD:
    df_spark_tr = spark.createDataFrame(pd.read_csv('train.csv'))
    df_spark_te = spark.createDataFrame(pd.read_csv('test.csv'))
else:
    df = pd.read_csv('movie_model.csv')
    # df = df.head(5000)

    df = df.sample(frac=1)

    tr_num = int((df.shape[0] * 80)/100)

    df_tr = df.iloc[:tr_num,:]
    df_te = df.iloc[tr_num:,:]

    df_tr.to_csv('train.csv')
    df_te.to_csv('test.csv')

    df_spark_tr = spark.createDataFrame(df_tr)
    df_spark_te = spark.createDataFrame(df_te)

rdd_tr = df_spark_tr.rdd
rdd_te = df_spark_te.rdd



def transform_train_data(rdd, tf=None, idf=None, train=True):
    # split positive and negative
    rdd_positive_samples = rdd.filter(lambda x: x['label']==1)
    rdd_negative_samples = rdd.filter(lambda x: x['label']==0)

    # remove labels
    rdd_positive_text = rdd_positive_samples.map(lambda x: x['text'])
    rdd_negative_text = rdd_negative_samples.map(lambda x: x['text'])

    if tf == None:
        tf = HashingTF()
    rdd_positive_text_tf = rdd_positive_text.map(lambda x: tf.transform(x))
    rdd_negative_text_tf = rdd_negative_text.map(lambda x: tf.transform(x))

    if idf == None:
        idf = IDF().fit(rdd_positive_text_tf.union(rdd_negative_text_tf))

    rdd_positive_text_features = idf.transform(rdd_positive_text_tf)
    rdd_negative_text_features = idf.transform(rdd_negative_text_tf)

    if train:
        rdd_positive_tr = rdd_positive_text_features.map(lambda x: LabeledPoint(1, x))
        rdd_negative_tr = rdd_negative_text_features.map(lambda x: LabeledPoint(0, x))
    else:
        rdd_positive_tr = rdd_positive_text_features.map(lambda x: (1, x))
        rdd_negative_tr = rdd_negative_text_features.map(lambda x: (0, x))

    rdd_train = rdd_positive_tr.union(rdd_negative_tr)

    return rdd_train, tf, idf


rdd_train , tf, idf = transform_train_data(rdd_tr)
                                           
if TRAIN:

    rmtree('./SVM/data')
    rmtree('./SVM/metadata')

    svm = SVMWithSGD.train(rdd_train, iterations=100)

    svm.save(spark.sparkContext, './SVM')

    print('\nTrain ended.\n')

else:
    svm = SVMModel.load(spark.sparkContext, './SVM')

    rdd_test = transform_train_data(rdd_te, tf, idf, False)

    pred_and_labels = rdd_test[0].map(lambda x: (x[0], svm.predict(x[1])))
    error = pred_and_labels.filter(lambda x: x[0] == x[1]).count() / float(rdd_test[0].count())
    print('\Testing ended.\n')
    print('Accuracy:', error)
    
