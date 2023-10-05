from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegression, MultilayerPerceptronClassifier, DecisionTreeClassifier, RandomForestClassifier, GBTClassifier, LinearSVC,NaiveBayes,FMClassifier
from pyspark.sql.functions import col
from pyspark.ml.feature import HashingTF, IDF, Tokenizer
from pyspark.ml import Pipeline, PipelineModel
import pandas as pd
import numpy as np
import random
from shutil import rmtree
from os import path
from utils import PATH_DATASET, models_dataframe, MODELS_DATAFRAME

TRAIN = False
TEST = True

spark = SparkSession.builder.appName('TRAIN_MODEL').getOrCreate()

df_spark_tr = spark.createDataFrame(pd.read_csv(f'{PATH_DATASET}train.csv').sample(frac=1)).select(col('label'), col('text'))
df_spark_te = spark.createDataFrame(pd.read_csv(f'{PATH_DATASET}test.csv')).select(col('label'), col('text'))

tokenizer = Tokenizer(inputCol="text", outputCol="words")
hashingTF = HashingTF(numFeatures=1000, inputCol=tokenizer.getOutputCol(), outputCol="features")

current_model = models_dataframe.RANDFOR

if current_model == models_dataframe.LOGREG:
    model_class = LogisticRegression(maxIter=100, regParam=0.001)
if current_model == models_dataframe.DECTREE:
    model_class = DecisionTreeClassifier(labelCol="label", featuresCol="features", maxDepth=10)
if current_model == models_dataframe.RANDFOR:
    model_class = RandomForestClassifier(labelCol="label", featuresCol="features", maxDepth=5, numTrees=50, seed=42)
if current_model == models_dataframe.GBTC:
    model_class = GBTClassifier(labelCol="label", featuresCol="features", maxIter=100, seed=42, maxDepth=10)
if current_model == models_dataframe.LINSVC:
    model_class = LinearSVC(maxIter=100, regParam=0.1)
if current_model == models_dataframe.NAIBAY:
    model_class = NaiveBayes(labelCol="label", featuresCol="features", smoothing=1.0, modelType="multinomial")
if current_model == models_dataframe.FMCLASS:
    model_class = FMClassifier(labelCol="label", featuresCol="features", stepSize=0.001)

if TRAIN:
    pipeline = Pipeline(stages=[tokenizer, hashingTF, model_class])

    model = pipeline.fit(df_spark_tr)

    model.write().overwrite().save(f'models/dataframe/{MODELS_DATAFRAME[current_model]}')

if TEST:
    model = PipelineModel.load(f'./models/dataframe/{MODELS_DATAFRAME[current_model]}')
    prediction = model.transform(df_spark_te)

    selected = prediction.select(col('label'), col('prediction'))

    # print(selected.rdd.filter(lambda x: x[0] == int(x[1])).count())
    # print(selected.rdd.count())
    acc = selected.rdd.filter(lambda x: x[0] == int(x[1])).count() / selected.rdd.count()
    print('acc: ', acc)

