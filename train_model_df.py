from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegression
from pyspark.sql.functions import col
from pyspark.ml.feature import HashingTF, IDF, Tokenizer
from pyspark.ml import Pipeline
import pandas as pd
import numpy as np
import random
from shutil import rmtree
from os import path
from utils import PATH_DATASET


spark = SparkSession.builder.appName('TRAIN_MODEL').getOrCreate()

df_spark_tr = spark.createDataFrame(pd.read_csv(f'{PATH_DATASET}train.csv')).select(col('label'), col('text'))
df_spark_te = spark.createDataFrame(pd.read_csv(f'{PATH_DATASET}test.csv')).select(col('label'), col('text'))

tokenizer = Tokenizer(inputCol="text", outputCol="words")
hashingTF = HashingTF(inputCol=tokenizer.getOutputCol(), outputCol="features")
lr = LogisticRegression(maxIter=10, regParam=0.001)
pipeline = Pipeline(stages=[tokenizer, hashingTF, lr])

model = pipeline.fit(df_spark_tr)


prediction = model.transform(df_spark_te)
selected = prediction.select(col('label'), col('prediction'))

print(selected.rdd.filter(lambda x: x[0] == int(x[1])).count())
print(selected.rdd.count())
acc = selected.rdd.filter(lambda x: x[0] == int(x[1])).count() / selected.rdd.count()
print('acc: ', acc)
print(df_spark_te.count())
# selected = prediction.select("id", "text", "probability", "prediction")
# for row in selected.collect():
#     rid, text, prob, prediction = row
#     print(
#         "(%d, %s) --> prob=%s, prediction=%f" % (
#             rid, text, str(prob), prediction   # type: ignore
#         )
#     )