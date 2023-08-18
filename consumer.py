from pyspark.sql import SparkSession
from pyspark.mllib.feature import IDF, HashingTF
from pyspark.mllib.classification import SVMWithSGD, SVMModel

dataset = []

spark = SparkSession.builder.appName('BIG_DATA_PROJECT').getOrCreate()

model = SVMModel.load(spark.sparkContext, './SVM')

def extract_feature_from_text(rdd):
    # split positive and negative
    rdd_positive_samples = rdd.filter(lambda x: x[0]==1)
    rdd_negative_samples = rdd.filter(lambda x: x[0]==0)

    # remove labels
    rdd_positive_text = rdd_positive_samples.map(lambda x: x[1])
    rdd_negative_text = rdd_negative_samples.map(lambda x: x[1])

    tf = HashingTF()
    rdd_positive_text_tf = rdd_positive_text.map(lambda x: tf.transform(x))
    rdd_negative_text_tf = rdd_negative_text.map(lambda x: tf.transform(x))

    
    idf = IDF().fit(rdd_positive_text_tf.union(rdd_negative_text_tf))

    rdd_positive_text_features = idf.transform(rdd_positive_text_tf)
    rdd_negative_text_features = idf.transform(rdd_negative_text_tf)

    rdd_positive_tr = rdd_positive_text_features.map(lambda x: (1, x))
    rdd_negative_tr = rdd_negative_text_features.map(lambda x: (0, x))

    rdd_feature = rdd_positive_tr.union(rdd_negative_tr)

    return rdd_feature

def func(batch_df, batch_id):
    df = batch_df.collect()
    
    for d in df:
        text = eval(d.value.decode('utf-8'))['text']
        label = eval(d.value.decode('utf-8'))['label']


        dataset.append((int(label), text))

    rdd_dataset = spark.sparkContext.parallelize(dataset)

    if rdd_dataset.count() > 0:
        # funzione che trasformi il testo in feature
        rdd_dataset_feature = extract_feature_from_text(rdd_dataset)

        # fare una map e richiamre il modello per fare le predict
        pred_and_labels = rdd_dataset_feature.map(lambda x: (x[0], model.predict(x[1])))

        # calcolare la percentuale di accuracy
        acc = pred_and_labels.filter(lambda x: x[0] == x[1]).count() / float(rdd_dataset_feature.count())
        print('Examples: ', rdd_dataset_feature.count())
        print('Accuracy: ', acc)

df = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", 'localhost:9092') \
    .option("subscribe", 'topic_1') \
    .load()

query = df.writeStream.foreachBatch(func).start()

query.awaitTermination()
