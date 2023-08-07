from pyspark.sql import SparkSession

def func(batch_df, batch_id):
    df = batch_df.collect()
    
    dataset = []
    for d in df:
        text = eval(d.value.decode('utf-8'))['text']
        label = eval(d.value.decode('utf-8'))['label']

        dataset.append((text, label))
        
    print(dataset)
    
    

spark = SparkSession.builder.appName('BIG_DATA_PROJECT').getOrCreate()

df = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", 'localhost:9092') \
    .option("subscribe", 'topic_1') \
    .load()



query = df.writeStream.foreachBatch(func).start()

query.awaitTermination()
