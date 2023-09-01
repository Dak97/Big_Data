from kafka import KafkaProducer
import pandas as pd
import json

df = pd.read_csv('datasets/movie_stream.csv')

# shuffle
df = df.sample(frac=1)

texts = df['text'].head(20000)
labels = df['label'].head(20000)


prod = KafkaProducer(bootstrap_servers='localhost:9092', value_serializer=lambda v: json.dumps(v).encode('utf-8'))


for text, label, i in zip(texts,labels, range(len(texts))):
    print(i)
    prod.send('topic_1', {'text': text, 'label': label})
    prod.flush()

