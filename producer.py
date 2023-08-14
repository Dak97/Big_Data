from kafka import KafkaProducer
import pandas as pd
import json

df = pd.read_csv('movie.csv')

print(df.size)

texts = df['text'].head(50)
labels = df['label'].head(50)

prod = KafkaProducer(bootstrap_servers='localhost:9092', value_serializer=lambda v: json.dumps(v).encode('utf-8'))


for text, label in zip(texts,labels):
    prod.send('topic_1', {'text': text, 'label': label})
    prod.flush()

