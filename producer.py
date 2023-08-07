from kafka import KafkaProducer
import pandas as pd
import json

df = pd.read_csv('movie.csv')

texts = df['text'].head(4)
labels = df['label'].head(4)

prod = KafkaProducer(bootstrap_servers='localhost:9092', value_serializer=lambda v: json.dumps(v).encode('utf-8'))


for text, label in zip(texts,labels):
    prod.send('topic_1', {'text': text, 'label': label})
    prod.flush()

