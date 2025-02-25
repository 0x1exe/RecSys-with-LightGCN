from confluent_kafka import Producer
import pandas as pd
import json

conf = {
    'bootstrap.servers': '127.0.0.1:9092'
}
producer = Producer(conf)

def delivery_report(err, msg):
    if err is not None:
        print('Message delivery failed: {}'.format(err))
    else:
        print('Message delivered to {} [{}]'.format(msg.topic(), msg.partition()))

csv_file = './u.data'
df = pd.read_csv(csv_file)

topic = 'user_interactions'
for index, row in df.iterrows():
    data = row.to_dict()
    message = json.dumps(data)
    
    producer.produce(topic, value=message.encode('utf-8'), callback=delivery_report)

producer.flush()