import requests
import time
import pickle
import random
import os
import json

print(os.getcwd())

with open('./Xy.pickle', 'rb') as handle:
    _, _, X_test, y_test = pickle.load(handle)

min_, max_ = 0, X_test.shape[0]-1

while True:
    try:
        time.sleep(1)
        id_ = random.randint(min_, max_)
        current = X_test.iloc[id_]
        current.Time = time.time()
        current = current.to_dict()
        current['target'] = int(y_test.iloc[id_])
        response = requests.get('https://api.randomdatatools.ru', params={'count': 2, 'params': 'LastName,FirstName,FatherName'})
        persons = [f"{person['LastName']} {person['FirstName']} {person['FatherName']}" for person in response.json()]
        current['from'], current['to'] = persons
        requests.post(url="http://localhost:5001/handler", json=json.dumps(current))
        print("request_sent")
    except:
        pass