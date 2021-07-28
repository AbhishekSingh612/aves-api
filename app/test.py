import requests
import json


res = requests.post("https://bird-species-class-prediction.herokuapp.com/predict",
                    files={'file': open('44.jpg', 'rb')})
print(res.text)

