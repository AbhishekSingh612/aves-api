import requests
import json


res = requests.post("https://bird-species-top5.herokuapp.com/predict-top5",
                    files={'file': open('44.jpg', 'rb')})
print(res.text)

