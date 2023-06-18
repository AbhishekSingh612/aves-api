import requests
import json

x = int(input("Enter.."))

if(x == 3):
    res = requests.post("https://bird-species-top5.herokuapp.com/predict-top5",
                        files={'file': open('44.jpg', 'rb')})
    print(res.text)
elif(x == 2):
    res = requests.post("https://bird-species-top5.herokuapp.com/predict",
                        files={'file': open('44.jpg', 'rb')})
    print(res.text)
elif(x == 1):
    res = requests.get("https://bird-species-top5.herokuapp.com/test")
    print("successs")
elif(x == 4):
    res = requests.post("http://localhost:5000/predict-top5",
                        files={'file': open('44.jpg', 'rb')})
    print(res.text)
else:
    print("invalid")
