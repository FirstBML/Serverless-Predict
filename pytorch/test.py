import requests

#url = 'http://localhost:8080/2015-03-31/functions/function/invocations'

url = 'https://vs8jkvk9xf.execute-api.eu-north-1.amazonaws.com/prod/predict-classifier'

request = {
    "url": "http://bit.ly/mlbookcamp-pants"
}

result = requests.post(url, json=request).json()
print(result)