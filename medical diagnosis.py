import requests

url = 'https://api.exchangerate.host/convert'
response = requests.get(url, params={
    "from": input(">Enter from Currency Code: "),
    "to": input(">Enter to Currency Code: "),
    "amount": input(">Input amount you want to convert")
})
data = response.json()["result"]

print(f">Converted Amount: {data}")