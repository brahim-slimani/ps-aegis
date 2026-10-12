import requests

response = requests.get("https://httpbin.org/ip")

#print(response.json())
print('Your IP is {0}'.format(response.json()['origin']))
