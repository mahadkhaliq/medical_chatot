import requests

# The URL of your Flask API
url = 'http://localhost:5000/query'

# The data to send in the request
data = {
    'query': 'What are the effects of high blood pressure?'
}

# Send the POST request
response = requests.post(url, json=data)

# Print the response
if response.status_code == 200:
    print('Response from API:', response.json())
else:
    print('Failed to get response, status code:', response.status_code)