import http.client, urllib.parse

conn = http.client.HTTPSConnection('api.thenewsapi.com')

params = urllib.parse.urlencode({
    'api_token': 'TFp9I8aFM42JBthMLpKdWNbRluZQ25rfQH8ttenQ',
    'categories': 'business,tech',
    'limit': 50,
    })

conn.request('GET', '/v1/news/all?{}'.format(params))

res = conn.getresponse()
data = res.read()

print(data.decode('utf-8'))