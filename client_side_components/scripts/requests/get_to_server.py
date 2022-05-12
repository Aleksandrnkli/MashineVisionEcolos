"""
This script sends get-request to server
"""

import requests

auth_token = '867a038bf8c51c8d5627d566aeb60ffbbdc5c615' 
server_url = "http://127.0.0.1:7000/api/lpr/" 
auth_header = {'Authorization': 'Token ' + auth_token}

r = requests.get(server_url, headers=auth_header)

if r.status_code == 200:
    print("Status: OK")
    print(r.json()['results'])
else:
    print(r.text)
    r.raise_for_status()
