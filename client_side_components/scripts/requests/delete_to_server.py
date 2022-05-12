"""
This script sends delete-request to server
"""

import requests

detection_id = 16
auth_token = '867a038bf8c51c8d5627d566aeb60ffbbdc5c615' 
server_url = f"http://127.0.0.1:7000/api/lpr/{detection_id}"
auth_header = {'Authorization': 'Token ' + auth_token}

r = requests.delete(server_url, headers=auth_header)

if r.status_code == 204:
    print("Status: Deleted")
else:
    print(r.text)
    r.raise_for_status()
