"""
This script sends patch-request to server
"""

import requests

auth_token = '867a038bf8c51c8d5627d566aeb60ffbbdc5c615' 
detection_id = 15
server_url = f"http://127.0.0.1:7000/api/lpr/{detection_id}/" 
auth_header = {'Authorization': 'Token ' + auth_token}
marked_as_error = True  
license_plate = 'BRNDNW18' 
data = {
    "license_plate": license_plate,
    "marked_as_error": marked_as_error
}

r = requests.patch(server_url, data=data, headers=auth_header)


if r.status_code == 200:
    print("Status: OK No content")
    print(r.text)
else:
    print(r.text)
    r.raise_for_status()
