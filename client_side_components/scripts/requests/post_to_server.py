"""
This script sends new LPR detection to the server by post-request
"""

import requests


auth_token = '867a038bf8c51c8d5627d566aeb60ffbbdc5c615' 
license_plate = 'CLT1803'
marked_as_error = True
image_filename = "image.jpg" 
server_url = "http://127.0.0.1:7000/api/lpr/" 
auth_header = {'Authorization': 'Token ' + auth_token}

files = {
    'image': (image_filename, open(image_filename, 'rb'), 'image/jpeg')
}

data = {
    'license_plate': license_plate,
    'marked_as_error': marked_as_error
}

# multiform-data http request
r = requests.post(server_url, files=files, data=data, headers=auth_header)

if r.status_code == 201:
    print("Status: Created")
else:
    print(r.text)
    r.raise_for_status()
