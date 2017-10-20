import jsonpickle
import numpy as np
import cv2
import json

from flask import Flask, request, Response
from model import Model

app = Flask(__name__)
m = Model()

@app.route('/api/detect', methods=['POST'])
def detect():
    r = request
    rgb_im_len = int(r.headers.get('rgb_im_len'))
    # convert string of image data to uint8
    nparr = np.fromstring(r.data[:rgb_im_len], np.uint8)
    im = cv2.imdecode(nparr, cv2.IMREAD_COLOR) # decode image

    output = m.detect(im)
    response = {'output': output}
    response_pickled = jsonpickle.encode(response)
    return Response(response=response_pickled, status=200, mimetype="application/json")


# start flask app
app.run(host="0.0.0.0", port=5000, threaded=True)
