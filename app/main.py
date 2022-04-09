from flask import Flask, request, jsonify
# from torch_utils import transform_image, get_prediction, get_top5
# from names import getName
from app.torch_utils import transform_image, get_prediction, get_top5
from app.names import getName

app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files.get('file')
        if file is None or file.filename == "":
            return jsonify({'error': 'No file'})
        if not allowed_file(file.filename):
            return jsonify({'error': 'Not supported'})

        try:

            img_bytes = file.read()
            tensor = transform_image(img_bytes)
            prediction = get_prediction(tensor)
            data = {'prediction': prediction.item(
            ), 'birdName': getName(prediction.item())}

            return jsonify(data)

        except:
            return jsonify({'error': 'Some Error'})


@app.route('/predict-top5', methods=['POST'])
def predict_top5():
    if request.method == 'POST':
        file = request.files.get('file')
        if file is None or file.filename == "":
            return jsonify({'error': 'No file'})
        if not allowed_file(file.filename):
            return jsonify({'error': 'Not supported'})

        try:

            img_bytes = file.read()
            tensor = transform_image(img_bytes)
            print("44")
            data = get_top5(tensor)
            print("top 5 main ", data)
            list = []
            list.append(data)
            return jsonify(list)

        except:
            return jsonify({'error': 'Some Error'})


@app.route('/test', methods=['GET'])
def testServer():
    return "Started"


Allowed_Extention = {'png', 'jpg', 'jpeg'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in Allowed_Extention
