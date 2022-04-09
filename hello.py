from flask import Flask, request, render_template
from flask_cors import CORS, cross_origin
app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route('/<user>/profile', methods=['GET'])
def hello_world(user):
    return 'hello' + user


@app.route('/', methods=['POST'])
@cross_origin()
def hello_post():
    user = request.form['nm']
    return '<h1>Hello GET </h1><br>' + user


if __name__ == '__main__':
    app.run()
