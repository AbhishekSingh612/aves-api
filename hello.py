from flask import Flask, request, render_template
app = Flask(__name__)


@app.route('/<user>/profile', methods=['GET'])
def hello_world(user):
    return 'hello' + user


@app.route('/', methods=['POST'])
def hello_post():
    user = request.form['nm']
    return '<h1>Hello GET </h1><br>' + user


if __name__ == '__main__':
    app.run()
