from flask import Flask
from app.route import hello_world, index, classification, results

def create_app():
    app = Flask(__name__)
    app.add_url_rule('/', '/', hello_world)
    app.add_url_rule('/index', 'index', index)
    app.add_url_rule('/classification', 'classification', classification)
    app.add_url_rule('/classification/results', '/classification/results', results, methods=['POST'])
    return app