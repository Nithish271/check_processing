from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from model import testing,evaluate

app = Flask(__name__)

@app.route('/sample')
def running():
    return 'flask is running'

if __name__ == '__main__':
    app.debug = True
    app.run()
