from flask import Flask, escape, request, json, jsonify
from flask_restful import Resource, Api, reqparse
from flask_cors import CORS, cross_origin

from nbc import NBC, prepare_text

app = Flask(__name__)
api = Api(app)

cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

# NaiveBayesianClassifier
nbc = NBC()


@app.route('/')
def hello():
    name = request.args.get("name", "World")
    return f'Hello, {escape(name)}!'


@app.route('/api/v1/nbc_analyze')
def nbc_analyze():
    phrase = request.args.get("phrase", "hi there I\'m an API")

    rows = list(
        map(lambda line: line.strip(),
            prepare_text(phrase)
            .split('.')
            )
    )
    prediction = nbc.prediction(rows)

    return jsonify(
        phrase=phrase,
        result=prediction
    )


if __name__ == "__main__":
    app.run()
