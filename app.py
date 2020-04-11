import json
from flask import Flask, escape, request, json, jsonify, abort
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
    return "Hello world!"


@app.route('/api/v1/nbc_analyze', methods=['POST'])
def nbc_analyze():
    params = request.get_json()

    if 'phrase' in params:
        phrase = params['phrase']
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
    else:
        abort(400)


@app.route('/api/v1/nbc_evaluate', methods=['POST'])
def nbc_array():
    params = request.get_json()
    if 'phrase' in params:
        phrase = params['phrase']
        rows = list(
            map(lambda line: line.strip(),
                prepare_text(phrase)
                .split('.')
                )
        )
        evaluate = nbc.evaluate(rows).tolist()
        return jsonify(
            phrase=phrase,
            evaluate=evaluate
        )


if __name__ == "__main__":
    app.run()
