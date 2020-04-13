import json
from flask import Flask, escape, request, json, jsonify, abort
from flask_restful import Resource, Api, reqparse
from flask_cors import CORS, cross_origin

import tensorflow as tf
import keras


session = keras.backend.get_session()
init = tf.global_variables_initializer()
session.run(init)
graph = tf.get_default_graph()

from nbc import NBC, prepare_text
from rnn import RNN

app = Flask(__name__)
api = Api(app)

cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

# NaiveBayesianClassifier
nbc = NBC()

rnn = RNN()

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
    else:
        abort(400)

@app.route('/api/v1/evaluate', methods=['POST'])
def rrn_array():
    global graph
    with session.as_default():
        with graph.as_default():
            params = request.get_json()
            if 'phrase' in params:
                phrase = params['phrase']
                rows = list(
                    map(lambda line: line.strip(),
                        prepare_text(phrase)
                        .split('.')
                        )
                )
                evaluate = rnn.evaluate(rows)
                return jsonify(
                    phrase=phrase,
                    evaluate=evaluate

                )
            else: 
                abort(400)

@app.route('/api/v1/analyze', methods=['POST'])
def rrn_analyze():
    global graph
    with session.as_default():
        with graph.as_default():
            params = request.get_json()

            if 'phrase' in params:
                phrase = params['phrase']
                rows = list(
                    map(lambda line: line.strip(),
                        prepare_text(phrase)
                        .split('.')
                        )
                )
                prediction = rnn.prediction(rows)
                return jsonify(
                    phrase=phrase,
                    result=prediction
                )
            else:
                abort(400)



if __name__ == "__main__":
    app.run()
