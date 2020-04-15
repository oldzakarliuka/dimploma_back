import json
from flask import Flask, escape, request, json, jsonify, abort
from flask_cors import CORS, cross_origin

# import tensorflow as tf
# import keras


# session = keras.backend.get_session()
# init = tf.global_variables_initializer()
# session.run(init)
# graph = tf.get_default_graph()

from nbc import NBC, prepare_text
#from rnn import RNN
from wtv import WTV, prepare_word

app = Flask(__name__)

cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

# NaiveBayesianClassifier
nbc = NBC()

#rnn = RNN()

wtv = WTV()

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

# @app.route('/api/v1/evaluate', methods=['POST'])
# def rrn_array():
#     global graph
#     with session.as_default():
#         with graph.as_default():
#             params = request.get_json()
#             if 'phrase' in params:
#                 phrase = params['phrase']
#                 rows = list(
#                     map(lambda line: line.strip(),
#                         prepare_text(phrase)
#                         .split('.')
#                         )
#                 )
#                 evaluate = rnn.evaluate(rows)
#                 return jsonify(
#                     phrase=phrase,
#                     evaluate=evaluate

#                 )
#             else: 
#                 abort(400)

# @app.route('/api/v1/analyze', methods=['POST'])
# def rrn_analyze():
#     global graph
#     with session.as_default():
#         with graph.as_default():
#             params = request.get_json()

#             if 'phrase' in params:
#                 phrase = params['phrase']
#                 rows = list(
#                     map(lambda line: line.strip(),
#                         prepare_text(phrase)
#                         .split('.')
#                         )
#                 )
#                 prediction = rnn.prediction(rows)
#                 return jsonify(
#                     phrase=phrase,
#                     result=prediction
#                 )
#             else:
#                 abort(400)

@app.route('/api/v1/similarity')
def wtv_similarity():
    first = request.args.get('first')
    sec = request.args.get('sec')

    if first and sec:
        f = prepare_word(first)
        s = prepare_word(sec)

        return jsonify(
            first=first,
            sec=sec,
            similarity=wtv.similarity((f,s))
        )
    else:
        abort(400)

@app.route('/api/v1/closestcosmul', methods=['POST'])
def wtv_closest_cosmul():
    params = request.get_json()

    print(params)

    if 'words' in params:
        words = [prepare_word(word) for word in params['words']]
        
        prediction = wtv.closest_cosmul(words=words)
        return jsonify(
            words=words,
            result=prediction
        )
    elif 'pos' in params and 'neg' in params:
        pos = [prepare_word(word) for word in params['pos']]
        neg = [prepare_word(word) for word in params['neg']]

        prediction = wtv.closest_cosmul(pos=pos,neg=neg)
        return jsonify(
            pos=pos,
            neg=neg,
            result=prediction
        )
      
    elif 'pos' in params:
        pos = [prepare_word(word) for word in params['pos']]

        prediction = wtv.closest_cosmul(pos=pos)
        return jsonify(
            pos=pos,
            result=prediction
        )
    elif 'neg' in params:
        neg = [prepare_word(word) for word in params['neg']]

        prediction = wtv.closest_cosmul(neg=neg)

        return jsonify(
            neg=neg,
            result=prediction
        )
    else:
        abort(400)

@app.route('/api/v1/closest', methods=['POST'])
def wtv_closest():
    params = request.get_json()

    print(params)

    if 'words' in params:
        words = [prepare_word(word) for word in params['words']]
        
        prediction = wtv.closest(words=words)
        return jsonify(
            words=words,
            result=prediction
        )
    elif 'pos' in params and 'neg' in params:
        pos = [prepare_word(word) for word in params['pos']]
        neg = [prepare_word(word) for word in params['neg']]

        prediction = wtv.closest(pos=pos,neg=neg)
        return jsonify(
            pos=pos,
            neg=neg,
            result=prediction
        )
      
    elif 'pos' in params:
        pos = [prepare_word(word) for word in params['pos']]

        prediction = wtv.closest(pos=pos)
        return jsonify(
            pos=pos,
            result=prediction
        )
    elif 'neg' in params:
        neg = [prepare_word(word) for word in params['neg']]

        prediction = wtv.closest(neg=neg)

        return jsonify(
            neg=neg,
            result=prediction
        )
    else:
        abort(400)

if __name__ == "__main__":
    app.run()


