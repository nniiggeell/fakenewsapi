# -*- coding: utf-8 -*-
"""
Created on Tue Jul 20 23:40:52 2021

@author: agent
"""

from flask import Flask, jsonify, request
import pickle
import pandas as pd
import string

def punctuation_removal(text):
    all_list = [char for char in text if char not in string.punctuation]
    clean_str = ''.join(all_list)
    return clean_str
# load model
model = pickle.load(open('model.pkl','rb'))

# app
app = Flask(__name__)

# routes
@app.route('/api', methods=['POST'])
def predict():
    # get data
    data = request.get_json(force = True)
    
    lines2 = []
    lines2.append(data)
    df = pd.DataFrame({'text' : lines2}).astype(str)
    df['text'] = df['text'].apply(lambda x: x.lower())
    input_df = df.apply(punctuation_removal)
    
    # predictions
    result = model.predict(input_df)
    
    # send back to browser
    output = result[0]

    # return data
    return jsonify(output)

if __name__ == '__main__':
    app.run(port = 5000, debug = True)
    
    
"""
# routes
@app.route('/', methods=['POST'])

def predict():
    # get data
    data = request.get_json(force = True)

    # convert data into dataframe
    data.update((x, [y]) for x, y in data.items())
    data_df = pd.DataFrame.from_dict(data)

    # predictions
    result = model.predict(data_df)

    # send back to browser
    output = {'results': int(result[0])}

    # return data
    return jsonify(results=output)

"""
