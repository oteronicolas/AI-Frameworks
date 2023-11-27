from annoy import AnnoyIndex
from flask import Flask, jsonify, request

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
 data = request.json
 df = data.get('dataframe')
 index_query = data.get('idx')
 k = data.get('n_reco')
 dim = 576
 annoy_index = AnnoyIndex(dim, 'angular')
 annoy_index.load('rec_imdb.ann')
 indices = annoy_index.get_nns_by_vector(df.iloc[index_query], k)
 return jsonify({"prediction": indices})



app.run(host='0.0.0.0', port=5000, debug=False)