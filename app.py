from flask import Flask, render_template, request, jsonify, send_file
import sys
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pickle
import pyodbc
import cv2
import base64
import logging

app = Flask(__name__)

class Database:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
        self.tfidf_matrix = None
        self.vectorizer = None

    def load_data(self):
        try:
            self.data = pd.read_csv(self.file_path)
        except Exception as e:
            logging.error(f'Could not read file from {self.file_path}: {e}')
            raise

    def preprocess_data(self, fit=0):
        self.vectorizer = TfidfVectorizer(analyzer='char',ngram_range=(2,5), encoding="utf-16")
        if(fit):
            self.tfidf_matrix = self.vectorizer.fit_transform(self.data.apply(lambda x: ' '.join(x.astype(str)), axis=1))
            try:
                pickle.dump(self.tfidf_matrix, open("matrix.pickle", 'wb'))
            except Exception as e:
                logging.error(f'Error on saving matrix.pickle: {e}')
                raise
        else:
            try:
                self.tfidf_matrix = pickle.load(open("matrix.pickle",'rb'))
            except Exception as e:
                logging.error(f'Error on loading matrix.pickle: {e}')
                raise
        self.data['text'] = self.data[self.data.columns].astype(str).agg(' '.join, axis=1)

class SearchEngine:
    def __init__(self, database):
        self.database = database

    def search(self, query):
        query_vector = self.database.vectorizer.transform([query])
        similarity_scores = cosine_similarity(query_vector, self.database.tfidf_matrix)
        sorted_indices = np.argsort(similarity_scores[0])[::-1]

        # Create a boolean mask for similarity scores above 0
        mask = similarity_scores[0] > 0

        # Get the indices pointing to similarity scores above 0
        above_zero_indices = np.argwhere(mask).flatten()

        # Filter sorted_indices using the boolean mask
        sorted_indices_above_zero = sorted_indices[np.isin(sorted_indices, above_zero_indices)]

        # Optional: Select the top N indices
        top_n_indices = sorted_indices_above_zero[:]
        
        x = self.database.data.iloc[top_n_indices]

        keywords = query.split()
        
        matches = pd.Series(0, index=x.index) # initialize matches as a zero series with same indexes as dataframe 

        for keyword in keywords:
            matches += x["text"].str.contains(keyword, case=False).astype(int)
        logging.debug('checked for matches in the results')

        full_match = matches == len(keywords)
        any_match = matches > 0
            
        x["Similarity"] = np.char.add( (np.round( similarity_scores[0][top_n_indices] * 100, 2 ) ).astype('str'), '%')
        x["Any Match"] = any_match
        x["Full Match"] = full_match
        logging.debug('Added similarity, any match, and full match')
        return x

class ExtractEngine:
    def __init__(self, frame):
        self.frame = frame

    # This function needs to be changed
    def extract(self, table_name, date_range):
        try:
            self.cnn = pyodbc.connect('Driver={SQL Server};'
                        'Server=server_name;'
                        'Database=database_name;'
                        'Trusted_Connection=yes;')
            
            sql_query = f"select * from {table_name}"
            df = pd.read_sql_query(sql_query, self.cnn)
        except Exception as e:
            logging.error(f'Extraction unsuccessful: {e}')
            raise
        self.cnn.close()

        return df


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    query = request.form.get('query')
    logging.debug(f'received query: {query}')
    results = search_engine.search(query)
    return results.to_json(orient = 'records') # Add HTTPS status codes

@app.route('/extract', methods=['POST'])
def extract():
    data = request.get_json()
    logging.debug(f'Received data {data}')

    # extract_engine = ExtractEngine()
    # results = extract_engine.extract(table_name, [start_date, end_date]) # this needs to come back
    # return results.to_json(orient = 'records')

    logging.debug(f'extracted data')

    return send_file('data.csv',  as_attachment=True)

@app.route('/retrieve_image', methods=['POST'])
def retrieve_image(rows=None):
    img = cv2.imread('static/css/images/id.jpg')
    logging.debug('Retrieved image from database')

     # Encode the image to Base64 and prepare the response data
    _, buffer = cv2.imencode('.jpg', img)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    response_data = {'image_base64': img_base64}
    logging.debug('Converted image to base64')

    return jsonify(response_data)

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, filename='app.log', format='%(asctime)s - %(levelname)s - %(message)s')
    logging.debug('Program starting')
    
    data_file = sys.argv[1]
    fit = sys.argv[2]
    logging.debug('datafile and fit specified properly')

    db = Database(data_file)
    db.load_data()
    db.preprocess_data(fit)
    logging.debug('data loaded and preprocessed')

    search_engine = SearchEngine(db)
    
    app.run(debug=True)
