from flask import Flask, render_template, request, jsonify, send_file
import sys
import pandas as pd
import numpy as np
import pickle
import pyodbc
import cv2
import base64
import logging
import time
from scipy.sparse import vstack
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from rank_bm25 import BM25Okapi
import nltk
from nltk.corpus import stopwords
import re
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from multiprocessing import Pool, cpu_count
import dask.dataframe as dd



app = Flask(__name__)

class Database:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
        self.index = None
        self.vectors = None
        self.model = SentenceTransformer('all-mpnet-base-v2')
        self.bm25 = None
        self.iterator = None
        self.lemmatizer = WordNetLemmatizer()
        self.stops = set(stopwords.words("english"))
 
    def load_data(self):
        try:
            # self.data = pd.read_csv(self.file_path)
            print(0)
            # self.data['text'] = self.data.apply(lambda x: ' '.join(x.astype(str)), axis=1)
            # self.data['text'] = self.data['Text'].apply(self.review_to_wordlist)
            
        except Exception as e:
            logging.error(f'Could not read file from {self.file_path}: {e}')
            raise

    def vectorize_data(self):
        self.iterator = pd.read_csv(self.file_path, chunksize=50000)
        self.index = faiss.index_factory(768, "Flat", faiss.METRIC_INNER_PRODUCT)
        self.bm25 = BM25Okapi()
        for i, df in enumerate(self.iterator):
            print(f"Batch: {i+1}")
            df['text'] = df['Text'].apply(self.review_to_wordlist)
            self.vectors = self.model.encode(df['text'].to_numpy()).astype("float32")
            faiss.normalize_L2(self.vectors)
            self.index.add(self.vectors)

            self.bm25.train(df['text'])
        self.bm25.finalize()
            

    def review_to_wordlist(self, review, remove_stopwords=True):
        # Clean the text, with the option to remove stopwords.
        
        # Convert words to lower case and split them
        words = word_tokenize(review.lower())

        # Optionally remove stop words (true by default)
        if remove_stopwords:
            words = [w for w in words if not w in self.stops]
        
        lemmatized_words = [self.lemmatizer.lemmatize(word) for word in words]
        
        review_text = " ".join(lemmatized_words)
        
        # Return a list of words
        return(review_text)

    def preprocess_data(self, fit=0):
        if(fit == '1'):
            try:
                self.vectorize_data()
                faiss.write_index(self.index, "index.faiss")
                with open('bm25_index.pkl', 'wb') as f:
                    pickle.dump(self.bm25, f)

            except Exception as e:
                logging.error(f'Error on saving index.faiss: {e}')
                raise
        else:
            try:
                self.index = faiss.read_index("index.faiss")
                with open('bm25_index.pkl', 'rb') as f:
                    self.bm25 = pickle.load(f)

            except Exception as e:
                logging.error(f'Error on loading index.faiss: {e}')
                raise
        self.data = pd.read_csv(self.file_path)

class SearchEngine:
    def __init__(self, database):
        self.database = database

    def search(self, query):
        tokenized_query = self.database.review_to_wordlist(query)
        query_vector = self.database.model.encode(tokenized_query).astype("float32")
        faiss.normalize_L2(np.array([query_vector]))
        
        D, I = self.database.index.search(np.array([query_vector]), 100)
        print("after index search")

        results = self.database.data.iloc[I[0]]

        bm25_scores = self.database.bm25.get_batch_scores(tokenized_query, I[0])
        bm25_scores = (bm25_scores-np.min(bm25_scores))/(np.max(bm25_scores)-np.min(bm25_scores))
        print("bm25 scores")
        # Here are 2 problems: 1- is that bm25 is taking the query instead of the tokenized query as is words withing the query

        keywords = tokenized_query.split()
        
        matches = pd.Series(0, index=results.index) # initialize matches as a zero series with same indexes as dataframe 

        for keyword in keywords:
            matches += results['Text'].str.contains(keyword, case=False).astype(int)
        matchez = ((matches-np.min(matches))/(np.max(matches)-np.min(matches))).to_numpy()
     
        fused_scores = ( D[0]*1.2 + bm25_scores + 0.8*matchez) / 3

        # Add scores column
        results = results.assign(score=fused_scores)

        # Sort by score
        results = results.sort_values('score', ascending=False)
        results["score"] = np.round(results["score"] * 100, 2)

        full_match = matches == len(keywords)
        any_match = matches > 0
            
        # x["Similarity"] = np.char.add( (np.round( similarity_scores[0][top_n_indices] * 100, 2 ) ).astype('str'), '%')
        results["Any Match"] = any_match
        results["Full Match"] = full_match
        return results
    

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
    print(query)
    results = search_engine.search(query)
    return results.to_json(orient = 'records') # Add HTTPS status codes

@app.route('/extract', methods=['POST'])
def extract():
    data = request.get_json()
    print(data)
    logging.debug(f'Received data {data}')

    # extract_engine = ExtractEngine()
    # results = extract_engine.extract(table_name, [start_date, end_date]) # this needs to come back
    # return results.to_json(orient = 'records')

    logging.debug(f'extracted data')

    return send_file('data.csv',  as_attachment=True)

if __name__ == '__main__':
    # nltk.download()
    t1 = time.time()
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
    t2 = time.time()
    print(t2-t1)
    app.run(debug=False)
