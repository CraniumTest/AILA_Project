from flask import Flask, request, jsonify
from transformers import pipeline
from sklearn.cluster import KMeans
import pandas as pd

# Initialize Flask app
app = Flask(__name__)

# Load pre-trained language model
qa_pipeline = pipeline('question-answering', model="distilbert-base-cased-distilled-squad")

# Dummy dataset for learning path
data = pd.DataFrame({
    'topic': ['math', 'science', 'history', 'arts'],
    'difficulty': [1, 2, 3, 2]
})

# KMeans clustering for learning path determination
kmeans = KMeans(n_clusters=2, random_state=0).fit(data[['difficulty']])

@app.route('/ask', methods=['POST'])
def ask():
    data = request.json
    question = data.get('question', '')
    context = data.get('context', 'Generic context')
    result = qa_pipeline({'question': question, 'context': context})
    return jsonify(result)

@app.route('/learning-path', methods=['POST'])
def learning_path():
    user_data = request.json
    user_scores = user_data.get('scores', [0])
    cluster = kmeans.predict([user_scores])
    return jsonify({'path': 'Beginner' if cluster[0] == 0 else 'Advanced'})

if __name__ == '__main__':
    app.run()
