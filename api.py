from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import pickle
import sklearn.impute
from datetime import datetime
from assets_data_prep import prepare_data

app = Flask(__name__)

if not hasattr(sklearn.impute.SimpleImputer, '_fill_dtype'):
    sklearn.impute.SimpleImputer._fill_dtype = object

with open('trained_model.pkl', 'rb') as f:
    model = pickle.load(f)

EXPECTED_FIELDS = [
    'primaryTitle',
    'runtimeMinutes',
    'startYear',
    'num_lead_actors',
    'Language',
    'Country',
    'plot',
    'genres'
]

columns_to_exclude = [
    'tconst',
    'primaryTitle',
    'startYear',
    'genres',
    'num_genres',
    'lead_actors_ids',
    'Language',
    'Country',
    'plot',
    'budget',
    'averageRating',
    'numVotes',
    'BoxOffice'
]

TEMP_TOP_LISTS = {
    'countries': [
        'united states', 'india', 'united kingdom', 'france',
        'italy', 'japan', 'canada', 'germany', 'spain', 'hong kong'
    ],
    'languages': [
        'english', 'french', 'hindi', 'spanish', 'italian',
        'japanese', 'german', 'tamil', 'telugu', 'malayalam'
    ],
    'genres': [
        'drama', 'comedy', 'romance', 'action', 'documentary',
        'crime', 'thriller', 'horror', 'adventure', 'mystery',
        'family', 'fantasy'
    ]
}

def is_numeric(value):
    try:
        float(value)
        return True
    except (ValueError, TypeError):
        return False

def validate_inputs(data):
    for field in EXPECTED_FIELDS:
        if field not in data or data[field] is None or str(data[field]).strip() == '':
            return f"status 400 : Oops! Looks like {field} is missing.", 400

    if not is_numeric(data['startYear']):
        return "status 400 : A release year should be a number, try again.", 400

    start_year = float(data['startYear'])

    if start_year < 0:
        return "status 400 : A negative year, seriously?", 400

    if start_year < 100:
        return "status 400 : Movies didn't exist back in ancient times! Please enter a valid year.", 400

    if start_year <= 1887:
        return "status 400 : No movies before 1888 (The Roundhay Garden Scene).", 400

    if start_year > datetime.now().year:
        return "status 400 : We appreciate your optimism, but the movie hasn't been released yet!", 400

    if not is_numeric(data['runtimeMinutes']):
        return "status 400 : Runtime Minutes must be a numeric value.", 400

    runtime = float(data['runtimeMinutes'])

    if runtime <= 0:
        return "status 400 : Illogical movie runtime", 400

    if not is_numeric(data['num_lead_actors']):
        return "status 400 : Number of Lead Actors must be a numeric value.", 400

    actors = float(data['num_lead_actors'])

    if actors < 0:
        return "status 400 : A negative number of actors, seriously? Try again.", 400

    return None, None

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(silent=True)

    if data is None:
        return jsonify({"error": "status 400 : Oops! Looks like the request body is missing."}), 400

    error_msg, status = validate_inputs(data)
    if error_msg:
        return jsonify({"error": error_msg}), status

    try:
        num_actors = int(float(data['num_lead_actors']))
        lead_actors_ids = ','.join([f"actor_{i}" for i in range(num_actors)]) if num_actors > 0 else ''

        row = {
            'primaryTitle':    [data['primaryTitle']],
            'runtimeMinutes':  [float(data['runtimeMinutes'])],
            'startYear':       [float(data['startYear'])],
            'lead_actors_ids': [lead_actors_ids],
            'Language':        [data['Language'] if data['Language'] else 'Unknown'],
            'Country':         [data['Country'] if data['Country'] else 'Unknown'],
            'plot':            [data['plot'] if data['plot'] else ''],
            'genres':          [data['genres'] if data['genres'] else 'Unknown']
        }

        df = pd.DataFrame(row)
        df_prepared = prepare_data(df, top_lists=TEMP_TOP_LISTS)

        X = df_prepared.drop(columns=columns_to_exclude, errors='ignore')

        prediction = model.predict(X)
        predicted_rating = round(float(prediction[0]), 1)

        return jsonify({"predicted_rating": predicted_rating}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)





