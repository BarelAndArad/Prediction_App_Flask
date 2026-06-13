import pandas as pd
import numpy as np
import re
import ast
from textblob import TextBlob

def clean_and_count_words(text):
    if pd.isna(text):
        return 0
    text = str(text).lower()
    clean_text = re.sub(r'[^a-z0-9\s]', '', text)
    return len(clean_text.split())

def parse_categories(text):
    if pd.isna(text):
        return []
    text = str(text).lower().strip()
    if text in ["", "[]", "\\n", "unknown", "not found", "none"]:
        return []
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, list):
            items = parsed
        else:
            items = [text]
    except:
        items = re.split(r"\||/|,| and ", text)
            
    cleaned = []
    for item in items:
        item = str(item).strip()
        item = re.sub(r"\[.*?\]", "", item)
        item = re.sub(r"\(.*?\)", "", item)
        item = item.strip()
        if item and item not in ["unknown", "not found", "none"]:
            cleaned.append(item)
    return cleaned

def get_sentiment(text):
    if pd.isna(text) or str(text).strip() == "":
        return 0.0, 0.0
    blob = TextBlob(str(text))
    return blob.sentiment.polarity, blob.sentiment.subjectivity

def clean_target_variable(df):

    df = df.copy()

    if 'averageRating' in df.columns:

        df['averageRating'] = pd.to_numeric(
            df['averageRating'],
            errors='coerce'
        )

        df = df[
            (df['averageRating'] >= 1.0) &
            (df['averageRating'] <= 10.0)
        ]

    return df

def normalize_column_names(df):

    df = df.copy()

    for col in df.columns:

        if col.lower() == 'genres' and col != 'genres':

            df = df.rename(
                columns={col: 'genres'}
            )

    return df

def fill_basic_missing_values(df):

    df = df.copy()

    fill_values = {
        'Country': 'Unknown',
        'Language': 'Unknown',
        'plot': '',
        'genres': 'Unknown',
        'primaryTitle': 'Unknown'
    }

    for col, value in fill_values.items():

        if col in df.columns:

            df[col] = df[col].fillna(value)

    if 'lead_actors_ids' in df.columns:

        df['lead_actors_ids'] = (
            df['lead_actors_ids']
            .apply(
                lambda x: [] if pd.isna(x) else x
            )
        )

    return df

def create_plot_features(df):

    df = df.copy()

    if 'plot' in df.columns:

        df['plot_word_count'] = (
            df['plot']
            .apply(clean_and_count_words)
        )

        sentiments = (
            df['plot']
            .apply(get_sentiment)
        )

        df['plot_polarity'] = [
            s[0] for s in sentiments
        ]

        df['plot_subjectivity'] = [
            s[1] for s in sentiments
        ]

    return df

def create_decade_feature(df):

    df = df.copy()

    if 'startYear' in df.columns:

        df['startYear'] = pd.to_numeric(
            df['startYear'],
            errors='coerce'
        )

        df.loc[
            (
                (df['startYear'] < 1874) |
                (df['startYear'] > 2030)
            ),
            'startYear'
        ] = np.nan

        df['decade'] = (
            (df['startYear'] // 10) * 10
        )

        df['decade'] = (
            df['decade']
            .fillna(0)
        )

    return df

def count_lead_actors(value):

    if isinstance(value, list):
        return len(value)

    text = str(value).strip()

    invalid_values = [
        '',
        '[]',
        '\\N',
        'nan',
        'None'
    ]

    if text in invalid_values:
        return 0

    try:

        parsed = ast.literal_eval(text)

        if isinstance(parsed, list):
            return len(parsed)

    except:
        pass

    if ',' in text:

        return len([
            i for i in text.split(',')
            if i.strip() != ''
        ])

    if ' ' in text:

        return len([
            i for i in text.split()
            if i.strip() != ''
        ])

    return 0

def create_lead_actors_feature(df):

    df = df.copy()

    if 'lead_actors_ids' in df.columns:

        df['num_lead_actors'] = (
            df['lead_actors_ids']
            .apply(count_lead_actors)
        )

    return df

def count_genres(value):

    if isinstance(value, list):
        return len(value)

    text = str(value).strip()

    invalid_values = [
        '',
        '[]',
        '\\N',
        'Unknown'
    ]

    if text in invalid_values:
        return 0

    try:

        parsed = ast.literal_eval(text)

        if isinstance(parsed, list):
            return len(parsed)

    except:
        pass

    if ',' in text:

        split_values = [
            item.strip()
            for item in text.split(',')
            if item.strip() != ''
        ]

        return len(split_values)

    if ' ' in text:

        split_values = [
            item.strip()
            for item in text.split()
            if item.strip() != ''
        ]

        return len(split_values)

    return 1

def create_genres_count_feature(df):

    df = df.copy()

    if 'genres' in df.columns:

        df['num_genres'] = (
            df['genres']
            .apply(count_genres)
        )

    return df

def get_top_category_lists(df, top_lists=None):

    if top_lists is None:

        top_countries = (
            df['Country_parsed']
            .explode()
            .value_counts()
            .head(10)
            .index
            .tolist()
        )

        top_languages = (
            df['Language_parsed']
            .explode()
            .value_counts()
            .head(10)
            .index
            .tolist()
        )

        top_genres = (
            df['genres_parsed']
            .explode()
            .value_counts()
            .head(12)
            .index
            .tolist()
        )

        current_top_lists = {
            'countries': top_countries,
            'languages': top_languages,
            'genres': top_genres
        }

    else:

        current_top_lists = top_lists

    return current_top_lists

def create_country_features(df, top_countries):

    df = df.copy()

    if 'Country_parsed' in df.columns:

        created_country_cols = []

        for country in top_countries:

            col_name = f"In_{country.replace('-', '_').replace(' ', '_').title()}"

            created_country_cols.append(col_name)

            df[col_name] = df['Country_parsed'].apply(
                lambda x: 1 if isinstance(x, list) and country in x else 0
            )

        if created_country_cols:

            df['In_Other_Country'] = (
                df[created_country_cols]
                .sum(axis=1)
                .eq(0)
                .astype(int)
            )

        df = df.drop(columns=['Country_parsed'])

    return df

def create_language_features(df, top_languages):

    df = df.copy()

    if 'Language_parsed' in df.columns:

        for lang in top_languages:

            col_name = f"is_lang_{lang.replace('-', '_').replace(' ', '_')}"

            df[col_name] = df['Language_parsed'].apply(
                lambda x: 1 if isinstance(x, list) and lang in x else 0
            )

        df = df.drop(columns=['Language_parsed'])

    return df

def create_genre_features(df, top_genres):

    df = df.copy()

    if 'genres_parsed' in df.columns:

        created_genre_cols = []

        for genre in top_genres:

            col_name = (
                f"is_genre_{genre.replace('-', '_').replace(' ', '_')}"
            )

            created_genre_cols.append(col_name)

            df[col_name] = (
                df['genres_parsed']
                .apply(
                    lambda x: 1 if isinstance(x, list) and genre in x else 0
                )
                * df['decade']
            )

        if created_genre_cols:

            df['is_genre_other'] = (
                (
                    df[created_genre_cols]
                    .sum(axis=1)
                    == 0
                )
                .astype(int)
                * df['decade']
            )

        df = df.drop(columns=['genres_parsed'])

    return df

def create_parsed_category_columns(df):

    df = df.copy()

    if 'Country' in df.columns:
        df['Country_parsed'] = df['Country'].apply(parse_categories)

    if 'Language' in df.columns:
        df['Language_parsed'] = df['Language'].apply(parse_categories)

    if 'genres' in df.columns:
        df['genres_parsed'] = df['genres'].apply(parse_categories)

    return df

def runtime_category(runtime):

    if pd.isna(runtime):
        return 'Unknown'

    if runtime < 60:
        return 'Short'

    elif runtime <= 120:
        return 'Standard'

    else:
        return 'Long'
    
def fit_top_lists(df_train):
    return get_top_category_lists(
        create_parsed_category_columns(
            fill_basic_missing_values(
                normalize_column_names(
                    clean_target_variable(df_train.copy())
                )
            )
        )
    )

def prepare_data(df, top_lists=None):

    if top_lists is None:
        top_lists = fit_top_lists(df)

    df_clean = df.copy()

    df_clean = clean_target_variable(df_clean)
    df_clean = normalize_column_names(df_clean)
    df_clean = fill_basic_missing_values(df_clean)
    df_clean = create_plot_features(df_clean)
    df_clean = create_decade_feature(df_clean)

    if 'lead_actors_ids' in df_clean.columns:
        df_clean['num_lead_actors'] = (
            df_clean['lead_actors_ids']
            .apply(count_lead_actors)
        )

    df_clean = create_genres_count_feature(df_clean)
    df_clean = create_parsed_category_columns(df_clean)

    top_countries = top_lists['countries']
    top_languages = top_lists['languages']
    top_genres = top_lists['genres']

    df_clean = create_country_features(df_clean, top_countries)
    df_clean = create_language_features(df_clean, top_languages)
    df_clean = create_genre_features(df_clean, top_genres)

    if 'runtimeMinutes' in df_clean.columns:

        df_clean['runtimeMinutes'] = pd.to_numeric(
            df_clean['runtimeMinutes'],
            errors='coerce'
        )

        df_clean.loc[
            (df_clean['runtimeMinutes'] <= 0) |
            (df_clean['runtimeMinutes'] > 420),
            'runtimeMinutes'
        ] = np.nan

        df_clean['runtime_category'] = (
            df_clean['runtimeMinutes']
            .apply(runtime_category)
        )

    return df_clean

