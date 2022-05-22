from operator import le
import pandas as pd
import numpy as np
import movie_info


def fetch_missing_values(data_revenue, fetch_genres=True, fetch_directors=True):
    animated = []
    spread_date(data_revenue, 'release_date')
    # Handing missing genres, directors
    print("Began filling missing values")

    data_revenue['genre'].fillna('NO_GENRE', inplace=True)
    data_revenue['director'].fillna('NO_DIRECTOR', inplace=True)

    counter = 0
    for movie in range(data_revenue.shape[0]):
        movie_name = data_revenue['movie_title'][movie].replace('…', '')
        if get_range(data_revenue['year'][movie] ,movie_name) == True:
            print(movie , ' ' , movie_name)
            counter = counter + 1
            # Genres
            if fetch_genres:
                if data_revenue['genre'][movie] == 'NO_GENRE':
                    movie_genre = movie_info.get_movie_genre(movie_name)
                    data_revenue['genre'][movie] = movie_genre

            # Directors
            if fetch_directors:
                if data_revenue['director'][movie] == 'NO_DIRECTOR':
                    movie_director = movie_info.get_movie_director(movie_name)
                    data_revenue['director'][movie] = movie_director

            # is animated ?
            genres = movie_info.get_animation_info(movie_name)
            if genres != "Unknown":
                if "#Animation" in genres:
                    animated.append("YES")
                else:
                    animated.append("NO")
            else:
                animated.append("NO")

            print(len(animated))
    try:
        data_revenue["animated"] = pd.Series(animated)
    except:
        print("Error animated")

    try:
        fill_mpaa(data_revenue)
    except:
        print("Error mpaa")
    
    print("Missing data filled")
    
    return data_revenue


def get_existing_directors(data_revenue):
    data_director = pd.read_csv('datasets/movie-director.csv')

    directors = [None] * data_revenue['movie_title'].count()

    for movie in range(0, data_revenue.shape[0]):
        movie_name = data_revenue['movie_title'][movie].replace('…', '')
        for i in range(0, data_director.shape[0]):
            if data_director['name'][i] == movie_name and data_director['director'][i] != 'full credits':
                directors[movie] = data_director['director'][i]
                break

    data_revenue['director'] = directors


def fill_mpaa(data_revenue):
    data_revenue['MPAA_rating'].fillna('UNKNOWN', inplace=True)
    for movie in range(0, data_revenue.shape[0]):
        movie_name = data_revenue['movie_title'][movie].replace('…', '')
        if get_range(data_revenue['year'][movie] ,movie_name) == True:
            if data_revenue['MPAA_rating'][movie] == 'UNKNOWN':
                mpaa = movie_info.get_movie_mpaa(movie_name)
                if mpaa == '15' or mpaa == 'A15':
                    mpaa = 'PG'
                if mpaa == 'Approved' or mpaa == 'TV-G':
                    mpaa = 'G'
                data_revenue['MPAA_rating'][movie] = mpaa

    return data_revenue



def spread_date(df, col_name):
    df[col_name] = pd.to_datetime(df[col_name])
    df['year'] = df[col_name].dt.year
    df['month'] = df[col_name].dt.month
    df['day'] = df[col_name].dt.day

    new_year = []
    for year in df['year']:
        if year > 2016:
            year = 1900 + (year % 100)
            new_year.append(year)
        else:
            new_year.append(year)
    df['year'] = new_year


def get_range(year, name):
    try:
        date = int(movie_info.get_movie_year(name))
        if int(year) in range(date - 1, date + 2):
            return True
        else:
            return False
    except:
        return False