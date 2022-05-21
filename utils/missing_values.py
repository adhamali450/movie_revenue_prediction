import pandas as pd
import numpy as np
import movie_info


def fetch_missing_values(data_revenue, fetch_genres=True, fetch_directors=True):

    # Handing missing genres, directors
    print("Began filling missing values")

    data_revenue['genre'].fillna('NO_GENRE', inplace=True)
    data_revenue['director'].fillna('NO_DIRECTOR', inplace=True)

    counter = 0
    for movie in range(data_revenue.shape[0]):
        movie_name = data_revenue['movie_title'][movie].replace('…', '')
        print(str(counter) + ' ' + movie_name)
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
        animated = []
        genres = movie_info.get_animation_info(movie_name)
        if genres != "Unknown":
            if "#Animation" in genres:
                animated.append("YES")
            else:
                animated.append("NO")
        else:
            animated.append("NO")

    data_revenue["animated"] = animated
    fill_mpaa(data_revenue)

    print("Missing data filled")
    
    return data_revenue


def get_existing_directors(data_revenue):
    data_director = pd.read_csv('datasets/movie-director.csv')

    directors = [None] * data_revenue['movie_title'].count()

    for movie in range(0, data_revenue.shape[0]):
        movie_name = data_revenue['movie_title'][movie]
        for i in range(0, data_director.shape[0]):
            if data_director['name'][i] == movie_name and data_director['director'][i] != 'full credits':
                directors[movie] = data_director['director'][i]
                break

    data_revenue['director'] = directors


def fill_mpaa(data_revenue):
    data_revenue['MPAA_rating'].fillna('UNKNOWN', inplace=True)
    for movie in range(0, data_revenue.shape[0]):

        movie_name = data_revenue['movie_title'][movie].replace('…', '')
        if data_revenue['MPAA_rating'][movie] == 'UNKNOWN':
            mpaa = movie_info.get_movie_mpaa(movie_name)
            if mpaa == '15' or mpaa == 'A15':
                mpaa = 'PG'
            if mpaa == 'Approved' or mpaa == 'TV-G':
                mpaa = 'G'
            data_revenue['MPAA_rating'][movie] = mpaa

    return data_revenue



