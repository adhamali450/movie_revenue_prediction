from operator import le
import requests
from sqlalchemy import null
from tmdbv3api import TMDb, Movie
from pyjsonq import JsonQ
import pandas as pd
import json


API_KEY = "b8656aad79d3af2e20690f7c808f7211"


def get_movie_id(api_key, movie_name):
    details_url = "https://api.themoviedb.org/3/search/movie?api_key={API_KEY}&query={movie_name}"
    movie_name = movie_name.strip().lower().replace(' ', '+')

    details_response = requests.get(
        details_url
        .replace('{API_KEY}', api_key)
        .replace('{movie_name}', movie_name)) \
        .json()

    try:
        return details_response['results'][0]['id']
    except:
        return 0


def get_movie_crew(api_key, movie_name):
    movie_id = get_movie_id(api_key, movie_name)

    if not movie_id:
        return 0

    tmdb = TMDb()
    tmdb.api_key = API_KEY
    movie = Movie()

    return eval("{\'crew\': " + str(movie.credits(movie_id)['crew']) + "}")


def get_movie_director(movie_name):
    movie_crew = get_movie_crew(API_KEY, movie_name)

    if not movie_crew:
        return 'Unknown'

    queryable_crew = JsonQ(data=movie_crew)
    director = queryable_crew.at('crew').where('job', '=', 'Director').get()

    return director[0]['name']


def get_movie_genre(movie_name):
    movie_id = get_movie_id(API_KEY, movie_name)

    if not movie_id:
        return 'Unknown'

    tmdb = TMDb()
    tmdb.api_key = API_KEY
    movie = Movie()

    return movie.details(movie_id)['genres'][0]['name']


def get_movie_mpaa(movie_name, year):
    header = {
        'accept': 'application/json',
        'Content-Type': 'application/x-www-form-urlencoded'
    }
    
    params = {
        'tt': movie_name
    }
    response = requests.get(
        "https://betterimdbot.herokuapp.com/", headers=header, params=params)
    return response.json()[1].keys()


