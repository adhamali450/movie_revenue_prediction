import pandas as pd
import movie_info

def fetch_missing_values(fetch_genres=True, fetch_directors=True):
    data_revenue = pd.read_csv('datasets/movies-revenue.csv')

    # Handing missing genres, directors

    print("Began filling missing values")

    directors = []
    animated = []
    data_revenue['genre'].fillna('NO_GENRE', inplace=True)
    for movie in range(data_revenue.shape[0]):
        movie_name = data_revenue['movie_title'][movie].replace('…', '')
        # Genres
        if fetch_genres:
            if data_revenue['genre'][movie] == 'NO_GENRE':
                movie_genre = movie_info.get_movie_genre(movie_name)
                data_revenue['genre'][movie] = movie_genre
        # Directors
        if fetch_directors:
            movie_director = movie_info.get_movie_director(movie_name)
            directors.append(movie_director)

        genres = movie_info.get_animation_info(movie_name)
        print(movie_name)
        print(genres)
        if genres != "Unknown":
            if "#Animation" in genres:
                animated.append("Yes")
            else:
                animated.append("NO")
        else:
            animated.append("Unknown")

    if fetch_directors:
        data_revenue['directors'] = directors

    data_revenue["animated"] = animated
    fill_mpaa(data_revenue)
    print("Missing data filled")
    
    return data_revenue



def fill_diff_directors(data_revenue):
    data = pd.read_csv('datasets/movie-director.csv')

    for movie in range(0, data_revenue.shape[0]):
        movie_name = data_revenue['movie_title'][movie].replace('…', '')
        for i in range(0, data.shape[0]):
            if data['name'][i] == movie_name and data['director'][i] != 'full credits':
                data_revenue['directors'][movie] = data['director'][i]
                break
        
    
    return data_revenue

def fill_mpaa(data_revenue):
    data_revenue['MPAA_rating'] = data_revenue['MPAA_rating'].fillna('Unknown')
    for movie in range(0, data_revenue.shape[0]):

        movie_name = data_revenue['movie_title'][movie].replace('…', '')
        if data_revenue['MPAA_rating'][movie] == 'Unknown':
            mpaa = movie_info.get_movie_mpaa(movie_name)
            if mpaa == '15' or mpaa == 'A15':
                mpaa = 'PG'
            if mpaa == 'Approved' or mpaa == 'TV-G' :
                mpaa = 'G'
            data_revenue['MPAA_rating'][movie] = mpaa
            print(mpaa)
    return data_revenue



