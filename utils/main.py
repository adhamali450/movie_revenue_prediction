from missing_values import *
import movie_info
import pandas as pd

# missing_values. \
#     get_existing_directors(missing_values.fetch_missing_values()) \
#     .to_csv("[P2-MERGED-COMPLETE]movies_revenue.csv", index=False)

data_revenue = pd.read_csv('./datasets/movies-revenue-classification.csv')
get_existing_directors(data_revenue)
fetch_missing_values(data_revenue)

data_revenue.to_csv('./[P2-MERGED-COMPLETE]movies_revenue.csv')
