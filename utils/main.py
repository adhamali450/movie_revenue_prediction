import missing_values
import pandas as pd

missing_values.fill_diff_directors(missing_values.fetch_missing_values()).to_csv("[MERGED-COMPLETE]movies_revenue.csv")

