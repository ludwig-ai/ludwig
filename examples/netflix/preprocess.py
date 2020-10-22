import os
from collections import deque

import pandas as pd

import ray
from ray.util.dask import ray_dask_get

import dask.dataframe as dd
from dask import delayed

# ray.init()

movie_titles = dd.read_csv('./data/movie_titles.csv',
                           encoding='ISO-8859-1',
                           header=None,
                           names=['Movie', 'Year', 'Name']).set_index('Movie')


def read_and_label_csv(filename):
    # Load single data-file
    df_raw = pd.read_csv(filename, header=None, names=['User', 'Rating', 'Date'], usecols=[0, 1, 2])

    # Find empty rows to slice dataframe for each movie
    tmp_movies = df_raw[df_raw['Rating'].isna()]['User'].reset_index()
    movie_indices = [[index, int(movie[:-1])] for index, movie in tmp_movies.values]

    # Shift the movie_indices by one to get start and endpoints of all movies
    shifted_movie_indices = deque(movie_indices)
    shifted_movie_indices.rotate(-1)

    # Gather all dataframes
    user_data = []

    # Iterate over all movies
    for [df_id_1, movie_id], [df_id_2, next_movie_id] in zip(movie_indices, shifted_movie_indices):

        # Check if it is the last movie in the file
        if df_id_1 < df_id_2:
            tmp_df = df_raw.loc[df_id_1 + 1:df_id_2 - 1].copy()
        else:
            tmp_df = df_raw.loc[df_id_1 + 1:].copy()

        # Create movie_id column
        tmp_df['Movie'] = movie_id

        # Append dataframe to list
        user_data.append(tmp_df)

    # Combine all dataframes
    df = pd.concat(user_data)
    del user_data, df_raw, tmp_movies, tmp_df, shifted_movie_indices, movie_indices, df_id_1, movie_id, df_id_2, next_movie_id

    return df


# create a list of functions ready to return a pandas.DataFrame
file_list = [f'./data/combined_data_{i + 1}.txt' for i in range(4)]
dfs = [delayed(read_and_label_csv)(fname) for fname in file_list]

# using delayed, assemble the pandas.DataFrames into a dask.DataFrame
ratings = dd.from_delayed(dfs)

ratings = ratings.repartition(12)
movie_titles = movie_titles.repartition(npartitions=1)
dataset = ratings.merge(movie_titles, how='inner', left_on='Movie', right_on='Movie')

res = dataset.to_parquet('./data/dataset.parquet', compute=True)
# res.compute(scheduler=ray_dask_get)
