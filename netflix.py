# Importing pandas and matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Read in the Netflix CSV as a DataFrame
netflix_df = pd.read_csv("netflix_data.csv")
# Start coding here! Use as many cells as you like
list=netflix_df[np.logical_and(netflix_df[ "release_year"]>=1990,netflix_df[ "release_year"]<2000)]["duration"]

print(list)
plt.plot(list)
plt.show()


duration_list={}
for row in np.nditer(list):
    row = row.item()
    r=0
    if row in duration_list:
        continue
    for rows in np.nditer(list):
        if (row==rows.item()):
            r=r+1
    if (r>0):
        duration_list[row] = r
print(duration_list)
duration=0
k=0
for key,value in duration_list.items():
    if value>duration:
        duration=value
        k=key
print(duration)
short_movie_count=0
for row in np.nditer(list):
    row = row.item()
    if(row<=90):
        short_movie_count=short_movie_count+1
print(short_movie_count)
