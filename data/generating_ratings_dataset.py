import random
import pandas as pd

users = 500
places = 671
rating_probability = 0.05  # Probability for each user-place combination being rated

ratings = []

for user in range(1, users + 1):
    for place in range(1, places + 1):
        if random.random() <= rating_probability:
            rating = random.choices(
                population=[1, 2, 3, 4, 5],
                weights=[0.05, 0.10, 0.35, 0.3, 0.2],
                k=1
            )[0]  # Select a rating based on the given weights
            row = {'User ID': user, 'Place ID': place, 'Rating': rating}
            ratings.append(row)

df = pd.DataFrame(ratings)

# Save the dataset to a CSV file
df.to_csv('ratings.csv', index=False)