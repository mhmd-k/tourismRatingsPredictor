import random
import pandas as pd

places = pd.read_csv('data/all_places.csv',encoding='MacRoman')
users = pd.read_csv('data/users.csv')
rating_probability = 0.1  # Probability for each user-place combination being rated

ratings = []

for user in range(1, len(users) + 1):
    for place in range(1, len(places) + 1):
        if random.random() <= rating_probability:
            rating = random.choices(
                population=[1, 2, 3, 4, 5],
                weights=[0.05, 0.10, 0.25, 0.35, 0.25],
                k=1
            )[0]  # Select a rating based on the given weights

            user_age = users.loc[user - 1, 'age']
            place_type = places.loc[place - 1, 'place_type']
            user_gender = users.loc[user - 1, 'gender']

            if user_age > 50 and place_type == 'old':
                rating = random.choices(
                    population=[4, 5],
                    weights=[0.5, 0.5],
                    k=1
                )[0]  # Set rating to 4 or 5
            elif user_gender == 'Female' and place_type == 'shopping':
                rating = random.choices(
                    population=[4, 5],
                    weights=[0.5, 0.5],
                    k=1
                )[0]  # Set rating to 4 or 5
            elif user_gender == 'Male' and user_age > 20 and user_age <= 30 and place_type == 'night':
                rating = random.choices(
                    population=[4, 5],
                    weights=[0.5, 0.5],
                    k=1
                )[0]  # Set rating to 4 or 5

            row = {'User ID': user, 'Place ID': place, 'Rating': rating}
            ratings.append(row)

df = pd.DataFrame(ratings)

# Save the dataset to a CSV file
df.to_csv('ratings.csv', index=False)