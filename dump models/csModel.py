import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load the datasets
users = pd.read_csv('drive/MyDrive/users.csv')
places = pd.read_csv('drive/MyDrive/all_places.csv', encoding='MacRoman')
ratings = pd.read_csv('drive/MyDrive/ratings.csv')

# Merge datasets based on user and place IDs
merged_data = ratings.merge(places, left_on='Place ID', right_on='id').merge(users, left_on='User ID', right_on='id')

# Prepare the data for modeling
X = merged_data[['age', 'gender', 'country']]
y = merged_data['Rating']

# Encode categorical variables
label_encoder = LabelEncoder()
X.loc[:, 'gender'] = label_encoder.fit_transform(X.loc[:, 'gender'])
X.loc[:, 'country'] = label_encoder.fit_transform(X.loc[:, 'country'])

# Scale numerical features
scaler = StandardScaler()
X.loc[:, ['age', 'country']] = scaler.fit_transform(X.loc[:, ['age', 'country']])

# Calculate pairwise cosine similarity between user profiles
similarity_matrix = cosine_similarity(X)

# Function to predict user ratings based on similarity
def predict_rating(user_id, place_id):
    user_index = merged_data[merged_data['User ID'] == user_id].index[0]
    place_index = merged_data[merged_data['Place ID'] == place_id].index[0]
    user_similarities = similarity_matrix[user_index]
    user_ratings = merged_data['Rating']
    weighted_ratings = user_similarities * user_ratings
    if user_similarities.sum() == 0:
        return None
    else:
        return weighted_ratings.sum() / user_similarities.sum()

# Example usage
user_id = 3 # ID of the user for whom you want to predict ratings
place_id =   # ID of the place for which you want to predict the user's rating
predicted_rating = predict_rating(user_id, place_id)
print(f'Predicted rating for user {user_id} and place {place_id}: {predicted_rating}')