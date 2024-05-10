import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler 
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras.metrics import MeanAbsoluteError
import matplotlib.pyplot as plt

# Load the datasets
users = pd.read_csv('drive/MyDrive/users.csv')
places = pd.read_csv('drive/MyDrive/all_places.csv', encoding='MacRoman')
ratings = pd.read_csv('drive/MyDrive/ratings.csv')

# Merge the datasets based on User ID and Place ID
combined_data = ratings.merge(places, left_on='Place ID', right_on='id').merge(users, left_on='User ID', right_on='id')

# Select the relevant features for the model
features = ['age', 'gender', 'country','Place ID', 'place_type', 'city_id']
target = 'Rating'  # Assuming the target variable is named 'Rating'

# Preprocess the data
le = LabelEncoder()
combined_data['gender'] = le.fit_transform(combined_data['gender'])
combined_data['country'] = le.fit_transform(combined_data['country'])
combined_data['place_type'] = le.fit_transform(combined_data['place_type'])

# Split the data into training and testing sets
train_data, test_data, train_labels, test_labels = train_test_split(combined_data[features], combined_data[target], test_size=0.2, random_state=42)

# Normalize the features
scaler = StandardScaler()
train_data = scaler.fit_transform(train_data)
test_data = scaler.transform(test_data)

# Build the neural network model
model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(len(features),)))
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='linear'))

# Compile the model
model.compile(loss='mse', optimizer=Adam(learning_rate=0.01), metrics=[MeanAbsoluteError()])

# Train the model
model.fit(train_data, train_labels, batch_size=32, epochs=10, validation_data=(test_data, test_labels))

# Evaluate the model
eval_loss, eval_mae = model.evaluate(test_data, test_labels)
print("Evaluation Loss:", eval_loss) # Evaluation Loss: 1.152194619178772
print("Evaluation MAE:", eval_mae) # Evaluation MAE: 0.9022572040557861