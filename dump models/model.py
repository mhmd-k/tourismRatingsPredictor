import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

users = pd.read_csv("drive/MyDrive/users.csv")
places = pd.read_csv("drive/MyDrive/all_places.csv", encoding="MacRoman")
ratings = pd.read_csv("drive/MyDrive/ratings.csv")

# Step 1: Calculate the average rating for each place
average_ratings = ratings.groupby("Place ID")["Rating"].mean()

# Step 2: Merge average ratings with the places DataFrame
places = places.merge(average_ratings, left_on="id", right_index=True, how="left")

# Step 3: Rename the merged column to "average rating"
places = places.rename(columns={"Rating": "Avg rating"})

# Step 4: Fill NaN values with a default value (e.g., 0)
places["Avg rating"] = places["Avg rating"].fillna(0)

merged_df = pd.merge(ratings, places, left_on="Place ID", right_on="id")
merged_df = pd.merge(merged_df, users, left_on="User ID", right_on="id")

# Calculate average rating per place_type per user
avg_ratings = (
    merged_df.groupby(["User ID", "place_type"])["Rating"].mean().reset_index()
)

# Pivot the data to create columns for each place_type
avg_ratings_pivot = avg_ratings.pivot(
    index="User ID", columns="place_type", values="Rating"
)

# Add the new columns to merged_df
merged_df = merged_df.merge(avg_ratings_pivot, on="User ID")

# Rename the columns
merged_df.rename(
    columns={
        "Shopping": "Shopping Avg",
        "Hotel": "Hotel Avg",
        "Night": "Night Avg",
        "Old": "Old Avg",
    },
    inplace=True,
)

Y = merged_df["Rating"].values
Xu = merged_df[
    [
        "age",
        "gender",
        "country",
        "hotel",
        "natural",
        "night",
        "old",
        "restaurant",
        "shooping",
    ]
]
Xp = merged_df[["place_type", "city_id", "Avg rating"]]

# Feature Extraction
label_encoder = LabelEncoder()
Xu["gender"] = label_encoder.fit_transform(Xu["gender"])
Xu["country"] = label_encoder.fit_transform(Xu["country"])
Xp["place_type"] = label_encoder.fit_transform(Xp["place_type"])
Xp["city_id"] = label_encoder.fit_transform(Xp["city_id"])

# Feature Scaling
item_train_unscaled = Xp
user_train_unscaled = Xu
y_train_unscaled = Y

scalerItem = StandardScaler()
scalerItem.fit(Xp)
item_train = scalerItem.transform(Xp)

scalerUser = StandardScaler()
scalerUser.fit(Xu)
user_train = scalerUser.transform(Xu)

scalerTarget = MinMaxScaler((-1, 1))
scalerTarget.fit(Y.reshape(-1, 1))
y_train = scalerTarget.transform(Y.reshape(-1, 1))

item_train, item_test = train_test_split(
    item_train, train_size=0.80, shuffle=True, random_state=1
)
user_train, user_test = train_test_split(
    user_train, train_size=0.80, shuffle=True, random_state=1
)
y_train, y_test = train_test_split(
    y_train, train_size=0.80, shuffle=True, random_state=1
)
print(f"place/item training data shape: {item_train.shape}")
print(f"user train data shape: {user_train.shape}")
print(f"y train data shape: {y_train.shape}")

tf.random.set_seed(1)
user_NN = keras.models.Sequential(
    [
        keras.layers.Dense(256, activation="relu"),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dense(32),
    ]
)

item_NN = tf.keras.models.Sequential(
    [
        keras.layers.Dense(256, activation="relu"),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dense(32),
    ]
)

# create the user input and point to the base network
input_user = tf.keras.layers.Input(shape=(9))
vu = user_NN(input_user)
vu = tf.linalg.l2_normalize(vu, axis=1)

# create the item input and point to the base network
input_item = tf.keras.layers.Input(shape=(3))
vm = item_NN(input_item)
vm = tf.linalg.l2_normalize(vm, axis=1)

# compute the dot product of the two vectors vu and vm
output = tf.keras.layers.Dot(axes=1)([vu, vm])

# specify the inputs and output of the model
model = tf.keras.Model([input_user, input_item], output)

cost_fn = tf.keras.losses.MeanSquaredError()
opt = keras.optimizers.Adam(learning_rate=0.01)
model.compile(optimizer="SGD", loss=cost_fn)

model.fit([user_train, item_train], y_train, epochs=30)

model.evaluate([user_test, item_test], y_test)  # 0.2685900926589966
