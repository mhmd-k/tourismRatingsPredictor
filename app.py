from flask import Flask, jsonify, request
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


# Create a Flask web application
app = Flask(__name__)
CORS(app)


# Define a route for your API endpoint
@app.route("/predict", methods=["POST"])
def predict():
    # Load the model
    model = tf.keras.models.load_model("./content-based-filtering-model/model.keras")

    # load the datasets
    users = pd.read_csv("./data/users.csv")
    places = pd.read_csv("./data/all_places.csv", encoding="MacRoman")
    ratings = pd.read_csv("./data/ratings.csv")

    # Calculate the average rating for each place
    average_ratings = ratings.groupby("Place ID")["Rating"].mean()

    # Merge average ratings with the places DataFrame
    places = places.merge(average_ratings, left_on="id", right_index=True, how="left")

    # Rename the merged column to "average rating"
    places = places.rename(columns={"Rating": "Avg rating"})

    # Fill NaN values with a default value (e.g., 0)
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
            "shopping",
        ]
    ]
    Xp = merged_df[["place_type", "city_id", "Avg rating"]]

    label_encoder_gender = LabelEncoder()
    label_encoder_country = LabelEncoder()
    label_encoder_place_type = LabelEncoder()
    label_encoder_city = LabelEncoder()

    label_encoder_gender.fit(Xu["gender"])
    label_encoder_country.fit(Xu["country"])
    label_encoder_place_type.fit(Xp["place_type"])
    label_encoder_city.fit(Xp["city_id"])

    Xu["gender"] = label_encoder_gender.transform(Xu["gender"])
    Xu["country"] = label_encoder_country.transform(Xu["country"])
    Xp["place_type"] = label_encoder_place_type.transform(Xp["place_type"])
    Xp["city_id"] = label_encoder_city.transform(Xp["city_id"])

    # scale the data
    scalerItem = StandardScaler()
    scalerItem.fit(Xp)

    scalerUser = StandardScaler()
    scalerUser.fit(Xu)

    scalerTarget = MinMaxScaler((-1, 1))
    scalerTarget.fit(Y.reshape(-1, 1))

    user = request.get_json()

    print("user: ", user)

    gender = label_encoder_gender.transform([user["gender"]])
    country = label_encoder_country.transform([user["country"]])

    user_vec = np.array(
        [
            user["age"],
            gender[0],
            country[0],
            user["shopping"],
            user["natural"],
            user["old"],
            user["hotel"],
            user["night"],
            user["restaurant"],
        ]
    )

    # generate and replicate the user vector to match the number of places in the data set.
    user_vecs = np.tile(user_vec, (671, 1))
    user_vecs.shape

    new_places_df = places

    new_places_df["place_type"] = label_encoder_place_type.transform(
        new_places_df["place_type"]
    )
    new_places_df["city_id"] = label_encoder_city.transform(new_places_df["city_id"])

    item_vecs = np.array(places[["place_type", "city_id", "Avg rating"]])

    # scale our user and item vectors
    suser_vecs = scalerUser.transform(user_vecs)
    sitem_vecs = scalerItem.transform(item_vecs)

    # make a prediction
    y_p = model.predict([suser_vecs, sitem_vecs])

    # unscale y prediction
    y_pu = scalerTarget.inverse_transform(y_p)

    p = places
    p["predicted_rating"] = y_pu
    p_sorted = p.sort_values(by="predicted_rating", ascending=False)
    p_sorted["place_type"] = label_encoder_place_type.inverse_transform(
        p_sorted["place_type"]
    )
    p_sorted["city_id"] = label_encoder_city.inverse_transform(p_sorted["city_id"])

    # add the city_name
    city_mapping = {1: "roma", 2: "milan", 3: "napoli", 4: "florence", 5: "venice"}
    p_sorted["cityName"] = p_sorted["city_id"].map(city_mapping)

    # renaming some columns
    p_sorted.rename(columns={"city_id": "cityId"}, inplace=True)
    p_sorted.rename(columns={"Avg rating": "avgRating"}, inplace=True)
    p_sorted.rename(columns={"predicted_rating": "predictedRating"}, inplace=True)
    p_sorted.rename(columns={"place_type": "placeType"}, inplace=True)

    needed_df = pd.DataFrame()
    needed_df["id"] = p_sorted["id"]
    needed_df["name"] = p_sorted["name"]
    needed_df["cityId"] = p_sorted["cityId"]
    needed_df["placeType"] = p_sorted["placeType"]
    needed_df["cityName"] = p_sorted["cityName"]
    needed_df["predictedRating"] = p_sorted["predictedRating"]
    needed_df["time"] = p_sorted["time"].fillna(0)

    # Return the predictions as a JSON response
    return jsonify(needed_df.to_dict(orient="records"))


# .venv\Scripts\activate
