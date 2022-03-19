import streamlit as st
import pandas as pd
import pickle

games_df = pd.DataFrame(pickle.load(open("./artifacts/games_dict.pkl", "rb")))
similarity = pickle.load(open("./artifacts/similarity.pkl", "rb"))

def recommend(game):
    game_index = games_df[games_df["name"] == game].index[0]
    recommendations = sorted(list(enumerate(similarity[game_index])), key=lambda x: x[1], reverse=True)[1:11]
    recommendations = list(map(lambda x: (games_df.iloc[x[0], :][1], games_df.iloc[x[0], :][0]), recommendations))
    return recommendations

games_list = games_df["name"].values

st.title("Game Recommender System")

selected_game = st.selectbox("Select Game", games_list)

if st.button("Recommend"):
    for game in recommend(selected_game):
        st.subheader(f"{game[0]}   [:video_game:]({game[1]})")