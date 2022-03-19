import numpy as np
import pandas as pd
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

df = pd.read_csv("./dataset/steam_games.csv", sep=",")
print(df.head())

print(df.isnull().sum())
df = df.fillna("")
print(df["all_reviews"].apply(lambda x: x.split(",")[0]).value_counts())

df["all_reviews"] = df["all_reviews"].apply(lambda x: "Positive" if "Very Positive" in x.split(",")[0] or "Overwhelmingly Positive" in x.split(",")[0] else "Negative")
df = df[df["all_reviews"] == "Positive"]
df = df[["url", "name", "publisher", "popular_tags", "genre"]]
print(df.head())

df["popular_tags"] = df["popular_tags"].apply(lambda x: x.split(","))
df["genre"] = df["genre"].apply(lambda x: x.split(","))
df["publisher"] = df["publisher"].apply(lambda x: [x.split(",")[0]])
df["popular_tags"] = df["popular_tags"].apply(lambda x: [i.replace(" ","-") for i in x])
df["genre"] = df["genre"].apply(lambda x: [i.replace(" ","-") for i in x])
df["publisher"] = df["publisher"].apply(lambda x: [i.replace(" ","-") for i in x])
df["tags"] = df["name"].apply(lambda x: x.split(" ")) + (df["publisher"] + df["popular_tags"] + df["genre"]).apply(lambda x: list(set(x)))
print(df.head())

df = df[["url", "name", "tags"]]
df["tags"] = df["tags"].apply(lambda x: " ".join(x))
df["tags"] = df["tags"].apply(lambda x: x.lower())
print(df.head())

ps = nltk.stem.porter.PorterStemmer()
df["tags"] = df["tags"].apply(lambda x: " ".join([ps.stem(i) for i in x.split(" ")]))
df = df.reset_index(drop=True)
print(df.head())

cv = CountVectorizer(max_features=5000, stop_words="english")
vectors = cv.fit_transform(df["tags"]).toarray()
similarity = cosine_similarity(vectors)
print(similarity.shape)

print(sorted(list(enumerate(similarity[0])), key=lambda x: x[1], reverse=True)[1:11])
print(df.iloc[1154, :][0])

df_dict = df.to_dict()
pickle.dump(df_dict, open("./artifacts/games_dict.pkl", "wb"))
pickle.dump(similarity, open("./artifacts/similarity.pkl", "wb"))