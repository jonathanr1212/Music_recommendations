import spotipy
import json
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pickle
import numpy as np
import streamlit as st

header = st.container()
dataset = st.container()
permissions = st.container()
modeling = st.container()
outputs = st.container()

with header:
    st.title('Welcome to the Spotify Recommendation Engine!')
    st.write("This webpage will allow you to enter a song into it and produce recommendations based on your song.")

             
             

with dataset:
             spot_df = pd.read_csv('data/spotify_playlist.csv')
             spot_df.head()
                       
                       
                       
with permissions:
    def get_keys(path):
        with open("/Users/Jonathan/Documents/Flatiron/phase_5/P5_spotify_recommendations/.secret/spotify_api.json") as f:
            return json.load(f)
                    
    keys = get_keys("/Users/Jonathan/.secret/spotify_api.json")
    client = keys['client']
    api_key = keys['api_key']
    auth_manager = SpotifyClientCredentials(client_id = client, client_secret = api_key)
    sp = spotipy.Spotify(auth_manager=auth_manager) 
                       
with modeling:
     with open('finalized_model.pkl', 'rb') as f:
            model = pickle.load(f)
form = st.form(key='my_form')
song =form.text_input("Song title: ", 'jimmy cooks')
artist = form.text_input("Artist name: ", 'drake')
submit_button = form.form_submit_button(label='Submit')

def get_song(song, artist):
        playlist_features_list = ["artist", "artist_id", "popularity", "album", "track_name", "track_id", 
                             "danceability", "energy", "key", "loudness", "mode", "speechiness",
                             "instrumentalness", "liveness", "valence", "tempo", "duration_ms", "time_signature"]
        song_df  = pd.DataFrame(columns = playlist_features_list)
        song = sp.search(q = 'track: {},  artist: {}'.format(song, artist), limit=1)
        try:
            
            for track in song:
                playlist_features = {}
                playlist_features["artist"] = song['tracks']['items'][0]['artists'][0]['name']
                playlist_features['artist_id'] = song['tracks']['items'][0]['artists'][0]['id']
                playlist_features['popularity'] = song['tracks']['items'][0]['popularity']
                playlist_features["album"] = song['tracks']['items'][0]['album']['name']
                playlist_features["track_name"] = song['tracks']['items'][0]['name']
                playlist_features["track_id"] = song['tracks']['items'][0]['id']

                audio_features = sp.audio_features(playlist_features["track_id"])[0]
                for feature in playlist_features_list[6:]:
                    playlist_features[feature] = audio_features[feature]

            track_df = pd.DataFrame(playlist_features, index = [0])
            song_df = pd.concat([song_df, track_df], ignore_index = True)
        except:
            song_df
        return song_df




def predict(song_title, artist, df):
        song = get_song(song_title, artist)
        new_df = pd.concat([song, spot_df], ignore_index=True)
        new_df = new_df.convert_dtypes()
        new_df_feat = new_df.select_dtypes(np.number)
        preds = model.predict(new_df_feat)
        pca_pipeline = Pipeline([('scaler', StandardScaler()), ('PCA', PCA(n_components=2))]) 
        coords = pca_pipeline.fit_transform(new_df_feat)
        new_df['cluster'] = preds
        new_df['x'], new_df['y'] = [x[0] for x in coords], [x[1] for x in coords]
        cluster = new_df.loc[new_df['track_name'] == new_df['track_name'][0], 'cluster'].to_list()[0]
        new_df = new_df.loc[new_df['cluster'] == cluster]
        
        
        return new_df

new_df = (predict(song, artist, spot_df))

with outputs:
    def dist(row):
            x = row['x']
            y = row['y']
            distance=np.sqrt((xt-x)**2 + (yt-y)**2)
            return distance


xt = new_df.loc[new_df['track_name'] == new_df['track_name'][0],'x']
yt = new_df.loc[new_df['track_name'] == new_df['track_name'][0],'y']

new_df['relationship'] = new_df.apply(dist, axis =1)
new_df.drop(["artist_id", "popularity", "track_id", "danceability", "energy", "key", "loudness", "mode", "speechiness", "instrumentalness", "liveness", "valence", "tempo", "duration_ms", "time_signature", "x", "y", "cluster"], axis=1, inplace=True)
new_df.sort_values('relationship')[0:10]




