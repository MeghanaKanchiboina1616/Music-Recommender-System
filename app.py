import streamlit as st
import pickle
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="Music Recommender",
    page_icon="🎵",
    layout="wide"
)

# Load CSS
with open("assets/style.css") as f:
    st.markdown(
        f"<style>{f.read()}</style>",
        unsafe_allow_html=True
    )

# Load files
df = pickle.load(open("songs_df.pkl", "rb"))
tfidf_matrix = pickle.load(open("tfidf_matrix.pkl", "rb"))
indices = pickle.load(open("indices.pkl", "rb"))

def recommend_songs(song_name, top_n=10):

    idx = indices[song_name]

    cosine_sim = cosine_similarity(
        tfidf_matrix[idx],
        tfidf_matrix
    ).flatten()

    sim_scores = list(enumerate(cosine_sim))

    sim_scores = sorted(
        sim_scores,
        key=lambda x: x[1],
        reverse=True
    )

    sim_scores = sim_scores[1:top_n+1]

    recommendations = []

    for i, score in sim_scores:

        recommendations.append({
            "song": df.iloc[i]["music_name"],
            "singer": df.iloc[i]["singer"],
            "year": df.iloc[i]["release"],
            "score": round(score * 100, 2)
        })

    return recommendations

# Header
st.markdown(
"""
<div class='main-title'>
🎵 Bollywood Music Recommendation System
</div>

<div class='subtitle'>
Discover songs similar to your favorite track
</div>
""",
unsafe_allow_html=True
)

# Search
song_name = st.selectbox(
    "Choose a Song",
    sorted(df["music_name"].unique())
)

if st.button("🎧 Get Recommendations"):

    recommendations = recommend_songs(song_name)

    st.success(
        f"Showing recommendations for: {song_name}"
    )

    for song in recommendations:
        st.markdown(
f"""
<div class='song-card'>

<div class='song-title'>
🎵 {song['song']}
</div>

<div class='song-singer'>
🎤 {song['singer']}
</div>

<div class='song-year'>
📅 {song['year']}
</div>

<div class='song-year'>
⭐ Similarity Score: {song['score']}%
</div>

</div>
""",
unsafe_allow_html=True
)