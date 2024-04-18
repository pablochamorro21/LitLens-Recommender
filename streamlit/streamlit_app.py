# LIBRARIES
import pickle
import pandas as pd
import numpy as np
import ast
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import pairwise_distances
import random 
from surprise.prediction_algorithms.algo_base import AlgoBase
from surprise import Reader, Dataset, SVD 
from surprise.model_selection import train_test_split

my_seed = 3325
random.seed(my_seed)
np.random.seed(my_seed)

# LOADING THE MAIN DATASETS
file_path = "/Users/pablochamorro/Desktop/Coding/Recommendation Engines/project/books_df.csv"
books_df = pd.read_csv(file_path)
file_path_1 = "/Users/pablochamorro/Desktop/Coding/Recommendation Engines/project/reviews_df.csv"
row_to_skip = 376979
reviews_df = pd.read_csv(file_path_1, skiprows=[row_to_skip])

# Convert 'read_at' timestamp to datetime
reviews_df['timestamp'] = pd.to_datetime(reviews_df['read_at'], format='%a %b %d %H:%M:%S %z %Y', errors='coerce')
reviews_df['timestamp'] = pd.to_datetime(reviews_df['timestamp'], utc=True, errors='coerce')
reviews_df = reviews_df.dropna(subset=['timestamp'])
reviews_df['timestamp.month'] = reviews_df['timestamp'].dt.month
bins = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
names = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
reviews_df['month'] = pd.cut(reviews_df['timestamp.month'], bins, labels=names, ordered=False)

all_df_filtered = reviews_df.merge(books_df, on='book_id')
all_df_filtered.set_index("book_id", inplace=True)

reader = Reader(rating_scale=(0, 5))
data = Dataset.load_from_df(reviews_df[['user_id', 'book_id', 'rating']], reader)

trainset, testset = train_test_split(data, test_size=.001, random_state=my_seed)

## LOADING THE MODELS

######   COLLABORATIVE + CONTEXT MODEL
n_factors = 1
n_epochs = 20
lr_all = 0.005
reg_all = 0.02

collaborative_f_model = SVD(n_factors=n_factors, n_epochs=n_epochs, lr_all=lr_all, reg_all=reg_all)
collaborative_f_model.fit(trainset)

######   RANDOM MODEL
from random_algorithm import MyRandomAlgorithm  

random_model = MyRandomAlgorithm()
random_model.fit(trainset)


######   POPULAR MODEL

from popular_algorithm import MyPopularAlgorithm  

popular_model = MyPopularAlgorithm()
popular_model.fit(trainset)

# FUNCTION FOR COLLABORATIVE FILTERING
def get_user_recommendations_CF_context(recommender, uid, df, n=10):

    user_books = df[df['user_id'] == uid].index
    user_book_ids = df[df.index.isin(user_books)].index.unique()
    unique_books = set()

    for book_id in user_book_ids:
        book_title = df.loc[int(book_id), 'title'].iloc[0]  
        unique_books.add((book_id, book_title))

    book_list = list(unique_books)
    df_books = pd.DataFrame(book_list, columns=['Book ID', 'Book Title'])

    print("Number of user book IDs:", len(user_book_ids))  # Debugging statement
    print("User book IDs:", user_book_ids)  # Debugging statement

    st.write("Books:", df_books)

    chosen_book_id = st.number_input("Input the Book ID", min_value=int(user_book_ids.min()), max_value=int(user_book_ids.max()), value=int(user_book_ids[0]), step=1, key="Book ID probability")

    if chosen_book_id:
        st.write("Chosen book ID:", chosen_book_id)

    print("Chosen book ID:", chosen_book_id)  # Debugging statement

    button2 = st.button('Get Recommendations', key='recommendations_prob')

    if button2:
        st.session_state['button2_clicked'] = True
    if st.session_state['button2_clicked']:
        month = df.loc[(df['user_id'] == uid) & (df.index == chosen_book_id), "month"].iloc[0]
        user_items = df[(df['user_id'] == uid) & (df['month'] == month)].index.unique()  
        all_items = df[df['month'] == month].index.unique()
        items_to_recommend = np.setdiff1d(all_items, user_items)
        
        predictions = []
        for iid in items_to_recommend:
            pred = recommender.predict(uid, iid, verbose=False)
            predictions.append({'book_id': iid, 'prediction': pred.est})

        predictions.sort(key=lambda x: x['prediction'], reverse=True)
        top_n_recommendations = predictions[:n]
        top_n_titles = [df[(df.index == pred['book_id']) & (df['month'] == month)]['title'].iloc[0] for pred in top_n_recommendations]
        top_n_author = [df[(df.index == pred['book_id']) & (df['month'] == month)]['authors'].iloc[0] for pred in top_n_recommendations]
        author_names = []
        for authors_str in top_n_author:
            authors_list = ast.literal_eval(authors_str)
            names = [author['name'] for author in authors_list]
            author_names.append(names[0])
        top_n_publisher = [df[(df.index == pred['book_id']) & (df['month'] == month)]['publisher'].iloc[0] for pred in top_n_recommendations]

        data = {'title': top_n_titles,
                'author': author_names,
                'publisher': top_n_publisher,
                'prediction': [pred['prediction'] for pred in top_n_recommendations]}
        df_recommendations = pd.DataFrame(data)
        st.write(df_recommendations)


#####   CONTENT-BASED (BERT) MODEL

# TOP N FUNCTION
def top_n(sim_df, df, df_all, item, uid, n):

    user_items = df_all[df_all['user_id'] == uid].index.unique()  
    all_items = df_all.index.unique()
    items_to_recommend = np.setdiff1d(all_items, user_items)
    recommendations_df = df[['description']].copy()
    recommendations_df['distance'] = sim_df.loc[item]
    chosen_book_name = df_all.loc[int(item), 'title']
    recommendations_df = recommendations_df.loc[items_to_recommend]
    recommendations_df = recommendations_df.join(df_all[['title']])
    recommendations_df = recommendations_df.loc[items_to_recommend]
    seen_titles = set()
    unique_recommendations = []

    for index, row in recommendations_df.iterrows():
        title = row['title']
        if title not in seen_titles:
            unique_recommendations.append(row)
            seen_titles.add(title)

    unique_recommendations = pd.DataFrame(unique_recommendations)
    # Create a series with the same length as unique_recommendations['title'] containing chosen_book_name
    chosen_book_series = pd.Series([chosen_book_name] * len(unique_recommendations), index=unique_recommendations.index)
    unique_recommendations = unique_recommendations[~((unique_recommendations['title'].equals(chosen_book_series)) & (unique_recommendations.index != int(item)))]
    unique_recommendations.drop(index=int(item), inplace=True, errors='ignore') # We drop it again, just in case! 
    unique_recommendations = unique_recommendations.sort_values('distance').head(n)

    return unique_recommendations

def get_user_recommendations_bert(uid, df, n=10):

    user_books = df[df['user_id'] == uid].index
    user_book_ids = df[df.index.isin(user_books)].index.unique()
    unique_books = set()

    for book_id in user_book_ids:
        book_title = df.loc[int(book_id), 'title'].iloc[0]  
        unique_books.add((book_id, book_title))

    book_list = list(unique_books)
    df_books = pd.DataFrame(book_list, columns=['Book ID', 'Book Title'])

    print("Number of user book IDs:", len(user_book_ids))  # Debugging statement
    print("User book IDs:", user_book_ids)  # Debugging statement

    st.write("Books:", df_books)

    chosen_book_id = st.number_input("Input the Book ID", min_value=int(user_book_ids.min()), max_value=int(user_book_ids.max()), value=int(user_book_ids[0]), step=1, key="Book ID probability")

    if chosen_book_id:
        st.write("Chosen book ID:", chosen_book_id)

    print("Chosen book ID:", chosen_book_id)  # Debugging statement

    button2 = st.button('Get Recommendations', key='recommendations_prob')

    if button2:
        st.session_state['button2_clicked'] = True
    if st.session_state['button2_clicked']:

        st.write("Please be patient, we are loading your recommendations (this might take around 1 minute)")

        top_books_ids = df.index.unique()
        filtered_books_df = books_df[books_df['book_id'].isin(top_books_ids)]
        filtered_books_df.reset_index(drop=True, inplace=True)
        filtered_books_df.shape[0]

        model = SentenceTransformer('distilbert-base-nli-mean-tokens')
        embeddings = model.encode(filtered_books_df['description'], show_progress_bar=True)

        filtered_books_df['title'] = filtered_books_df['title'].str.strip().str.lower()
        filtered_books_df.set_index('book_id', inplace=True)

        embeddings_df = pd.DataFrame(embeddings)
        embeddings_df.index = filtered_books_df.index

        sim = pairwise_distances(embeddings_df, embeddings_df, metric='cosine')
        bert_sim_df = pd.DataFrame(sim, columns=filtered_books_df.index)
        bert_sim_df.index = filtered_books_df.index


        # Get top n recommendations    
        recommendations = top_n(bert_sim_df, filtered_books_df, df, chosen_book_id, uid, n)
        authors = df.loc[recommendations.index, 'authors']
        publishers = df.loc[recommendations.index, 'publisher']
        authors = authors.drop_duplicates()
        recommendations["authors"] = authors
        publishers = publishers.drop_duplicates()
        recommendations["publishers"] = publishers
        recommendations['authors'].fillna('[]', inplace=True)
        recommendations['authors'] = recommendations['authors'].apply(ast.literal_eval)
        recommendations['author_name'] = recommendations['authors'].apply(lambda x: x[0]['name'] if x else None)
        recommendations = recommendations[['title', 'author_name', 'publishers', 'description', 'distance']]
        
        st.write(recommendations)



#####   FUNCTIONS FOR RANDOM + POPULAR MODEL (if users = 0)
def get_user_recommendations_Random(recommender, uid, df, n=10):

    user_books = df[df['user_id'] == uid].index
    user_book_ids = df[df.index.isin(user_books)].index.unique()
    unique_books = set()

    for book_id in user_book_ids:
        book_title = df.loc[int(book_id), 'title'].iloc[0]  
        unique_books.add((book_id, book_title))

    print(f"Books for user: {uid} with user_id: {uid}")
    for book_id, book_title in unique_books:
        print(f"Book ID: {book_id}. Book title: {book_title}")

    user_items = df[(df['user_id'] == uid)].index.unique()  
    all_items = df.index.unique()
    items_to_recommend = np.setdiff1d(all_items, user_items)
    
    predictions = []
    for iid in items_to_recommend:
        pred = recommender.estimate(uid, iid)
        predictions.append({'book_id': iid, 'prediction': pred})

    predictions.sort(key=lambda x: x['prediction'], reverse=True)
    top_n_recommendations = predictions[:n]
    top_n_titles = [df[(df.index == pred['book_id'])]['title'].iloc[0] for pred in top_n_recommendations]
    top_n_author = [df[(df.index == pred['book_id'])]['authors'].iloc[0] for pred in top_n_recommendations]
    author_names = []
    for authors_str in top_n_author:
        authors_list = ast.literal_eval(authors_str)
        names = [author['name'] for author in authors_list]
        author_names.append(names[0])
    top_n_publisher = [df[(df.index == pred['book_id'])]['publisher'].iloc[0] for pred in top_n_recommendations]

    data = {'title': top_n_titles,
            'author': author_names,
            'publisher': top_n_publisher} # No prediction, as they are random, and the values are above 5, which shouldn't be
    df_recommendations = pd.DataFrame(data)

    return df_recommendations


def get_user_recommendations_Popular(recommender, uid, df, n=50):

    user_books = df[df['user_id'] == uid].index
    user_book_ids = df[df.index.isin(user_books)].index.unique()
    unique_books = set()

    for book_id in user_book_ids:
        book_title = df.loc[int(book_id), 'title'].iloc[0]  
        unique_books.add((book_id, book_title))

    print(f"Books for user: {uid} with user_id: {uid}")
    for book_id, book_title in unique_books:
        print(f"Book ID: {book_id}. Book title: {book_title}")

    user_items = df[(df['user_id'] == uid)].index.unique()  
    all_items = df.index.unique()
    items_to_recommend = np.setdiff1d(all_items, user_items)
    
    predictions = []
    for iid in items_to_recommend:
        pred = recommender.estimate(uid, iid)
        predictions.append({'book_id': iid, 'prediction': pred})

    predictions.sort(key=lambda x: x['prediction'], reverse=True)
    top_n_recommendations = predictions[:n]
    top_n_titles = [df[(df.index == pred['book_id'])]['title'].iloc[0] for pred in top_n_recommendations]
    top_n_author = [df[(df.index == pred['book_id'])]['authors'].iloc[0] for pred in top_n_recommendations]
    author_names = []
    for authors_str in top_n_author:
        authors_list = ast.literal_eval(authors_str)
        names = [author['name'] for author in authors_list]
        author_names.append(names[0])
    top_n_publisher = [df[(df.index == pred['book_id'])]['publisher'].iloc[0] for pred in top_n_recommendations]

    data = {'title': top_n_titles,
            'author': author_names,
            'publisher': top_n_publisher} 
    df_recommendations = pd.DataFrame(data)

    return df_recommendations

def random_popular_recommender(popular_rs, random_rs, uid, df):

    df_popular_recommendations = get_user_recommendations_Popular(popular_rs, uid, df)
    df_random_recommendations = get_user_recommendations_Random(random_rs, uid, df)

    sample_popular_rows = df_popular_recommendations.sample(n=6)
    sample_random_rows = df_random_recommendations.sample(n=4)

    final_df = pd.concat([sample_popular_rows, sample_random_rows])

    # Shuffle recommendations
    shuffled_df = final_df.sample(frac=1).reset_index(drop=True)

    st.write(shuffled_df)


#####   FUNCTION FOR RANDOM MODEL (if probability < 1%)
def get_user_recommendations_Random_probability(recommender, uid, df, n=10):
    user_books = df[df['user_id'] == uid].index
    user_book_ids = df[df.index.isin(user_books)].index.unique()
    unique_books = set()

    for book_id in user_book_ids:
        book_title = df.loc[int(book_id), 'title'].iloc[0]  
        unique_books.add((book_id, book_title))

    book_list = list(unique_books)
    df_books = pd.DataFrame(book_list, columns=['Book ID', 'Book Title'])

    print("Number of user book IDs:", len(user_book_ids))  # Debugging statement
    print("User book IDs:", user_book_ids)  # Debugging statement

    st.write("Books:", df_books)

    chosen_book_id = st.number_input("Input the Book ID", min_value=int(user_book_ids.min()), max_value=int(user_book_ids.max()), value=int(user_book_ids[0]), step=1, key="Book ID probability")

    if chosen_book_id:
        st.write("Chosen book ID:", chosen_book_id)

    print("Chosen book ID:", chosen_book_id)  # Debugging statement

    button2 = st.button('Get Recommendations', key='recommendations_prob')

    if button2:
        st.session_state['button2_clicked'] = True
    if st.session_state['button2_clicked']:
        user_items = df[(df['user_id'] == uid)].index.unique()  
        all_items = df.index.unique()
        items_to_recommend = np.setdiff1d(all_items, user_items)
        
        predictions = []
        for iid in items_to_recommend:
            pred = recommender.estimate(uid, iid)
            predictions.append({'book_id': iid, 'prediction': pred})

        predictions.sort(key=lambda x: x['prediction'], reverse=True)
        top_n_recommendations = predictions[:n]
        top_n_titles = [df[(df.index == pred['book_id'])]['title'].iloc[0] for pred in top_n_recommendations]
        top_n_author = [df[(df.index == pred['book_id'])]['authors'].iloc[0] for pred in top_n_recommendations]
        author_names = []
        for authors_str in top_n_author:
            authors_list = ast.literal_eval(authors_str)
            names = [author['name'] for author in authors_list]
            author_names.append(names[0])
        top_n_publisher = [df[(df.index == pred['book_id'])]['publisher'].iloc[0] for pred in top_n_recommendations]

        data = {'title': top_n_titles,
                'author': author_names,
                'publisher': top_n_publisher} # No prediction, as they are random, and the values are above 5, which shouldn't be
        df_recommendations = pd.DataFrame(data)

        st.write(df_recommendations)


#   user_id above 20 = 91efd74c298b00a55ef9bcf345ec9cc3
#   user_id below 20 = ca803a8c561eb0a4304e7a91a3841e50
#   user_id = NaN for Random Recommender

##### STREAMLIT

import streamlit as st

def set_theme():
    st.set_page_config(
        page_title="LitLens",  
        page_icon="📚",  
        layout="wide"
    )

    st.markdown("""
    <style>
    /* General settings for background and text */
    html, body, .stApp {
        margin: 0;
        padding: 0;
        height: 100%;
        background-color: #1B1B1B !important;  /* Black background */
        color: #FFFFFF !important;  /* White text */
        font-family: 'Helvetica', sans-serif;
    }

    /* Button styling with more specificity */
    button {
        color: #FFFFFF !important;  /* White text */
        border: 1px solid #FFFFFF !important;  /* White border */
        background-color: #004d00 !important;  /* Dark Green background */
    }

    /* Input fields styling */
    .stTextInput input, .stTextInput label {
        color: #FFFFFF !important;  /* White text */
        background-color: #333333 !important;  /* Dark input fields */
        border-color: #FFFFFF !important;  /* White border */
    }

    /* Titles and headers styling and centering */
    h1 {
        color: #66cc66 !important;  /* Lighter Green for title */
        text-align: center !important;
    }
    /* More precise targeting for dynamically generated classes */
    [data-testid="stMarkdownContainer"] h2,
    [data-testid="stMarkdownContainer"] span {
        color: #FFFFFF !important;  /* White text for subheader */
        text-align: center !important;
    }

    /* Centering the main container */
    .main .block-container {
        max-width: 800px;
        margin: auto;
        padding-top: 5rem;
        padding-right: 1rem;
        padding-left: 1rem;
        padding-bottom: 5rem;
    }

    /* Centering the image */
    .stImage {
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

def main():
    set_theme()
    
    st.title("LitLens")
    
    st.image("/Users/pablochamorro/Desktop/Coding/Recommendation Engines/project/logo.png", width=200, use_column_width=True, output_format='auto')  # Ensure path is correct

    text = st.empty()

    st.markdown("""
    - **Example user_id above 20**:  `91efd74c298b00a55ef9bcf345ec9cc3`
    - **Example user_id below 20**:  `ca803a8c561eb0a4304e7a91a3841e50`
    - **user_id = NaN for Random Recommender**
    """, unsafe_allow_html=True)

    user_id = text.text_input("Input your USER ID", value="", key="1", placeholder="Enter USER ID here")

    num_ratings = int(all_df_filtered[all_df_filtered['user_id'] == user_id].shape[0])
    n = 10

    def initialize_state():
        st.session_state['button1_clicked'] = st.session_state.get('button1_clicked', False)
        st.session_state['button2_clicked'] = st.session_state.get('button2_clicked', False)
    initialize_state()

    button1 = st.button('Submit', key='Initial Submit')

    if button1:
        st.session_state['button1_clicked'] = True
    
    if random.random() >= 0.99:
        get_user_recommendations_Random_probability(random_model, user_id, all_df_filtered, n)
    else:
        if num_ratings > 20: 
            st.write("Using the COLLABORATIVE FILTERING MODEL")
            get_user_recommendations_CF_context(collaborative_f_model, user_id, all_df_filtered, n)
        elif num_ratings > 0:
            st.write("Using the CONTENT MODEL")
            get_user_recommendations_bert(user_id, all_df_filtered, n)
        else:
            st.write("Using the RANDOM + POPULAR MODEL")
            random_popular_recommender(popular_model, random_model, user_id, all_df_filtered)

if __name__ == "__main__":
    main()
