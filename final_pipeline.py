# LIBRARIES
import pickle
import pandas as pd
import numpy as np
import ast
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import pairwise_distances
import random 
from surprise.prediction_algorithms.algo_base import AlgoBase


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

## LOADING THE MODELS

######   COLLABORATIVE + CONTEXT MODEL
file_path = "/Users/pablochamorro/Desktop/Coding/Recommendation Engines/project/collaborative_filtering_model.pkl"
with open(file_path, 'rb') as file:
    collaborative_f_model = pickle.load(file)

######   RANDOM MODEL
import numpy as np
from surprise.prediction_algorithms.algo_base import AlgoBase

class MyRandomAlgorithm(AlgoBase):

    def __init__(self):
        AlgoBase.__init__(self)

    def estimate(self, u, i):
        # Note u and i are not used, this RS does not personalize based on neither the user nor the item
        return np.random.normal(loc=self.train_mean, scale=self.train_std)

    def fit(self, trainset):
        AlgoBase.fit(self, trainset)

        ratings = [r for (_, _, r) in self.trainset.all_ratings()]
        self.train_mean = np.mean(ratings)
        self.train_std = np.std(ratings)

        return self

file_path = "/Users/pablochamorro/Desktop/Coding/Recommendation Engines/project/random_model.pkl"
with open(file_path, 'rb') as file:
    random_model = pickle.load(file)

# FUNCTION FOR COLLABORATIVE FILTERING
def get_user_recommendations_CF_context(recommender, uid, df, n=10):

    user_books = df[df['user_id'] == uid].index
    user_book_ids = df[df.index.isin(user_books)].index.unique()
    unique_books = set()

    for book_id in user_book_ids:
        book_title = df.loc[int(book_id), 'title'].iloc[0]  # Access the first element of the Series
        unique_books.add((book_id, book_title))

    print(f"Books for user: {uid} with user_id: {uid}")
    for book_id, book_title in unique_books:
        print(f"Book ID: {book_id}. Book title: {book_title}")

    chosen_book_id = int(input("Enter the book_id of the book you want recommendations for: ").strip())
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

    return df_recommendations


#####   CONTENT-BASED (BERT) MODEL

# Dataframes loading and preparation!

top_books_ids = all_df_filtered.index.unique()
filtered_books_df = books_df[books_df['book_id'].isin(top_books_ids)]
filtered_books_df.reset_index(drop=True, inplace=True)
filtered_books_df.shape[0]

model = SentenceTransformer('distilbert-base-nli-mean-tokens')
embeddings = model.encode(filtered_books_df['description'], show_progress_bar=True)

filtered_books_df['title'] = filtered_books_df['title'].str.strip().str.lower()
filtered_books_df.set_index('book_id', inplace=True)

embeddings_df = pd.DataFrame(embeddings)
embeddings_df.index = filtered_books_df.index

def compute_similarities(vectorization_df):
  sim = pairwise_distances(vectorization_df, vectorization_df, metric='cosine')
  sim_df = pd.DataFrame(sim, columns=filtered_books_df.index)
  sim_df.index = filtered_books_df.index
  return sim_df

bert_sim_df = compute_similarities(embeddings_df)

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

def get_user_recommendations_bert(uid, df, filtered_books_df, bert_sim_df, n=10):
    user_books = df[df['user_id'] == uid].index
    user_book_ids = df[df.index.isin(user_books)].index.unique()
    unique_books = set()

    for book_id in user_book_ids:
        book_title = df.loc[int(book_id), 'title'].iloc[0]  # Access the first element of the Series
        unique_books.add((book_id, book_title))

    print(f"Books for user: {uid} with user_id: {uid}")
    for book_id, book_title in unique_books:
        print(f"Book ID: {book_id}. Book title: {book_title}")

    chosen_book_id = int(input("Enter the book_id of the book you want recommendations for: ").strip())

    # Get top n recommendations    
    chosen_book_name = df.loc[int(chosen_book_id), 'title']
    recommendations = top_n(bert_sim_df, filtered_books_df, df, chosen_book_id, uid, n+1)
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
    
    return chosen_book_name, recommendations



#####   FUNCTION FOR RANDOM MODEL (if users = 0)
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

#####   FUNCTION FOR RANDOM MODEL (if probability < 1%)
def get_user_recommendations_Random_probability(recommender, uid, df, n=10):

    user_books = df[df['user_id'] == uid].index
    user_book_ids = df[df.index.isin(user_books)].index.unique()
    unique_books = set()

    for book_id in user_book_ids:
        book_title = df.loc[int(book_id), 'title'].iloc[0]  
        unique_books.add((book_id, book_title))

    print(f"Books for user: {uid} with user_id: {uid}")
    for book_id, book_title in unique_books:
        print(f"Book ID: {book_id}. Book title: {book_title}")

    chosen_book_id = int(input("Enter the book_id of the book you want recommendations for: ").strip())
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


#####   FINAL FUNCTION
def finalfunc():
    user_id = input("Please enter your user id: ")
    num_ratings = int(all_df_filtered[all_df_filtered['user_id'] == user_id].shape[0])
    n = 10

    if random.random() >= 0.99:
        df_recommendations = get_user_recommendations_Random_probability(random_model, user_id, all_df_filtered, n)
        return df_recommendations
    
    else:
        if num_ratings > 20 : 
            # Collaborative Filterning Model
            print("Using the COLLABORATIVE FILTERNING MODEL")
            df_recommendations = get_user_recommendations_CF_context(collaborative_f_model, user_id, all_df_filtered, n)
            return df_recommendations

        elif num_ratings > 0:
            # Content-model 
            print("Using the CONTENT MODEL")
            book_name, df_recommendations = get_user_recommendations_bert(user_id, all_df_filtered, filtered_books_df, bert_sim_df, n)
            return df_recommendations

        else:
            print("Using the RANDOM MODEL")
            df_recommendations = get_user_recommendations_Random(random_model, user_id, all_df_filtered, n)
            return df_recommendations
        

#   user_id above 20 = 91efd74c298b00a55ef9bcf345ec9cc3
#   user_id below 20 = ca803a8c561eb0a4304e7a91a3841e50
#   user_id = NaN for Random Recommender

df_recommendations = finalfunc()
df_recommendations.head(20) # 20 just in case!