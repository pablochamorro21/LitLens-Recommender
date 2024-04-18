# Libraries
import pickle
import pandas as pd
import numpy as np
import ast
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import pairwise_distances
import random 
from surprise.prediction_algorithms.algo_base import AlgoBase

my_seed = 3325
random.seed(my_seed)
np.random.seed(my_seed)


path = " "
# /Users/pablochamorro/Desktop/Coding/Recommendation Engines/project/
# /Users/alexanderbenady/venv/recommendations/

# Loadging the item metadata + user-item interaction datasets
file_path = path + "books_df.csv"
books_df = pd.read_csv(file_path)
file_path_1 = path + "reviews_df.csv"
row_to_skip = 376979
reviews_df = pd.read_csv(file_path_1, skiprows=[row_to_skip])

# Convert 'read_at' timestamp to datetime,and classifying with month names for contextual pre-filtering
reviews_df['timestamp'] = pd.to_datetime(reviews_df['read_at'], format='%a %b %d %H:%M:%S %z %Y', errors='coerce')
reviews_df['timestamp'] = pd.to_datetime(reviews_df['timestamp'], utc=True, errors='coerce')
reviews_df = reviews_df.dropna(subset=['timestamp'])
reviews_df['timestamp.month'] = reviews_df['timestamp'].dt.month
bins = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
names = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
reviews_df['month'] = pd.cut(reviews_df['timestamp.month'], bins, labels=names, ordered=False)

all_df_filtered = reviews_df.merge(books_df, on='book_id')
all_df_filtered.set_index("book_id", inplace=True)

# Loading the models

## Collaborative + Context model

file_path = path + "collaborative_filtering_model.pkl"
with open(file_path, 'rb') as file:
    collaborative_f_model = pickle.load(file)

## Random model

class MyRandomAlgorithm(AlgoBase):
    """
    A recommendation algorithm that predicts ratings based on a normal distribution
    centered around the mean of all known ratings. 
    Attributes:
        train_mean (float): The mean of all ratings in the training dataset.
        train_std (float): The standard deviation of all ratings in the training dataset.

    Methods:
        fit(trainset): Learns the mean and standard deviation of the ratings from the training set.
        estimate(u, i): Predicts a random rating for the given user and item,
    """

    def __init__(self):
        """
        Initializes the MyRandomAlgorithm instance, inheriting from AlgoBase.
        """
        AlgoBase.__init__(self)

    def estimate(self, u, i):
        """
        Predict a rating for a given user and item using a random normal distribution.

        Parameters:
            u (int): User ID (not used in the prediction).
            i (int): Item ID (not used in the prediction).

        Returns:
            float: A randomly generated rating based on the normal distribution parameters.
        """
        # Generate a random rating based on the trained mean and standard deviation.
        # User and item IDs are not utilized in generating this rating, reflecting
        # the non-personalized nature of this algorithm.
        return np.random.normal(loc=self.train_mean, scale=self.train_std)

    def fit(self, trainset):
        """
        Fit the algorithm to the given training set to learn the distribution of the ratings.

        Parameters:
            trainset (Trainset): A trainset object as defined in the Surprise library.

        Returns:
            MyRandomAlgorithm: The instance of MyRandomAlgorithm with trained mean and std.
        """
        # Call base class fit method (necessary for setup within the Surprise library framework)
        AlgoBase.fit(self, trainset)

        # Extract all ratings from the trainset to calculate the mean and standard deviation.
        ratings = [r for (_, _, r) in self.trainset.all_ratings()]
        self.train_mean = np.mean(ratings)
        self.train_std = np.std(ratings)

        # Return self to allow for chaining or method calls after fitting.
        return self

file_path = path + "random_model.pkl"
with open(file_path, 'rb') as file:
    random_model = pickle.load(file)


## Popular Model

class MyPopularAlgorithm(AlgoBase):
    """
    A recommendation algorithm that predicts ratings based on the average ratings
    of items from the training dataset. 

    Attributes:
        mean_rating_per_item_df (DataFrame): A pandas DataFrame containing the mean
        ratings for each item in the training set.

    Methods:
        fit(trainset): Calculates the mean rating for each item in the training dataset.
        estimate(u, i): Predicts the rating for a specified item based on its average rating.
    """

    def __init__(self):
        """
        Initializes the MyPopularAlgorithm instance, inheriting from AlgoBase.
        """
        AlgoBase.__init__(self)

    def estimate(self, u, i):
        """
        Predict a rating for a given user and item based on the item's average rating.

        Parameters:
            u (int): User ID (not used in the prediction).
            i (int): Item ID used to lookup the item's average rating.

        Returns:
            float: The predicted rating for the item. If the item has no previous ratings,
            returns the global mean rating from the training set.
        """
        # Check if the item exists in the mean rating DataFrame and return its rating.
        # If not found, return the global mean rating from the training set.
        return self.mean_rating_per_item_df.loc[i]['rating'] if i in self.mean_rating_per_item_df.index else self.trainset.global_mean

    def fit(self, trainset):
        """
        Fit the algorithm to the given training set to compute the average rating for each item.

        Parameters:
            trainset (Trainset): A trainset object as defined in the Surprise library.

        Returns:
            MyPopularAlgorithm: The instance of MyPopularAlgorithm with populated mean ratings.
        """
        # Call base class fit method to perform initial setup.
        AlgoBase.fit(self, trainset)

        # Create a DataFrame from all ratings in the trainset, grouped by item.
        ratings_df = pd.DataFrame([(i, r) for (_, i, r) in self.trainset.all_ratings()], columns=['item', 'rating'])
        
        # Compute the mean rating for each item and store it in a DataFrame.
        self.mean_rating_per_item_df = ratings_df.groupby('item').agg({'rating': 'mean'})

        # Return self to allow method chaining or calls after fitting.
        return self

file_path = path + "popular_model.pkl"
with open(file_path, 'rb') as file:
    popular_model = pickle.load(file)


# Function for collaborative filtering

def get_user_recommendations_CF_context(recommender, uid, df, n=10):
    """
    Generate top-n book recommendations for a given user based on collaborative filtering,
    taking into account the context of the month of previous interactions.

    Parameters:
        recommender (AlgoBase): The collaborative filtering model to use for predictions.
        uid (int): The user ID for whom recommendations are to be generated.
        df (DataFrame): The dataframe containing user interactions with books.
        n (int): The number of recommendations to generate.

    Returns:
        DataFrame: A dataframe containing the top-n recommended books with their titles,
                   authors, publishers, and predicted ratings.
    """
    
    # Retrieve all books associated with the user and store unique identifiers.
    user_books = df[df['user_id'] == uid].index
    user_book_ids = df[df.index.isin(user_books)].index.unique()
    unique_books = set()

    # Collect titles of books for display and further selection.
    for book_id in user_book_ids:
        book_title = df.loc[int(book_id), 'title'].iloc[0]
        unique_books.add((book_id, book_title))

    print(f"Books for user: {uid} with user_id: {uid}")
    for book_id, book_title in unique_books:
        print(f"Book ID: {book_id}. Book title: {book_title}")

    # Allow user to select a book to base recommendations on, typically the most recently read.
    chosen_book_id = int(input("Enter the book_id of the book you want recommendations for: ").strip())
    month = df.loc[(df['user_id'] == uid) & (df.index == chosen_book_id), "month"].iloc[0]

    # Filter items based on the same month of the chosen book.
    user_items = df[(df['user_id'] == uid) & (df['month'] == month)].index.unique()
    all_items = df[df['month'] == month].index.unique()
    items_to_recommend = np.setdiff1d(all_items, user_items)

    # Generate predictions for each item not yet interacted with by the user during the same month.
    predictions = []
    for iid in items_to_recommend:
        pred = recommender.predict(uid, iid, verbose=False)
        predictions.append({'book_id': iid, 'prediction': pred.est})

    # Sort predictions to get the highest rated items first.
    predictions.sort(key=lambda x: x['prediction'], reverse=True)
    top_n_recommendations = predictions[:n]

    # Collect details of the recommended books for display.
    top_n_titles = [df[(df.index == pred['book_id']) & (df['month'] == month)]['title'].iloc[0] for pred in top_n_recommendations]
    top_n_author = [df[(df.index == pred['book_id']) & (df['month'] == month)]['authors'].iloc[0] for pred in top_n_recommendations]
    author_names = []
    for authors_str in top_n_author:
        authors_list = ast.literal_eval(authors_str)
        names = [author['name'] for author in authors_list]
        author_names.append(names[0])
    top_n_publisher = [df[(df.index == pred['book_id']) & (df['month'] == month)]['publisher'].iloc[0] for pred in top_n_recommendations]

    # Compile recommendation data into a DataFrame for easy display or export.
    data = {
        'title': top_n_titles,
        'author': author_names,
        'publisher': top_n_publisher,
        'prediction': [pred['prediction'] for pred in top_n_recommendations]
    }
    df_recommendations = pd.DataFrame(data)

    return df_recommendations



# Content-based (BERT) model

## Dataframes loading and preparation

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

## Functions

def compute_similarities(vectorization_df):
    """
    Calculate the pairwise cosine similarities between all vectors in a given DataFrame.
    
    Args:
        vectorization_df (DataFrame): A DataFrame where each row represents an embedding vector
        for an item (book).

    Returns:
        DataFrame: A DataFrame containing cosine similarity scores between items, where each
        row and column index corresponds to an item's index in the original DataFrame.
    """
    # Compute cosine similarity matrix from the vectorized DataFrame.
    sim = pairwise_distances(vectorization_df, vectorization_df, metric='cosine')

    # Convert the numpy array returned by pairwise_distances into a DataFrame.
    sim_df = pd.DataFrame(sim, columns=vectorization_df.index)
    sim_df.index = vectorization_df.index

    return sim_df

bert_sim_df = compute_similarities(embeddings_df)

def top_n(sim_df, df, df_all, item, uid, n):
    """
    Generate top-n similar items based on cosine similarity scores for a given item.

    Args:
        sim_df (DataFrame): DataFrame containing cosine similarity scores.
        df (DataFrame): DataFrame containing specific columns like 'description'.
        df_all (DataFrame): Comprehensive DataFrame containing all relevant book data.
        item (int): Index of the item used as the basis for recommendations.
        uid (int): User ID for whom recommendations are being generated.
        n (int): Number of recommendations to generate.

    Returns:
        DataFrame: A DataFrame of the top-n recommended items sorted by similarity, excluding
        the input item itself.
    """
    # Determine items previously interacted with by the user to exclude them.
    user_items = df_all[df_all['user_id'] == uid].index.unique()  
    all_items = df_all.index.unique()
    items_to_recommend = np.setdiff1d(all_items, user_items)

    # Prepare a DataFrame for recommendation.
    recommendations_df = df[['description']].copy()
    recommendations_df['distance'] = sim_df.loc[item]
    chosen_book_name = df_all.loc[int(item), 'title']
    recommendations_df = recommendations_df.loc[items_to_recommend]
    recommendations_df = recommendations_df.join(df_all[['title']])
    
    # Filter and collect unique recommendations.
    seen_titles = set()
    unique_recommendations = []
    for index, row in recommendations_df.iterrows():
        title = row['title']
        if title not in seen_titles:
            unique_recommendations.append(row)
            seen_titles.add(title)

    # Convert list to DataFrame and remove any duplicate recommendations.
    unique_recommendations = pd.DataFrame(unique_recommendations)
    chosen_book_series = pd.Series([chosen_book_name] * len(unique_recommendations), index=unique_recommendations.index)
    unique_recommendations = unique_recommendations[~((unique_recommendations['title'] == chosen_book_series) & (unique_recommendations.index != int(item)))]
    unique_recommendations.drop(index=int(item), inplace=True, errors='ignore') 
    unique_recommendations = unique_recommendations.sort_values('distance').head(n)

    return unique_recommendations

import ast

def get_user_recommendations_bert(uid, df, filtered_books_df, bert_sim_df, n=10):
    """
    Generates personalized book recommendations for a specific user based on a selected book
    using a BERT-based similarity matrix.

    Parameters:
        uid (int): The user ID for whom recommendations are to be generated.
        df (DataFrame): The main DataFrame containing user-book interactions.
        filtered_books_df (DataFrame): DataFrame containing books eligible for recommendations.
        bert_sim_df (DataFrame): A DataFrame representing the pairwise cosine similarity scores between book descriptions.
        n (int): Number of recommendations to return.

    Returns:
        tuple: The name of the chosen book and a DataFrame containing the top-n recommended books,
               including titles, authors, publishers, descriptions, and their similarity scores.
    """
    # Retrieve all books associated with the user and collect their titles.
    user_books = df[df['user_id'] == uid].index
    user_book_ids = df[df.index.isin(user_books)].index.unique()
    unique_books = set()

    for book_id in user_book_ids:
        book_title = df.loc[int(book_id), 'title'].iloc[0]  # Retrieve the title from the DataFrame
        unique_books.add((book_id, book_title))

    # Display books associated with the user for selection.
    print(f"Books for user: {uid} with user_id: {uid}")
    for book_id, book_title in unique_books:
        print(f"Book ID: {book_id}. Book title: {book_title}")

    # Prompt the user to select a book ID to base recommendations on.
    chosen_book_id = int(input("Enter the book_id of the book you want recommendations for: ").strip())
    chosen_book_name = df.loc[int(chosen_book_id), 'title']

    # Generate top n recommendations using the BERT-based similarity matrix.
    recommendations = top_n(bert_sim_df, filtered_books_df, df, chosen_book_id, uid, n)

    # Retrieve authors and publishers from the recommendations.
    authors = df.loc[recommendations.index, 'authors']
    publishers = df.loc[recommendations.index, 'publisher']
    authors = authors.drop_duplicates()
    recommendations["authors"] = authors
    publishers = publishers.drop_duplicates()
    recommendations["publishers"] = publishers

    # Process the authors' information to extract the primary author name.
    recommendations['authors'].fillna('[]', inplace=True)
    recommendations['authors'] = recommendations['authors'].apply(ast.literal_eval)
    recommendations['author_name'] = recommendations['authors'].apply(lambda x: x[0]['name'] if x else None)

    # Prepare the final DataFrame to return.
    recommendations = recommendations[['title', 'author_name', 'publishers', 'description', 'distance']]

    return chosen_book_name, recommendations


# Functions for Random + Popular (if users = 0)

def get_user_recommendations_Random(recommender, uid, df, n=10):
    """
    Generates random book recommendations for a user based on the random recommender's
    predictions, ensuring that recommended books have not been interacted with by the user.

    Parameters:
        recommender (AlgoBase): The random recommendation algorithm.
        uid (int): The user ID for whom recommendations are to be generated.
        df (DataFrame): The dataframe containing user-book interactions.
        n (int): The number of recommendations to generate.

    Returns:
        DataFrame: A dataframe containing the top-n random recommended books, including
                   titles, authors, and publishers.
    """
    # Fetch all books associated with the user and collect unique identifiers.
    user_books = df[df['user_id'] == uid].index
    user_book_ids = df[df.index.isin(user_books)].index.unique()
    unique_books = set()

    # Collect the titles of books associated with this user for reference.
    for book_id in user_book_ids:
        book_title = df.loc[int(book_id), 'title'].iloc[0]  # Access the title directly from the DataFrame
        unique_books.add((book_id, book_title))

    # Display books for user debug or selection.
    print(f"Books for user: {uid} with user_id: {uid}")
    for book_id, book_title in unique_books:
        print(f"Book ID: {book_id}. Book title: {book_title}")

    # Identify books that the user has not interacted with to make recommendations.
    user_items = df[df['user_id'] == uid].index.unique()  
    all_items = df.index.unique()
    items_to_recommend = np.setdiff1d(all_items, user_items)

    # Generate predictions for each non-interacted item using the random recommender.
    predictions = []
    for iid in items_to_recommend:
        pred = recommender.estimate(uid, iid)
        predictions.append({'book_id': iid, 'prediction': pred})

    # Sort the predictions to select the top n.
    predictions.sort(key=lambda x: x['prediction'], reverse=True)
    top_n_recommendations = predictions[:n]

    # Collect details of the recommended books to display or further analysis.
    top_n_titles = [df.loc[pred['book_id'], 'title'] for pred in top_n_recommendations]
    top_n_author = [df.loc[pred['book_id'], 'authors'] for pred in top_n_recommendations]
    author_names = []
    for authors_str in top_n_author:
        authors_list = ast.literal_eval(authors_str)
        names = [author['name'] for author in authors_list]
        author_names.append(names[0])
    top_n_publisher = [df.loc[pred['book_id'], 'publisher'] for pred in top_n_recommendations]

    # Assemble the recommendation data into a DataFrame for easy output.
    data = {
        'title': top_n_titles,
        'author': author_names,
        'publisher': top_n_publisher
    }
    df_recommendations = pd.DataFrame(data)

    return df_recommendations


def get_user_recommendations_Popular(recommender, uid, df, n=50):
    """
    Generates book recommendations for a user based on the popularity of the books,
    determined by a recommender system that prioritizes items with higher ratings.
    
    Parameters:
        recommender (AlgoBase): An instance of a recommender algorithm that supports the
                                'estimate' method for predicting item scores.
        uid (int): The user ID for whom recommendations are to be generated.
        df (DataFrame): The dataframe containing user-book interactions and other metadata.
        n (int): The number of recommendations to return.

    Returns:
        DataFrame: A dataframe containing the top-n popular recommended books for the user,
                   including titles, authors, and publishers.
    """
    # Retrieve all books associated with the user and store their titles.
    user_books = df[df['user_id'] == uid].index
    user_book_ids = df[df.index.isin(user_books)].index.unique()
    unique_books = set()

    for book_id in user_book_ids:
        book_title = df.loc[int(book_id), 'title'].iloc[0]  
        unique_books.add((book_id, book_title))

    # Display books for user interaction.
    print(f"Books for user: {uid} with user_id: {uid}")
    for book_id, book_title in unique_books:
        print(f"Book ID: {book_id}. Book title: {book_title}")

    # Determine books that have not been interacted with by the user.
    user_items = df[df['user_id'] == uid].index.unique()  
    all_items = df.index.unique()
    items_to_recommend = np.setdiff1d(all_items, user_items)

    # Estimate scores for these items using the recommender and gather predictions.
    predictions = []
    for iid in items_to_recommend:
        pred = recommender.estimate(uid, iid)
        predictions.append({'book_id': iid, 'prediction': pred})

    # Sort predictions by estimated scores to select the top n.
    predictions.sort(key=lambda x: x['prediction'], reverse=True)
    top_n_recommendations = predictions[:n]

    # Collect details of recommended books.
    top_n_titles = [df.loc[pred['book_id'], 'title'] for pred in top_n_recommendations]
    top_n_author = [df.loc[pred['book_id'], 'authors'] for pred in top_n_recommendations]
    author_names = []
    for authors_str in top_n_author:
        authors_list = ast.literal_eval(authors_str)
        names = [author['name'] for author in authors_list]
        author_names.append(names[0])
    top_n_publisher = [df.loc[pred['book_id'], 'publisher'] for pred in top_n_recommendations]

    # Prepare the final recommendations DataFrame.
    data = {
        'title': top_n_titles,
        'author': author_names,
        'publisher': top_n_publisher
    }
    df_recommendations = pd.DataFrame(data)

    return df_recommendations


def random_popular_recommender(popular_rs, random_rs, uid, df):
    """
    Combines recommendations from a popularity-based recommender and a random recommender
    to provide a diverse set of book recommendations for a specific user.

    Parameters:
        popular_rs (AlgoBase): An instance of a recommender system that suggests items based on popularity.
        random_rs (AlgoBase): An instance of a recommender system that suggests items randomly.
        uid (int): The user ID for whom recommendations are to be generated.
        df (DataFrame): The dataframe containing user-book interactions.

    Returns:
        DataFrame: A shuffled dataframe containing a combined list of recommendations from both
                   the popularity-based and random recommenders.
    """
    # Retrieve recommendations from the popularity-based recommender.
    df_popular_recommendations = get_user_recommendations_Popular(popular_rs, uid, df)
    # Retrieve recommendations from the random recommender.
    df_random_recommendations = get_user_recommendations_Random(random_rs, uid, df)

    # Sample a specified number of recommendations from both sets.
    sample_popular_rows = df_popular_recommendations.sample(n=6)
    sample_random_rows = df_random_recommendations.sample(n=4)

    # Combine both sets of recommendations into a single DataFrame.
    final_df = pd.concat([sample_popular_rows, sample_random_rows])

    # Shuffle the combined DataFrame to mix the recommendations from both strategies.
    shuffled_df = final_df.sample(frac=1).reset_index(drop=True)

    return shuffled_df

## Function for probabilistic random model (if probability < 1%)

def get_user_recommendations_Random_probability(recommender, uid, df, n=10):
    """
    Generates random book recommendations for a user with a specified probability 
    of the recommendations being truly random, intended for situations where a random 
    outcome is less frequent but needed.

    Parameters:
        recommender (AlgoBase): An instance of a recommender system capable of random predictions.
        uid (int): The user ID for whom recommendations are to be generated.
        df (DataFrame): The dataframe containing user-book interactions.
        n (int): The number of recommendations to generate.

    Returns:
        DataFrame: A dataframe containing the top-n random recommended books, including
                   titles, authors, and publishers. No actual prediction scores are included
                   since the recommendation is based on a random generation.
    """
    # Identify books associated with the user and store their titles.
    user_books = df[df['user_id'] == uid].index
    user_book_ids = df[df.index.isin(user_books)].index.unique()
    unique_books = set()

    # Collect book details for user-interaction reference.
    for book_id in user_book_ids:
        book_title = df.loc[int(book_id), 'title'].iloc[0]
        unique_books.add((book_id, book_title))

    # Output books for user verification or selection.
    print(f"Books for user: {uid} with user_id: {uid}")
    for book_id, book_title in unique_books:
        print(f"Book ID: {book_id}. Book title: {book_title}")

    # User selects a book, typically to filter or focus recommendations.
    chosen_book_id = int(input("Enter the book_id of the book you want recommendations for: ").strip())
    user_items = df[df['user_id'] == uid].index.unique()
    all_items = df.index.unique()

    # Determine eligible books for recommendation.
    items_to_recommend = np.setdiff1d(all_items, user_items)

    # Estimate random predictions for these items.
    predictions = []
    for iid in items_to_recommend:
        pred = recommender.estimate(uid, iid)
        predictions.append({'book_id': iid, 'prediction': pred})

    # Sort predictions to find the highest random scores.
    predictions.sort(key=lambda x: x['prediction'], reverse=True)
    top_n_recommendations = predictions[:n]

    # Collect details for the top recommended books.
    top_n_titles = [df.loc[pred['book_id'], 'title'] for pred in top_n_recommendations]
    top_n_author = [df.loc[pred['book_id'], 'authors'] for pred in top_n_recommendations]
    author_names = []
    for authors_str in top_n_author:
        authors_list = ast.literal_eval(authors_str)
        names = [author['name'] for author in authors_list]
        author_names.append(names[0])
    top_n_publisher = [df.loc[pred['book_id'], 'publisher'] for pred in top_n_recommendations]

    # Prepare the recommendations in a DataFrame for output.
    data = {
        'title': top_n_titles,
        'author': author_names,
        'publisher': top_n_publisher
    }
    df_recommendations = pd.DataFrame(data)

    return df_recommendations


# Final Function

def finalfunc():
    """
    Determines the appropriate recommendation strategy based on the number of ratings a user has.
    Offers random, collaborative filtering, content-based, or a combination of popular and random
    recommendations based on user profile characteristics.

    Returns:
        DataFrame: A dataframe containing the recommended books for the user based on the chosen
                   recommendation strategy.

    Notes:
        - Random recommendations are used when a user has no rating history or based on a 1% chance
          for diversity.
        - Collaborative filtering is used for users with more than 20 ratings, leveraging their rich
          interaction history.
        - Content-based recommendations are used for users with fewer than 20 but more than 0 ratings,
          ideal for newer users still building their profile.
        - Combines popular and random recommendations as a fallback when no user-specific data is available.
    """
    user_id = input("Please enter your user id: ")
    num_ratings = int(all_df_filtered[all_df_filtered['user_id'] == user_id].shape[0])
    n = 10

    # Randomly decide if recommendations should be completely random, based on a 1% chance.
    if random.random() >= 0.99:
        print("Using the RANDOM PROBABILITY MODEL")
        df_recommendations = get_user_recommendations_Random_probability(random_model, user_id, all_df_filtered, n)
        return df_recommendations
    else:
        if num_ratings > 20:
            # Use the collaborative filtering model for users with substantial rating history.
            print("Using the COLLABORATIVE FILTERING MODEL")
            df_recommendations = get_user_recommendations_CF_context(collaborative_f_model, user_id, all_df_filtered, n)
            return df_recommendations
        elif num_ratings > 0:
            # Use the content-based model for users with some, but not extensive, rating history.
            print("Using the CONTENT MODEL")
            book_name, df_recommendations = get_user_recommendations_bert(user_id, all_df_filtered, filtered_books_df, bert_sim_df, n)
            return df_recommendations
        else:
            # For users with no ratings, use a mix of random and popular recommendations.
            print("Using the RANDOM + POPULAR MODEL")
            df_recommendations = random_popular_recommender(popular_model, random_model, user_id, all_df_filtered)
            return df_recommendations
        

#   user_id above 20 = 91efd74c298b00a55ef9bcf345ec9cc3
#   user_id below 20 = ca803a8c561eb0a4304e7a91a3841e50
#   user_id = NaN for Random Recommender

df_recommendations = finalfunc()
print(df_recommendations) # 20 just in case.
