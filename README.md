# LitLens Recommender

<img src="./logo.png" width="389" height="357">

## Overview
LitLens is a sophisticated book recommendation engine designed for a literature e-commerce platform, aiming to connect readers with literature that resonates with their interests. LitLens stands out by not relying on a single algorithm but by blending different techniques to create a more accurate and personalised shopping experience: non-personalised, collaborative filtering, content-based, and context-aware algorithms. This multidimensional approach is tailored to enhance user engagement and satisfaction on literature e-commerce sites, ultimately driving sales through smarter recommendations.

## Data Preparation

​​The dataset used was sourced from a scrapped Goodreads comprehensive dataset from 2017: https://mengtingwan.github.io/data/goodreads#datasets

It was of a size too big for us to handle : 228,648,342 interactions, 2,360,655 items, and 876,145 users. We initially loaded manageable subsets of the large dataset and enhanced processing efficiency by removing irrelevant columns and records with missing values in key columns. After merging several subsets, we also eliminated duplicates to preserve data integrity.

The resulting dataset contained details on 25,833 books. To enrich these details, we integrated additional datasets with information about authors and book genres. We focused on a subset of 376,978 reviews that included ratings and written feedback, anticipating their utility for potential NLP applications. Furthermore, we assumed these would be of higher quality and reliability than a review with no text. However, we later removed another 57,000 reviews due to date issues.

The original dataset lacked comprehensive user data, prompting us to create a synthetic table listing each user's ID and their number of reviews for 35,692 users. Finally, we converted the data from JSON to CSV format for easier access and manipulation in future analyses.

Our final data can be found:
- https://drive.google.com/file/d/1vbjF4F5D2gJnZXmRmT0_j-O_wCM6iaLB/view?usp=drive_link
- https://drive.google.com/file/d/1li7yAIjZkc-YJQH9YFxV2zo7BhIM-kX7/view?usp=sharing
  
### Data Exploration

- Ratings predominantly range from 4/5 to 5/5, likely due to selection bias as users typically choose books they expect to enjoy.
- The mean rating of 3.75 with a standard deviation of 1.2248 suggests a preference for higher ratings, yet with a notable spread.
- This variance implies potential model biases towards higher-rated books, which could hinder the exploration of a broader range of literature (**popularity bias**). 

- On average, books received 15 reviews, but 5% of books accounted for 65% of all ratings, illustrating the **"long tail" effect**.
- This skew towards a few books could cause recommendation systems to favour these popular choices, overshadowing lesser-known titles.
- Addressing this, mechanisms need to be implemented to ensure that long-tail books with fewer reviews are also recommended, as they play a significant role in diversifying the offerings and sales. 

- Fiction dominates the dataset, making up 24% of the books, followed by history (14%) and romance (12%). 
- The wide range of genres reflects diverse reader interests, necessitating a recommendation system adept at handling various literary categories.
- While popular genres like fiction are more frequently recommended, the system **should also cater to niche genres** such as poetry and comics to serve all reader segments effectively.

## Model Development

- Before implementing the algorithms, we used a new dataframe which only included the **users with more than 20 ratings**
- Users with more ratings tend to provide more stable and reliable data. By focusing on these users, the recommender system can make use of a dataset that is less susceptible to randomness or biases.
- The dataset was also is **split into training and test sets**, with 20% of the data reserved for testing and the rest for training, using a predefined random seed for reproducibility.

### Non-personalised: Popular and Random

**random_RS**: Our random algorithm calculates the mean and standard deviation of the ratings in the training set. These statistics are used to generate predictions with an estimate method, which generates a predicted rating. It outputs a random rating value drawn from a normal distribution centred around the mean with variability defined by the standard deviation (no personalisation based on user or item data). This recommender has the lowest accuracy of all but can be used to introduce diversity, coverage, and serendipity.

**popular_RS**: Our popular algorithm recommends items by calculating their average ratings from all users. The fit method computes each item's mean rating from the training dataset. This calculation is stored in a dataframe. The estimate method predicts an item's rating: it returns the item's mean rating if available, or the global mean otherwise. Like the random recommender, it can provide an effective strategy for establishing baselines and tackling cold-start problems.

### Collaborative filtering

For choosing our CF algorithm, we first ranked every single algorithm available in surprise, performing cross-validation on the dataset with 5 folds and ranking based on RMSE. We tried:
- **Memory based**: KNNBasic, KNNWithMeans, KNNWithZScore, KNNBaseline, both item-based and user-based versions
- **Model based**: SVD, SVDpp, NMF, CoClustering.
- The SVD has the best performance of all, followed closely by the item-based KNNBaseline.
- We decided to do a Grid search hyperparameter tuning for both before making a decision. 

For **SVD**, the grid included various values for:
- n_factors (the number of latent factors)
- n_epochs (the number of iterations over the entire dataset)
- lr_all (the learning rate for all parameters)
- reg_all (the regularisation term for all parameters)
- It uses 3-fold cross-validation to evaluate each combination and computes the RMSE and other metrics.
- After the grid search, the best parameters for minimising RMSE are extracted and the score is generated.

| Parameter  | Value  |
|------------|--------|
| n_factors  | 1      |
| n_epochs   | 20     |
| lr_all     | 0.005  |
| reg_all    | 0.02   |
| **Best RMSE** | **1.0531** |

For the **knnBaseline (item-based)** hyperparameter tuning:
- There are bsl_options, which are the baseline estimates configuration, the method can be either ALS or SGD
- The regularisation term reg has several values to choose from.
- After the grid search, the best parameters that led to the lowest RMSE and the corresponding RMSE score are retrieved. 

| Parameter | Value |
|-----------|-------|
| method    | als   |
| reg       | 1     |
| **Best RMSE** | **1.0902** |

The **SVD model is slightly better and was selected**, and we trained it with our whole data and pickled it for easy access.

**Collaborative_f_model**: Our collaborative filtering algorithm uses Singular Value Decomposition (SVD) configured with a minimal complexity and a focus on avoiding overfitting. Following hyperparameter tuning, it was set to 1 latent factor, as it proved to be the best in terms of RMSE, 20 training epochs, a learning rate for all parameters of 0.005, and a regularisation for all parameters is set to 0.02, adding a penalty for larger parameters to the optimization process to further prevent overfitting.

### Content-based model

**BERT**: A pre-trained model, SentenceTransformer, specifically distilbert-base-nli-mean-tokens, is used to convert book descriptions into embeddings. The embeddings for the book descriptions are computed and then put into a DataFrame. The compute_similarities function calculates the cosine pairwise distances between all book embeddings, creating a similarity matrix. The top_n function generates top n recommendations for a given book item. We then used this to calculate the NDCG. The function get_user_recommendations_bert provides personalised book recommendations to a user based on a similarity matrix computed with BERT embeddings.

**Categorical**: A new DataFrame df_books_cat is created with the publisher, author name, and genre names. A “soup” of categorical features is created by joining the publisher, author, and genre names into a single string. The CountVectorizer is used to transform the “soup” into a matrix of token counts, representing the presence of categorical features. Cosine similarity is computed from the count matrix to determine the similarity between books based on categorical features. We then used this to calculate the NDCG. The function get_user_recommendations_categories provides the  book recommendations on the similarity matrix generated with the 3 categorical features.

**Based on the NDCG, we selected the BERT model going forward**.

**BERT:**

| Item       | NDCG       |
|------------|------------|
| Item 1     | 0.8036     |
| Item 2     | 0.7332     |

**Categorical:**

| Item       | NDCG       |
|------------|------------|
| Item 1     | 0.2000     |
| Item 2     | 0.4894     |

### Contextual pre-filtering

Focusing on seasonal/monthly data allows the model to better capture seasonal trends and events that influence reading habits, improving the accuracy of recommendations. It also helps the model adapt to timely changes in book popularity and user interests, reflecting current relevancies.

For the contextual evaluation, the script iterates over a dictionary that contains the different subsets split on **season**. The SVD model is re-trained and re-evaluated on each subset (split by seasons, and the NDCG score for each context is printed out. Since there is an improvement for seasons, it was tried with **months^^, which performed even better and was **selected as the one to combine with our CF model**. 

Here are the formatted markdown tables for both the seasonal and monthly contextual pre-filtering:

**SVD + Season Contextual Pre-Filtering:**

| Season  | NDCG   |
|---------|--------|
| Winter  | 0.9509 |
| Spring  | 0.9540 |
| Summer  | 0.9501 |
| Autumn  | 0.9548 |

**SVD + Month Contextual Pre-Filtering:**

| Month     | NDCG   |
|-----------|--------|
| January   | 0.9613 |
| February  | 0.9583 |
| March     | 0.9650 |
| April     | 0.9624 |
| May       | 0.9601 |
| June      | 0.9590 |
| July      | 0.9579 |
| August    | 0.9557 |
| September | 0.9621 |
| October   | 0.9623 |
| November  | 0.9603 |
| December  | 0.9601 |

## Final Code

First, install these libraries in your environment:

```pip install pandas numpy sentence-transformers scikit-learn scikit-surprise``` 

If you want to use the streamlit app, read further on.

### Logic
- Users with fewer than 20 ratings but more than zero are in a transitional phase where they have started interacting with the system but haven't provided enough data for an effective collaborative filtering.
- The content-based BERT model doesn't rely on other users' data but instead focuses on the content of the items themselves, so it does not depend on the interactions.
- Therefore, the **users with fewer than 20 ratings will receive recommendations from the content-based model**, and the **users with more than 20 from the collaborative filtering model**.
- This tiered approach ensures that all users receive good quality personalised recommendations regardless of how many interactions they've logged.
- Furthermore, for **new users**, we deal with the **cold-start problem** by recommending a shuffled dataframe containing a combined list of recommendations from both the popularity-based (using the top 50 items), and random recommenders.

### Process

The final code firstly loads the two dataframes, books_df (books metadata) and reviews_df (user-item interaction), and then converts the timestamp format and categorises each month for the contextual filtering.

**Loading models**:
- **Collaborative filtering model + Context**: A trained and tuned SVD, loaded from a pickle file to not have to re-train.
- **Random model**: Also loaded from a pickled file, but likewise loaded with the Random Class (as if not it doesn’t understand the different methods).
- **Popular model**: Also loaded from a pickle file, measured by the average rating.

**Functions**:
- **Random Recommendation (MyRandomAlgorithm)**: This class overrides the estimate method of AlgoBase to return a random rating derived from a normal distribution centered at the training data’s mean rating with its standard deviation. 
- **Popular Recommendation (MyPopularAlgorithm)**: This class calculates the average rating for each item during the fitting process and stores it. The estimate method then provides predictions based on this precomputed average
- **Collaborative Filtering with Contextual Data**: Generates recommendations based on a user’s historical interactions and contextual information (the month). It selects books that are contextually relevant (e.g., books read or popular in a particular month) and predicts user preferences for books they haven't interacted with yet.
- **BERT Content-Based Model**: Utilizes the SentenceTransformer model to convert book descriptions into numerical embeddings. These embeddings are then used to calculate cosine similarities between books. A function computes these similarities and uses them to suggest books similar to those the user has shown interest in, based on content.

**Combining Recommendations**
- The system decides the recommendation strategy based on the number of ratings a user has:
- Users with more than 20 ratings: Uses the collaborative filtering  + context model.
- Users with less than 20 ratings, but more than 0: Uses the content-based BERT model.
- Users with no ratings (user id = NaN): leveraging a mixture of randomness, to explore diverse options, and popularity, to ensure some level of user satisfaction from generally liked books. This combined approach balances exploration and exploitation for the cold-start problem. 
  - **Random probability**: For any user, there is a 1% chance that the recommendations are random. By introducing random recommendations, the system can break away from the cycle of frequently recommended items, thereby giving less popular or newer items a chance to be seen. This approach helps increase the coverage of the catalog. It allows allows us to test how users respond to different types of content outside their usual consumption patterns

**Final Recommendation Function (finalfunc)**
- **User Interaction**: Prompts the user to input a user ID.
- **Recommendation Decision Logic**: Decides which recommendation approach to use based on the number of ratings associated with the user ID.
- Executes the appropriate recommendation function and displays the results.
  - Potentially, you could specify a book for more targeted suggestions.

### Streamlit 

First, install streamlit on top of the previously mentioned libraries:

```pip install pandas numpy sentence-transformers scikit-learn scikit-surprise streamlit``` 

Open the folder ./streamlit, and execute the .py file called streamlit_app.py:

```streamlit run streamlit_app.py```

If it does not automatically open, run http://localhost:8501 in your browser. 

## Conclusion

To ensure optimal performance, our monitoring strategy will encompass tracking critical metrics such as response times, system throughput, and error rates using tools like Datadog for real-time analytics and alerts. We rigorously check data quality and flag anomalies automatically, complemented by A/B testing and continuous evaluation of model performance. 
For future developments, we will implement strategies such as bandit algorithms for balancing exploration and exploitation, and demographic filtering to address the cold start problem by using user demographic data to tailor initial recommendations. 
LitLens will iterate and improve, combining advanced recommendation strategies to deliver more tailored and personalised book suggestions, enhancing user engagement and increasing sales.

