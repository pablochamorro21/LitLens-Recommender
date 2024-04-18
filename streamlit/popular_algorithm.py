import numpy as np
from surprise.prediction_algorithms.algo_base import AlgoBase
import pandas as pd

class MyPopularAlgorithm(AlgoBase):

    def __init__(self):
        AlgoBase.__init__(self)

    def estimate(self, u, i):
        # Note u is not used, this RS does not personalize based on the user
        if i in self.mean_rating_per_item_df.index:
          return self.mean_rating_per_item_df.loc[i]['rating']
        else:
          return self.trainset.global_mean

    def fit(self, trainset):
        AlgoBase.fit(self, trainset)

        ratings_df = pd.DataFrame([[i, r] for (_, i, r) in self.trainset.all_ratings()],
                                  columns=['item', 'rating'])

        self.mean_rating_per_item_df = (ratings_df
          .groupby('item')
          .agg({'rating': 'mean'})
        )

        return self
