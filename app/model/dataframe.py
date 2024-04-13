# imports
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import KNNImputer


class IMDB_df(pd.DataFrame):
    """
    add methods to dataframe
    """
    df = None

    @staticmethod
    def get_df():
      if IMDB_df.df is None:
        df = IMDB_df.load_imdb()\
                    .drop_dup()\
                    .to_bin("imdb_score")\
                    .str_to_num([*IMDB_df.str_columns(), "imdb_score"])\
                    .drop_useless()\
                    .fill_na()
      return df
    
    @staticmethod
    def see_all(n=None):
      """
      show n lines from dataframe
      """
      pd.set_option('display.max_columns', n)
      pd.set_option('display.max_rows', n)


    @property
    def _constructor(self):
        return IMDB_df

    @staticmethod
    def load_imdb():
      """
      load IMDB dataset into df
      """
      df = IMDB_df(pd.read_csv("https://raw.githubusercontent.com/louiskuhn/IA-P3-Euskadi/main/Projets/Projet%20P5%20-%20IMDB/5000_movies_bis.csv"))
      df["director_name"] = df["director_name"].str.strip()
      df["movie_title"] = df["movie_title"].str.strip()
      return df

    def replace(self):
      """
      replace minor elements from language and country
      """
      self["language"] = self["language"].where(self["language"] == "English", "other")
      self["language"] = self["language"].astype(object)

      self["country"] = self["country"].where(self["country"].isin(["UK", "USA"]), "other")
      self["country"] = self["country"].astype(object)

      return self

    def encode_cat(self):
      """
      encode string variables, you should prefer str_to_num
      """
      encoder = LabelEncoder()
      columns = self.dtypes[(self.dtypes == "object")].index
      self[columns] = self[columns].apply(lambda x: encoder.fit_transform(x))
      return self

    def drop_useless(self):
      """
      drops "cast_total_fb_likes","plot_keywords", "movie_imdb_link", "gross", "movie_fb_likes"
      """
      self.drop(["cast_total_fb_likes","plot_keywords", "movie_imdb_link", "gross", "movie_fb_likes"], axis=1, inplace=True)
      return self

    def drop_dup(self):
      """
      drop duplicates
      """
      self.drop_duplicates(inplace=True)
      self.drop_duplicates(subset=["movie_title", "director_name"], inplace=True)
      return self

    def fill_na(self):
      """
      fill NaN values based on KNN
      """
      imputer = KNNImputer(n_neighbors=5)
      filled = imputer.fit_transform(self.values)
      new_df = IMDB_df(filled, columns=self.columns)
      new_df.num_to_str = self.num_to_str
      return new_df

    @staticmethod
    def str_columns():
      """
      list of str columns
      """
      return ["color", "director_name", "actor_1_name", "actor_2_name", "actor_3_name", "genres", "movie_title", "content_rating", "country", "language", "movie_imdb_link", "plot_keywords"]

    def to_int(self, label):
      """
      convert a column to int
      """
      self[label] = self[label].astype(int)
      return self

    def to_bin(self, label, bins=[i for i in range(11)], labels=["awful", "bad", "average", "interesting", "good"]):
      """
      convert column to bin (for classification)
      """
      self[label] = pd.cut(self[label], range(0, 11, 2), labels=labels, right=True)
      return self

    def str_to_num(self, columns):
      """
      converts all str columns to label-encoded.
      A dict referencing index and str will be hold by th e class object
      """
      #self.loc["color"] = self["color"].str.lower()
      #self["color"].str.strip()
      if not hasattr(self, "num_to_str"):
        self.num_to_str = {}

      for column in columns:
        values = self[column].value_counts() # index = ['color', nan, 'black and white']
        value_dict = {i: index for index, i in enumerate(values.index)}
        value_dict[np.NaN] = -1
        self.num_to_str[column] = {value: key for key, value in value_dict.items()}

        self[column] = self[column].apply(lambda x: value_dict[x])
        self[column] = self[column].astype(float)
      return self


    def top_correlated(self, label, n=10):
      """
      gets top n correlated column for a given column
      """
      corr = self.corr()
      most_correlated = self.corr()[label]
      most_correlated = most_correlated.sort_values(ascending=False)[1:]
      return most_correlated[:n].index

#example
IMDB_df.see_all(10)
df = IMDB_df.load_imdb()\
            .drop_dup()\
            .to_bin("imdb_score")\
            .str_to_num([*IMDB_df.str_columns(), "imdb_score"])\
            .fill_na()
