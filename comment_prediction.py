import argparse
import json
import gzip
import os
import numpy as np
from sklearn.linear_model import  Ridge
import pandas as pd
from nltk.tokenize import word_tokenize
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from lemmagen3 import Lemmatizer
from urllib.parse import urlsplit
from sklearn.preprocessing import MultiLabelBinarizer, LabelBinarizer, StandardScaler
from scipy.sparse import hstack, csr_matrix
from nltk.stem import WordNetLemmatizer

def read_json(data_path: str) -> list:
    with gzip.open(data_path, 'rt', encoding='utf-8') as f:
        return json.load(f)


class RTVSlo:

    def __init__(self):
        self.model = Ridge(alpha=1.0)
        # self.lemmatizer = Lemmatizer("sl")
        self.lemmatizer = WordNetLemmatizer()
        self.vectorizer = TfidfVectorizer()
        self.item_list_url = []
        self.item_list_authors = []
        self.lb_topics = None
        self.lb_day_of_week = None

    def preprocess_text(self, text):
        try:
            split_result = urlsplit(text)
            if split_result.scheme and split_result.netloc:
                # Split the path by '/' and then filter segments based on length
                path_segments = split_result.path.strip('/').split('/')
                subtopics = []
                for segment in path_segments[:-2]:
                    subtopics.append(segment)
                return subtopics
        except ValueError:
            pass
        words = word_tokenize(text.lower()) 

        table = str.maketrans('', '', string.punctuation)
        words = [word.translate(table) for word in words if word.isalpha()]

        stop_words = set(stopwords.words('slovene'))
        words = [word for word in words if word not in stop_words]

        lemmatized_words = [self.lemmatizer.lemmatize(word) for word in words]

        preprocessed_text = ' '.join(lemmatized_words)
        return preprocessed_text
        
    def combine_figure_captions(self, figures):
        captions = [figure.get('caption', '') for figure in figures]
        return ' '.join(captions)
    
    def get_one_hot_encoded(self, data, data_type, column):
        if data_type == "test": 
            if column == "url":
                item_list = self.item_list_url
                mlb = MultiLabelBinarizer(classes=item_list)
                one_hot_encoded = mlb.fit_transform(data)
            elif column == "authors":
                item_list = self.item_list_authors
                mlb = MultiLabelBinarizer(classes=item_list)
                one_hot_encoded = mlb.fit_transform(data)
            elif column == "topics":
                one_hot_encoded = self.lb_topics.transform(data)
            else:
                one_hot_encoded = self.lb_day_of_week.transform(data)
        else:
            if column == "day_of_week":
                item_list = list(set(data))
                self.lb_day_of_week = LabelBinarizer()
                self.lb_day_of_week.fit(item_list)
                one_hot_encoded = self.lb_day_of_week.transform(data)
            elif column == "topics":
                item_list = list(set(data))
                self.lb_topics = LabelBinarizer()
                self.lb_topics.fit(item_list)
                one_hot_encoded = self.lb_topics.transform(data)
            else:
                item_set = set(item for element in data for item in element if item)
                item_list = list(item_set)
                if column == "url":
                    self.item_list_url = item_list
                else:
                    self.item_list_authors = item_list
                mlb = MultiLabelBinarizer(classes=item_list)
                one_hot_encoded = mlb.fit_transform(data)

        return one_hot_encoded


    
    def fit(self, train_data: list):
        df = pd.DataFrame(train_data)
        df.dropna(subset=['topics'], inplace=True)
        df['combined_paragraphs'] = df['paragraphs'].apply(lambda x: ' '.join(x))
        df['combined_keywords'] = df['keywords'].apply(lambda x: ' '.join(x))
        df['combined_figure_captions'] = df['figures'].apply(self.combine_figure_captions)

        df['date'] = pd.to_datetime(df['date'])
        df['hour'] = df['date'].dt.hour
        df['day_of_week'] = df['date'].dt.day_name()
        df['n_figures']= df['figures'].apply(len)
        df['n_paragraphs']= df['paragraphs'].apply(len)


        df['clean_title'] = df['title'].apply(self.preprocess_text)
        df['clean_lead'] = df['lead'].apply(self.preprocess_text)
        df['clean_combined_paragraphs'] = df["combined_paragraphs"].apply(self.preprocess_text)
        df['clean_topics'] = df['topics'].apply(self.preprocess_text)
        df['clean_combined_keywords'] = df['combined_keywords'].apply(self.preprocess_text)
        df['clean_combined_figure_captions'] = df['combined_figure_captions'].apply(self.preprocess_text)
        df['clean_url'] = df['url'].apply(self.preprocess_text)

        df['combined_text'] = (
            df['clean_title'] + ' ' +
            df['clean_lead'] + ' ' +
            df['clean_combined_paragraphs'] 
            # df['clean_combined_keywords'] + ' ' +
            # df['clean_combined_figure_captions']
        )

        df['article_length']= df['combined_text'].apply(len)

        one_hot_subtopics = csr_matrix(self.get_one_hot_encoded(df['clean_url'],"train", "url"))

        one_hot_authors = csr_matrix(self.get_one_hot_encoded(df['authors'], "train", "authors"))

        one_hot_topics = csr_matrix(self.get_one_hot_encoded(df['clean_topics'], "train", "topics"))

        one_hot_day_of_week = csr_matrix(self.get_one_hot_encoded(df['day_of_week'], "train", "day_of_week"))

        # Convert continuous features to sparse matrices
        hour_sparse = csr_matrix(df['hour']).T

        article_length_sparse = csr_matrix(df['article_length']).T

        n_figures_sparse = csr_matrix(df['n_figures']).T

        n_paragraphs_sparse = csr_matrix(df['n_paragraphs']).T


        X_text_vectorized = self.vectorizer.fit_transform(df['combined_text'])

        # Combine all features into a single sparse matrix
        X_sparse = hstack([X_text_vectorized, one_hot_subtopics, one_hot_authors, one_hot_topics,
                        one_hot_day_of_week, hour_sparse, article_length_sparse, 
                        n_figures_sparse, n_paragraphs_sparse]).tocsr()


        # Define continuous features' indices in the stacked sparse matrix
        continuous_start_idx = X_text_vectorized.shape[1] + one_hot_subtopics.shape[1] + one_hot_authors.shape[1] + one_hot_topics.shape[1] + one_hot_day_of_week.shape[1]
        continuous_end_idx = continuous_start_idx + 4  # Assuming there are 4 continuous features

        # Extract continuous features
        X_continuous = X_sparse[:, continuous_start_idx:continuous_end_idx]

        # Normalize continuous features
        scaler = StandardScaler(with_mean=False)  # with_mean=False to maintain sparsity
        X_continuous_normalized = scaler.fit_transform(X_continuous)

        # Replace the continuous features in the combined sparse matrix with the normalized ones
        X_sparse_normalized = hstack([X_sparse[:, :continuous_start_idx], X_continuous_normalized])

        # Split the data into training and testing sets
        y = np.sqrt(df['n_comments'])

        self.model.fit(X_sparse_normalized, y)        

        
    def predict(self, test_data: list) -> np.array:
        df = pd.DataFrame(test_data)
        df.dropna(subset=['topics'], inplace=True)
        df['combined_paragraphs'] = df['paragraphs'].apply(lambda x: ' '.join(x))
        df['combined_keywords'] = df['keywords'].apply(lambda x: ' '.join(x))
        df['combined_figure_captions'] = df['figures'].apply(self.combine_figure_captions)
        df['date'] = pd.to_datetime(df['date'])
        df['hour'] = df['date'].dt.hour
        df['day_of_week'] = df['date'].dt.day_name()
        df['n_figures']= df['figures'].apply(len)
        df['n_paragraphs']= df['paragraphs'].apply(len)

        df['clean_title'] = df['title'].apply(self.preprocess_text)
        df['clean_lead'] = df['lead'].apply(self.preprocess_text)
        df['clean_combined_paragraphs'] = df["combined_paragraphs"].apply(self.preprocess_text)
        df['clean_topics'] = df['topics'].apply(self.preprocess_text)
        df['clean_combined_keywords'] = df['combined_keywords'].apply(self.preprocess_text)
        df['clean_combined_figure_captions'] = df['combined_figure_captions'].apply(self.preprocess_text)
        df['clean_url'] = df['url'].apply(self.preprocess_text)

        df['combined_text'] = (
            df['clean_title'] + ' ' +
            df['clean_lead'] + ' ' +
            df['clean_combined_paragraphs']
            # df['clean_combined_keywords'] + ' ' +
            # df['clean_combined_figure_captions']
        )

        df['article_length']= df['combined_text'].apply(len)

        one_hot_subtopics = csr_matrix(self.get_one_hot_encoded(df['clean_url'],"test", "url"))

        one_hot_authors = csr_matrix(self.get_one_hot_encoded(df['authors'], "test", "authors"))

        one_hot_topics = csr_matrix(self.get_one_hot_encoded(df['clean_topics'], "test", "topics"))

        one_hot_day_of_week = csr_matrix(self.get_one_hot_encoded(df['day_of_week'], "test", "day_of_week"))

        # Convert continuous features to sparse matrices
        hour_sparse = csr_matrix(df['hour']).T

        article_length_sparse = csr_matrix(df['article_length']).T

        n_figures_sparse = csr_matrix(df['n_figures']).T

        n_paragraphs_sparse = csr_matrix(df['n_paragraphs']).T


        X_text_vectorized = self.vectorizer.transform(df['combined_text'])

        # Combine all features into a single sparse matrix
        X_sparse = hstack([X_text_vectorized, one_hot_subtopics, one_hot_authors, one_hot_topics,
                        one_hot_day_of_week, hour_sparse, article_length_sparse, 
                        n_figures_sparse, n_paragraphs_sparse]).tocsr()

        # Define continuous features' indices in the stacked sparse matrix
        continuous_start_idx = X_text_vectorized.shape[1] + one_hot_subtopics.shape[1] + one_hot_authors.shape[1] + one_hot_topics.shape[1] + one_hot_day_of_week.shape[1]
        continuous_end_idx = continuous_start_idx + 4 

        # Extract continuous features
        X_continuous = X_sparse[:, continuous_start_idx:continuous_end_idx]

        # Normalize continuous features
        scaler = StandardScaler(with_mean=False)  # with_mean=False to maintain sparsity
        X_continuous_normalized = scaler.fit_transform(X_continuous)

        # Replace the continuous features in the combined sparse matrix with the normalized ones
        X_sparse_normalized = hstack([X_sparse[:, :continuous_start_idx], X_continuous_normalized])

        return  self.model.predict(X_sparse_normalized) ** 2

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('train_data_path', type=str)
    parser.add_argument('test_data_path', type=str)
    args = parser.parse_args()

    train_data = read_json(args.train_data_path)
    test_data = read_json(args.test_data_path)

    rtv = RTVSlo()
    rtv.fit(train_data)
    predictions = rtv.predict(test_data)

    # y_test = pd.DataFrame(test_data)['n_comments']
    # r2 = r2_score(y_test, predictions)
    # mae = mean_absolute_error(y_test, predictions)
    # print("R2 score:", r2)
    # print("MAE score:", mae)

    if os.path.exists('predictions.txt'):
        os.remove('predictions.txt')

    np.savetxt('predictions.txt', predictions)

if __name__ == '__main__':
    main()
