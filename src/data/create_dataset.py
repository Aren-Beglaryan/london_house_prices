import os
import pickle

import pandas as pd
from google.cloud import bigquery
from google.oauth2 import service_account
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from src.data import data_processing

DATA_SAVE_PATH = 'data_files/'
CREDENTIALS_JSON_PATH = r"C:\Users\Sololearn\Desktop\london_house_price_prediction\configs\credentials_view.json"

BEDROOMS_MEAN = 2.6061767824355426
BEDROOMS_STD = 1.1694933172332571
LATITUDE_MEAN = 51.51417806518142
LATITUDE_STD = 0.06153182726811449
LONGITUDE_MEAN = -0.10760776920789263
LONGITUDE_STD = 0.09535970982651915
DAYS_FROM_MIN_DATE_MEAN = 6157.339425638108
DAYS_FROM_MIN_DATE_STD = 3006.2284278681286
DISTANCE_FROM_CENTER_MEAN = 8.967594045364942
DISTANCE_FROM_CENTER_STD = 3.3223895064504845


def load_data():
    """
    This function loads the data from server
    :return: pandas DataFrame
    """
    credentials = service_account.Credentials.from_service_account_file(
        CREDENTIALS_JSON_PATH, scopes=["https://www.googleapis.com/auth/cloud-platform"],
    )

    bqclient = bigquery.Client(credentials=credentials, project=credentials.project_id)
    table = bigquery.TableReference.from_string(
        "candidate-testing.house_data.london_house_prices"
    )
    rows = bqclient.list_rows(
        table
    )
    df = rows.to_dataframe()
    return df


def preprocess_data(df):
    """
    This function is doing cleaning and preprocessing
    :param df: pandas DataFrame
    :return: pandas DataFrame
    """
    df = data_processing.clean_data(df)
    # df = data_processing.remove_outlier(df, 'price')
    df = data_processing.address_processing(df)
    df = data_processing.create_column_road(df)
    df = data_processing.date_preprocess(df)
    df = data_processing.distance_from_center(df)
    df = data_processing.drop_columns(df)
    return df


def train_test_val_split(df):
    """
    This function splits the data into train, test, validation
    :param df: pandas DataFrame
    :return: tuple with 3 pandas DataFrame
    """
    train, test = train_test_split(df, test_size=0.3, random_state=42)
    val, test = train_test_split(test, test_size=0.5, random_state=42)
    return train, test, val


def one_hot_encode_fit_transform(df):
    """
    This function fits, transforms the one hot encoding and saves that
    :param df: pandas DataFrame
    :return: pandas DataFrame
    """
    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit(df[['type', 'area', 'tenure', 'road']])
    encodings = enc.transform(df[['type', 'area', 'tenure', 'road']]).toarray()
    cols = enc.get_feature_names()
    one_hot_data = pd.DataFrame(data=encodings, columns=cols, index=df.index)
    df = df.drop(['type', 'area', 'tenure', 'road'], axis=1)
    df = df.join(one_hot_data)
    with open("encoder", "wb") as f:
        pickle.dump(enc, f)
    return df


def one_hot_encode_transform(df,
                             encoder_path=r'C:\Users\Sololearn\Desktop\london_house_price_prediction\src\data\encoder'):
    """
    This function loads and transforms the one hot encoding
    :param df: pandas DataFrame
    :return: pandas DataFrame
    """
    with open(encoder_path, "rb") as f:
        enc = pickle.load(f)
    encodings = enc.transform(df[['type', 'area', 'tenure', 'road']]).toarray()
    cols = enc.get_feature_names()
    one_hot_data = pd.DataFrame(data=encodings, columns=cols, index=df.index)
    df = df.drop(['type', 'area', 'tenure', 'road'], axis=1)
    df = df.join(one_hot_data)
    return df


def normalize(df):
    """
    This function normalizes the numeric columns with given mean and std
    :param df:
    :return:
    """
    df.bedrooms = (df.bedrooms - BEDROOMS_MEAN) / BEDROOMS_STD
    df.latitude = (df.latitude - LATITUDE_MEAN) / LATITUDE_STD
    df.longitude = (df.longitude - LONGITUDE_MEAN) / LONGITUDE_STD
    df.days_from_min_date = (df.days_from_min_date - DAYS_FROM_MIN_DATE_MEAN) / DAYS_FROM_MIN_DATE_STD
    df.distance_from_center = (df.distance_from_center - DISTANCE_FROM_CENTER_MEAN) / DISTANCE_FROM_CENTER_STD
    return df


def save_data(train, test, val):
    if not os.path.exists(DATA_SAVE_PATH):
        os.makedirs(DATA_SAVE_PATH)

    train.to_csv(os.path.join(DATA_SAVE_PATH,'train.csv'), encoding='utf-8', index=False)
    test.to_csv(os.path.join(DATA_SAVE_PATH,'test.csv'), encoding='utf-8', index=False)
    val.to_csv(os.path.join(DATA_SAVE_PATH,'val.csv'), encoding='utf-8', index=False)


if __name__ == "__main__":
    print('started')
    df = load_data()
    print('data loaded')
    df = preprocess_data(df)
    print('data preprocessed')
    train, test, val = train_test_val_split(df)
    train = one_hot_encode_fit_transform(train)
    test = one_hot_encode_transform(test)
    val = one_hot_encode_transform(val)
    print('one hot encoded')
    train = normalize(train)
    test = normalize(test)
    val = normalize(val)
    print('normalized')
    save_data(train, test, val)
    print('saved')