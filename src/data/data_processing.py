import re

import geopy.distance
import numpy as np
import pandas as pd


def clean_data(df):
    """
    This function drops the duplicates and data issues from our data
    :param df: pandas DataFrame
    :return: pandas DataFrame
    """
    df = df.drop_duplicates().reset_index(drop=True)
    df = df[~((df.latitude == 0) & (df.longitude == 0))]

    # if everything is same except type we have to drop that rows
    df = df.drop_duplicates([col for col in df.columns if col != 'type'], keep=False)

    # if everything is same except bedrooms we have to drop that rows
    df = df.drop_duplicates([col for col in df.columns if col != 'bedrooms'], keep=False)

    # if everything is same except latitude and longitude we have to drop that rows
    df = df.drop_duplicates([col for col in df.columns if col != 'latitude' and col != 'longitude'])

    # if everything is same except area we have to drop that rows
    df = df.drop_duplicates([col for col in df.columns if col != 'area'], keep=False)

    # if everything is same except price we have to drop that rows
    df = df.drop_duplicates([col for col in df.columns if col != 'price'], keep=False)

    # if everything is same except tenure we have to drop that rows
    df = df.drop_duplicates([col for col in df.columns if col != 'tenure'], keep=False)

    # if everything is same except is_newbuild we have to drop that rows
    df = df.drop_duplicates([col for col in df.columns if col != 'is_newbuild'], keep=False)

    return df


# def remove_outlier(df, col_name):
#     """
#     This function removes outliers with IQR method depending on col_name
#     :param df: pandas DataFrame
#     :param col_name: string, column from df
#     :return: pandas DataFrame
#     """
#     q1 = df[col_name].quantile(0.25)
#     q3 = df[col_name].quantile(0.75)
#     iqr = q3 - q1  # Interquartile range
#     low = q1 - 1.5 * iqr
#     high = q3 + 1.5 * iqr
#     df = df.loc[(df[col_name] > low) & (df[col_name] < high)]
#     return df


def address_processing(df):
    """
    This function makes the addresses lowercase and deletes the ' s
    :param df: pandas DataFrame
    :return: pandas DataFrame
    """
    df['address'] = df['address'].apply(lambda x: x.lower().replace("'", ""))
    return df


def _extract_road(x):
    """This function extracts the road/street/... from input string
    """
    road_matcher = re.compile(r'(\w+\s?\w+),\s?london,')
    res = road_matcher.findall(x)
    if res:
        return res[0]
    return None


def create_column_road(df):
    """
    This function creates road column in df which extracts from address column
    :param df: pandas DataFrame
    :return: pandas DataFrame
    """
    df['road'] = df['address'].apply(_extract_road)
    top5_roads = df['road'].value_counts().index[:5]
    df['road'] = np.where(df['road'].isin(top5_roads), df['road'], 'Other')
    return df


def date_preprocess(df):
    """
    This function creates days_from_min_date column in df
    which is integers and shows difference of the date and minimum date in our data
    :param df: pandas DataFrame
    :return: pandas DataFrame
    """
    day = pd.to_datetime(np.datetime64(min(df.date))).tz_localize(None)
    df['date'] = df['date'].dt.tz_localize(None)
    df['min_date'] = day
    df['days_from_min_date'] = (df.date - df.min_date).dt.days
    return df


def _distance_from_center(coordinate):
    """
    This function calculates the distance of given coordinates from center of London
    :param coordinate: list of 2 floats
    :return: float
    """
    latitude = float(coordinate.split(',')[0])
    longitude = float(coordinate.split(',')[1])
    center = [51.509865, -0.118092]
    coordinate = [latitude, longitude]
    distance = geopy.distance.distance(center, coordinate).km
    return distance


def distance_from_center(df):
    """
    This function adds distance_from_center column in df which is the distance in kilometers
    :param df:pandas DataFrame
    :return: pandas DataFrame
    """
    df['coordinate'] = df['latitude'].astype(str) + ',' + df['longitude'].astype(str)
    df['distance_from_center'] = df.coordinate.apply(_distance_from_center)
    df = df.drop('coordinate', axis=1)
    return df


def drop_columns(df):
    """
    This function drops the unnecessary columns
    :param df: pandas DataFrame
    :return: pandas DataFrame
    """
    df = df.drop(['address', 'date', 'min_date'], axis=1)
    return df
