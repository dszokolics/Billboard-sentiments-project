import billboard
import datetime as dt
import pandas as pd
import json
from tqdm import tqdm
import time
import requests
import re
from bs4 import BeautifulSoup
import boto3


def get_billboard_hot100(year_start,
                         year_end=dt.date.today().year,
                         df=pd.DataFrame(columns=['year', 'month']),
                         tries=3):
    """Collects Billboard Hot-100 songs at the month starts for the specified
    years.

    Args:
        year_start (int): First year to collect.
        year_end (int): Last year to collect.
        df (pd.DataFrame): Data collected so far. The function handles
            exceptions recursively. If an error occured during the collection,
            it will call itself at the end.
        tries (int): Number of recursive iterations.

    Returns:
        pd.DataFrame: Data with Billboard Hot-100 songs.

    """

    df = df.copy()
    df_list = []
    err = False
    for year in tqdm(range(year_start, year_end+1)):
        for month in range(1, 13):

            if sum((df['year'] == year) & (df['month'] == month)) == 0:
                try:
                    cd = billboard.ChartData(name="hot-100",
                                             date=dt.date(year, month, 1),
                                             timeout=45)
                    data = pd.DataFrame(json.loads(cd.json())['entries'])
                    data['year'] = year
                    data['month'] = month
                    df_list.append(data)
                except Exception as e:
                    print(e)
                    if (year != 2019) | (month < 12):
                        err = True

                time.sleep(10)

    df_list.append(df)
    df = pd.concat(df_list)

    if (err is True) and (tries > 0):
        df = df.append(get_billboards(year_start, year_end, df, tries=tries-1))

    return df


def get_lyrics_genius(artist, song_title):
    """Get a lyrics for a given song from genius.com

    Args:
        artist (str): Name of the artist.
        song_title (str): Title of the song.

    Returns:
        str on None: lyrics of the song (if the retrieval was successful).

    """

    artist = artist.lower()
    song_title = song_title.lower()
    # remove all except alphanumeric characters from artist and song_title
    artist = re.sub('[^A-Za-z0-9\s]+', "", artist)
    artist = artist.replace(" ", "-")
    song_title = re.sub('[^A-Za-z0-9\s]+', "", song_title)
    song_title = song_title.replace(" ", "-")

    url = f"http://genius.com/{artist}-{song_title}-lyrics"

    try:
        content = requests.get(url)
        soup = BeautifulSoup(content.text, 'html.parser')
        lyrics = soup.find_all(attrs={"class": "lyrics"})[0].get_text()
        return lyrics

    except Exception as e:
        print("Exception occurred \n" + str(e))
        return None


def get_lyrics(artists, titles):
    """Get lyrics for a bunch of songs.

    Args:
        artists (list of str): Artist names.
        titles (list of str): Song titles.

    Returns:
        list: list of lyrics.

    """

    return [get_lyrics_genius(a, t) for a, t in zip(artists, titles)]


def aws_get_sentiment(comprehend, text):
    res = comprehend.detect_sentiment(Text=text, LanguageCode='en')
    return res['SentimentScore']


def get_sentiments(comprehend, lyrics):
    """Collects the sentiment predictions from AWS for a list of lyrics.

    Args:
        lyrics (list of str): Lyrics of songs.

    Returns:
        pd.DataFrame: Sentiment scores for the lyrics.

    """

    sentiments = list()
    for lyric in tqdm(lyrics):
        try:
            # Get the sentiments for each lines, and average them.
            lyric = lyric.replace("<br/>", "").split("\n")
            lyric = ([l for l in lyric if (len(l) > 1) and (not l.startswith('['))])

            snts = [aws_get_sentiment(comprehend, l) for l in lyric]
            snts = pd.DataFrame(snts).mean().transpose()

        except Exception as e:
            print(f"{id}: {e}")
            snts = pd.DataFrame({'missing': [None]})

        sentiments.append(snts)

    sent_df = pd.concat(sentiments, axis=1).transpose()
    sent_df.drop(columns=0, inplace=True)
    sent_df.reset_index(drop=True, inplace=True)

    return sent_df


if __name__ == "__main__":

    # Set parameters
    year_start = 1980
    top_n = 5  # Number of songs to keep for each Billboard
    aws_keys = pd.read_csv("accessKeys.csv")
    acces_key = aws_keys['Access key ID'].values[0]
    secret_key = aws_keys['Secret access key'].values[0]

    # Get data from Billboard
    df = get_billboard_hot100(year_start)
    df.drop_duplicates(inplace=True)

    # Filter and clean dataset
    df = df.loc[df['rank'] <= top_n, ['artist', 'title', 'year']]
    df.drop_duplicates(['artist', 'title'], keep='first', inplace=True)

    # Get the lyrics
    df['lyrics'] = [get_lyrics_genius(a, t) for a, t in zip(df.artist, df.title)]

    # Set up AWS comprehend client
    comprehend = boto3.client(service_name='comprehend',
                              region_name='eu-west-1',
                              aws_access_key_id=acces_key,
                              aws_secret_access_key=secret_key)

    # Get sentiments
    sent_df = get_sentiments(comprehend, df.lyrics)
    df = pd.concat([df.reset_index(drop=True), sent_df], axis=1, sort=False)
