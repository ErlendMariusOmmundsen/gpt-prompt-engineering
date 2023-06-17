# Adapted from https://deepnote.com/@ramshankar-yadhunath/Scraping-TED-7d1a82e1-e6e1-4d16-8dae-d70e4ab21731

import requests
from bs4 import BeautifulSoup
from selenium import webdriver
import time
import json
import pandas as pd
import numpy as np
import random
import html

import warnings

warnings.filterwarnings("ignore")

base_url = "https://www.ted.com/talks"
topic_tags = [
    "Climate Change",
    "Technology",
    "Design",
    "Future",
]


def get_page_urls(page_num, topics=topic_tags, base_url=base_url):
    """
    Get all talk URLs from a given page_num
    """

    try:
        response = requests.get(
            base_url,
            params={"language": "en", "page": page_num, "topics[]": topic_tags},
            headers={"User-agent": "Scraping TED"},
            timeout=(5, 5),  # 5 secs to establish connection and 5 secs to get data
        )
    except Timeout:
        print("The request has timed out. Retry.")

    if response.status_code != 200:
        return -1  # unsucccesful request

    # parse HTML
    soup = BeautifulSoup(response.content, "html.parser")
    talk_links = soup.find_all("div", class_="media__image")
    urls = [talk.a["href"] for talk in talk_links]

    return urls


def create_url(topics, page_num, base=base_url):
    page_str = "?page=" + str(page_num)
    topic_str = ""
    for topic in topics:
        topic_str += "&topics[]=" + topic
    url = base + page_str + topic_str
    return url


def get_urls(base_url, start_page, end_page):
    """
    Get all talk URLs
    """

    talk_urls = []
    for page in range(start_page, (end_page + 1)):
        # send in a range of pages; will help if connection breaks in between
        urls = get_page_urls(page)
        if urls == -1:
            # graceful degradation
            print("An unsuccesful request encountered")
            print(f"So far {len(talk_urls)} talk URLs have been collected")
            print(f"Last page collected was {page-1}")
            break
        talk_urls = talk_urls + urls
        time.sleep(2)  # add a delay

    return talk_urls


def cleaned_transcript(raw):
    """
    Extract transcript text from fetched json data.
    """

    transcript = ""

    js = json.loads(raw)
    paragraphs_obj = js["props"]["pageProps"]["transcriptData"]["translation"][
        "paragraphs"
    ]

    for paragraph in paragraphs_obj:
        cues = paragraph["cues"]
        for line in cues:
            transcript += line["text"] + " "

    return transcript


def extract_transcript(pg_url):
    """
    Extract transcript
    """

    response = requests.get(
        pg_url,
        headers={"User-agent": "Extracting Transcripts Bot"},
        timeout=(5, 5),  # 5 secs to establish connection and 5 secs to get data
    )

    # parse HTML
    soup = BeautifulSoup(response.content, "html.parser")

    try:
        # text = soup.find(id='__NEXT_DATA__').contents[0]
        text = soup.find(type="application/ld+json").contents[0]

        topics_list = soup.find("aside").find("ul").getText(",")
        # print(text)
        transcript = json.loads(text)["transcript"]
        transcript = html.unescape(transcript)

        description = json.loads(text)["description"]
        description = html.unescape(description)

        title = json.loads(text)["name"]
        title = html.unescape(title)

        duration = json.loads(text)["duration"]
        duration = html.unescape(duration)

        upload_date = json.loads(text)["uploadDate"]
        upload_date = html.unescape(upload_date)

    except:
        title = np.nan
        description = np.nan
        transcript = np.nan
        duration = np.nan
        upload_date = np.nan
        topics_list = np.nan

    title_str = str(soup.find("title"))
    try:
        speaker = title_str.partition(":")[0][7:]
    except:
        speaker = np.nan

    return (title, topics_list, speaker, description, transcript, duration, upload_date)


def make_transcript_dataframe(url_list):
    """
    Make transcript dataframe
    """

    titles = []
    topics = []
    speakers = []
    descriptions = []
    transcripts = []
    durations = []
    upload_dates = []

    ctr = 1

    for url in url_list:
        # get the transcript page
        url = url.replace("?language=en", "") + "/transcript"

        print(f"Scraping URL {ctr}")

        (
            title,
            topics_list,
            speaker,
            description,
            transcript,
            duration,
            upload_date,
        ) = extract_transcript(url)

        titles.append(title)
        topics.append(topics_list)
        speakers.append(speaker)
        descriptions.append(description)
        transcripts.append(transcript)
        durations.append(duration)
        upload_dates.append(upload_date)

        # add a time delay
        time.sleep(random.randint(100, 1000) / 1000)

        ctr += 1

    df = pd.DataFrame(
        {
            "title": titles,
            "topics": topics,
            "speaker": speakers,
            "description": descriptions,
            "transcript": transcripts,
            "duration": durations,
            "upload_date": upload_dates,
        }
    )

    return df


urls = get_urls(base_url, 1, 1)
urls = ["https://www.ted.com" + url for url in urls]

# saving the urls
url_file = open("Talk_URLs.txt", "w")
url_file.write("\n".join(urls))
url_file.close()


# get urls from file
urls = open("Talk_URLs.txt").read().splitlines()[:10]

df = make_transcript_dataframe(urls)

# store data as .csv
df.to_csv(
    "data/transcript_data.csv",
    index=False,
    encoding="utf-8-sig",
    mode="a",
    header=False,
)
