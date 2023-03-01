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
warnings.filterwarnings('ignore')

base_url = "https://www.ted.com/talks"
topics = ["AI"]

def get_page_urls(page_num, topics=topics, base_url=base_url):
    """
    Get all talk URLs from a given page_num
    """

    try:
        response = requests.get(
            base_url,
            params={'language':'en', 'page':page_num, 'topics[]':topics },
            headers = {'User-agent': 'Scraping TED'},
            timeout=(5,5)  # 5 secs to establish connection and 5 secs to get data
        )
    except Timeout:
        print("The request has timed out. Retry.")

    if response.status_code != 200:
        return -1    # unsucccesful request

    # parse HTML
    soup = BeautifulSoup(response.content, 'html.parser')
    talk_links = soup.find_all('div', class_ = 'media__image')
    urls = [talk.a['href'] for talk in talk_links]

    return urls

def create_url( topics, page_num, base=base_url):
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
    for page in range(start_page, (end_page+1)):
        # send in a range of pages; will help if connection breaks in between
        urls = get_page_urls(page)
        if urls == -1:
            # graceful degradation
            print('An unsuccesful request encountered')
            print(f'So far {len(talk_urls)} talk URLs have been collected')
            print(f'Last page collected was {page-1}')
            break
        talk_urls = talk_urls + urls
        time.sleep(2)    # add a delay


    return talk_urls

urls = get_urls(base_url, 1, 1)
urls = ["https://www.ted.com"+url for url in urls]
print(urls[:10])

# saving the urls
url_file = open('Talk_URLs.txt', 'w')
url_file.write('\n'.join(urls))
url_file.close()


def extract_data_dict(pg_url):
    """
    Get necessary data from a given URL
    """

    response = requests.get(
        pg_url,
        headers = {'User-agent': 'Talk Scraper Bot'},
        timeout = (5,5)
    )

    if response.status_code != 200:
        return -1    # unsucccesful request

    # source: https://github.com/The-Gupta/TED-Scraper/blob/master/Scraper.ipynb

    # parse HTML
    html_text = response.text

    # indexing to get data
    start_index = html_text.find('<script data-spec="q">q("talkPage.init",')
    end_index = html_text[start_index:].find(')</script>')
    script_tag   =  html_text[start_index: start_index + end_index]
    json_obj  =  script_tag[len('<script data-spec="q">q("talkPage.init",'):]

    # convert to dictionary
    try:
        data_dic = json.loads(json_obj)
        data_dic = data_dic['__INITIAL_DATA__']
    except:
        data_dic = {}

    return data_dic

def get_talk_attr(data_dic):
    """
    Get talk attributes
    """

    if data_dic == {}:
        return None
    
    talk_name = data_dic["name"].split(':')[1].strip()

    try:
        descr = data_dic["description"]
        eve = data_dic["event"]
        views = data_dic["viewed_count"]
        dur = data_dic['talks'][0]['duration']
        tags = data_dic['talks'][0]['tags']
        rec_date = data_dic['talks'][0]['recorded_at']  # string format
        pub_date = data_dic['talks'][0]['player_talks'][0]['published']   # weird format, convert later
    except:
        descr = np.nan
        eve = np.nan
        views = np.nan
        dur = np.nan
        tags = np.nan
        rec_date = np.nan
        pub_date = np.nan

    return descr, eve, talk_name, views, dur, tags, rec_date, pub_date

    
def get_speaker_attr(data_dic):
    """
    Get speaker attributes
    """

    if data_dic == {}:
        return np.nan, np.nan, np.nan, np.nan, np.nan
    
    talk_name = data_dic["name"].split(':')[1].strip()
    try:
        sp_name = data_dic['speakers'][0]['firstname'] + " " + data_dic['speakers'][0]['lastname']
        sp_title = data_dic['speakers'][0]['title']
        sp_descr = data_dic['speakers'][0]['description']
        sp_bio = data_dic['speakers'][0]['whotheyare']
    except:
        sp_name = np.nan
        sp_title = np.nan
        sp_descr = np.nan
        sp_bio = np.nan

    return talk_name, sp_name, sp_title, sp_descr, sp_bio

def make_talk_dataframe(url_list):
    """
    talk dataframe
    """

    # init
    descr = []
    eve = []
    talk_name = []
    views = []
    dur = []
    tags = []
    rec_date = []
    pub_date = []

    ctr = 1

    for url in url_list:
        print(f'Scraping URL {ctr}')

        dd = extract_data_dict(url)

        try:
            d, e, tn, v, du, t, rec, pub = get_talk_attr(dd)
        except:
            continue

        descr.append(d)
        eve.append(e)
        talk_name.append(tn)
        views.append(v)
        dur.append(du)
        tags.append(t)
        rec_date.append(rec)
        pub_date.append(pub)

        # update counter
        ctr += 1

    df = pd.DataFrame(
        {
            'talk_desc': descr,
            'event': eve,
            'talk_name': talk_name,
            'views': views,
            'duration': dur,
            'tags': tags,
            'recorded_at': rec_date,
            'published on': pub_date
        }
    )

    return df


def make_speaker_dataframe(url_list):
    """
    Make dataframe of data from urls
    """

    # init

    #descr = eve = talk_name = views = dur = tags = rec_date = pub_date = []
    talk_name = []
    sp_name = []
    sp_title = []
    sp_descr = []
    sp_bio = []

    ctr = 1

    for url in url_list:
        print(f'Scraping URL {ctr}')

        dd = extract_data_dict(url)

        t, sn, st, sd, sb = get_speaker_attr(dd)

        talk_name.append(t)
        sp_name.append(sn)
        sp_title.append(st)
        sp_descr.append(sd)
        sp_bio.append(sb)

        # update counter
        ctr += 1

        # add delay of 2 secs between requests
        # time.sleep(2)

    df = pd.DataFrame(
        {
            'talk': talk_name,
            'speaker': sp_name,
            'speaker_title': sp_title,
            'speaker_occ': sp_descr,
            'speaker_bio': sp_bio
        }
    )

    return df



# get urls from file
urls = open("Talk_URLs.txt").read().splitlines()[:5]

print(urls)
# make the function calls     --- for speaker data
df = make_speaker_dataframe(urls)
#df = make_talk_dataframe(urls)



print(df.info())


# store data as .csv
df.to_csv("talk_data.csv", index=False)


def cleaned_transcript(raw):
    """
    Extract transcript text from fetched json data.
    """

    transcript = ""

    js = json.loads(raw)
    paragraphs_obj = js['props']['pageProps']['transcriptData']['translation']['paragraphs']
    

    for paragraph in paragraphs_obj:
        cues = paragraph['cues']
        for line in cues:
            transcript += line['text'] + " "

    return transcript

def extract_transcript(pg_url):
    """
    Extract transcript
    """

    response = requests.get(
        pg_url,
        headers = {'User-agent': 'Extracting Transcripts Bot'},
        timeout=(5,5)  # 5 secs to establish connection and 5 secs to get data
    )

    # parse HTML
    soup = BeautifulSoup(response.content, 'html.parser')

    try:
        #text = soup.find(id='__NEXT_DATA__').contents[0]
        text = soup.find(type="application/ld+json").contents[0]
        transcript = json.loads(text)['transcript']
        transcript = html.unescape(transcript)
        #transcript =  js
    except:
     transcript = np.nan

    title=str(soup.find('title'))
    try:
        title = title.split(":")[1]
        title = title.split(" | ")[0].strip()
    except:
        pass

    return title, transcript

def make_transcript_dataframe(url_list):
    """
    Make transcript dataframe
    """

    titles = []
    transcripts = []

    ctr = 1

    for url in url_list:

        # get the transcript page
        url = url.replace("?language=en", "") + "/transcript"


        print(f'Scraping URL {ctr}')

        title, transcript = extract_transcript(url)
        print(transcript)

        titles.append(title)
        transcripts.append(transcript)

        # add a time delay
        time.sleep(random.randint(100,1000)/1000)

        ctr += 1

    df = pd.DataFrame(
        {
            'title': titles,
            'transcript': transcripts
        }
    )

    return df


df = make_transcript_dataframe(urls)


# store data as .csv
df.to_csv('transcript_data.csv', index=False, encoding='utf-8-sig')



