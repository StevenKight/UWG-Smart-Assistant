"""
Gets current events from the West Georgia Events Calendar.

Pylint: 10.00 (August 25, 2022)
"""

import calendar
import os
from urllib.request import urlopen
import json

from bs4 import BeautifulSoup

__author__ = "Steven Kight"
__version__ = "1.5"
__pylint__ = "2.14.4"

def write_json_info(new_data: dict, filename='smart_assistant/conversation/uwg/Events.json') -> None:
    """
    Write new information to the Events Json File.

    ### Parameters
        `new_data` (`dict`): The new information to be added to the json file.
        `filename` (`str`): The json file to add the new information to.
    """

    with open(filename,'r+', encoding="utf8") as file:
        file_data = json.load(file)
        file_data["Events"].append(new_data)
        file.seek(0)
        json.dump(file_data, file, indent = 4)

def get_html() -> BeautifulSoup:
    """
    Uses BeautifulSoup to get html data from the url.

    ### Return
        The html of the webpage as provided by `BeautifulSoup`.
    """

    url = "https://www.westga.edu/calendar/"

    with urlopen(url) as page:
        html_bytes = page.read()

    html = html_bytes.decode("utf-8")

    soups = BeautifulSoup(html, features = "html.parser")

    return soups

def dates_times(html: BeautifulSoup) -> list:
    """
    Get and format the dates.

    ### Parameters
        `html` (`BeautifulSoup`): The html of the calendar webpage.
    ### Return
        A list of dates, a list of start times, and a list of end times.
    """

    event_dates = str(html.find_all("div", {"class": "listing-date"}))

    dates_times_list = event_dates.split('</div>, <div class="listing-date">')

    dates_times_list[0] = dates_times_list[0].replace('[<div class="listing-date">', '')
    end_index_dt = len(dates_times_list)-1
    dates_times_list[end_index_dt] = dates_times_list[end_index_dt].replace('</div>]', '')

    dates = []
    start_times = []
    end_times = []

    for date_time in dates_times_list:
        date_time = date_time.split(' / ')
        date_month = calendar.month_name [list(calendar.month_abbr).index(date_time[0][0:3])]
        dates.append(date_time[0].replace(date_time[0][0:3], date_month))

        time = date_time[1].replace(date_time[1][0:5], '')
        if '-' in time:
            split_time = time.split (' - ')
            start_times.append(split_time[0])
            end_times.append(split_time[1])
        else:
            start_times.append(time)
            end_times.append(time)

    return dates, start_times, end_times

def locations(html) -> list:
    """
    Get and format the locations.

    ### Parameters
        `html` (`BeautifulSoup`): The html of the calendar webpage.
    ### Return
        A list of locations of the current events.
    """

    event_locations = str(html.find_all("span", {"class": "category-box"}))

    location_split_point = '</span></strong></span>, <span class="category-box"><strong><span>'
    locations_list = event_locations.split(location_split_point)

    locations_list[0] = locations_list[0].replace('[<span class="category-box"><strong><span>', '')
    end_index = len(locations_list)-1
    locations_list[end_index] = locations_list[end_index].replace('</span></strong></span>]', '')

    return locations_list

def titles(html) -> list:
    """
    Get and format titles

    ### Parameters
        `html` (`BeautifulSoup`): The html of the calendar webpage.
    ### Return
        A list of the event names.
    """

    event_titles = str(html.find_all("div", {"class": "listing-content"}))
    titles_list = event_titles.split('</div>, <div class="listing-content">')
    for index, title in range(len(titles_list)):
        titles_list[index] = title.split("\n")[2][24:].replace('</a></h3>', '')

    return titles_list

def create_events() -> None:
    """
    Create the new events and store to json file
    """

    json_path = 'smart_assistant/conversation/uwg/Events.json'

    if os.path.exists(json_path):
        os.remove(json_path)

    new_dictionary = {"Events": []}
    json_string = json.dumps(new_dictionary, indent=4)
    with open("smart_assistant/conversation/uwg/Events.json", "w", encoding="utf8") as inform:
        inform.write(json_string)
        inform.close()

    soup = get_html()

    date_list, start_times_list, end_times_list = dates_times(soup)
    locations_list_returned = locations(soup)
    title_list_returned = titles(soup)

    date_location_len = len(date_list) == len(locations_list_returned)
    date_titles_len = len(date_list) == len(title_list_returned)

    if date_location_len and date_titles_len:
        for place, date in enumerate(date_list):
            new_info = {"Name": title_list_returned[place],
                        "Location": locations_list_returned[place],
                        "Date": date,
                        "Start Time": start_times_list[place],
                        "End Time": end_times_list[place]
                    }

            write_json_info(new_info)
