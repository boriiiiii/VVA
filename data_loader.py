import requests
from bs4 import BeautifulSoup
from io import StringIO

import pandas as pd
import numpy as np


def load_raw_data():
    """Load all raw CSV datasets from the F1 Championship Archive."""
    return {
        'circuits':               pd.read_csv("F1 Championship Archive/circuits.csv"),
        'constructors':           pd.read_csv("F1 Championship Archive/constructors.csv"),
        'constructor_results':    pd.read_csv("F1 Championship Archive/constructor_results.csv"),
        'constructor_standings':  pd.read_csv("F1 Championship Archive/constructor_standings.csv"),
        'driver_standings':       pd.read_csv("F1 Championship Archive/driver_standings.csv"),
        'drivers':                pd.read_csv("F1 Championship Archive/drivers.csv"),
        'lap_times':              pd.read_csv("F1 Championship Archive/lap_times.csv"),
        'pit_stops':              pd.read_csv("F1 Championship Archive/pit_stops.csv"),
        'qualifying':             pd.read_csv("F1 Championship Archive/qualifying.csv"),
        'results':                pd.read_csv("F1 Championship Archive/results.csv"),
        'seasons':                pd.read_csv("F1 Championship Archive/seasons.csv"),
        'sprint_results':         pd.read_csv("F1 Championship Archive/sprint_results.csv"),
        'status':                 pd.read_csv("F1 Championship Archive/status.csv"),
        'races':                  pd.read_csv("F1 Championship Archive/races.csv"),
        'weather':                pd.read_csv("F1 Championship Archive/F1_Meteo_2022_2024.csv"),
    }


def length_turn(url, cId, len_turn_data):
    """Scrape circuit Length and Turns from a Wikipedia infobox and append to len_turn_data."""
    response = requests.get(url)

    # Parse the HTML content
    soup = BeautifulSoup(response.content, "html.parser")

    # Find the table element
    table = soup.find("table", class_="infobox")
    df = pd.read_html(StringIO(str(table)))[0]

    # If Length/Turns is not available set them to 0.
    try:
        length = df.loc[np.where(df == 'Length')[0][0]].iloc[1]
        turns = df.loc[np.where(df == 'Turns')[0][0]].iloc[1]
    except IndexError:
        length = '0.0000'
        turns = '0'

    # Append the list: [circuitId, Length, Turns]
    len_turn_data.append([cId, length, turns])


def scrape_circuit_info(df_circuits):
    """Scrape Length and Turns for every circuit and merge into df_circuits."""
    len_turn_data = []

    # Fetch Length and Turns for each circuit according to circuitId
    for cId, url in zip(df_circuits.circuitId, df_circuits.url):
        length_turn(url, cId, len_turn_data)

    # Convert Length and Turns columns to float and int types
    df_len_turn = pd.DataFrame(data=len_turn_data, columns=["circuitId", "Length", "Turns"])
    df_len_turn['Length'] = df_len_turn['Length'].str[:5].astype(float)
    df_len_turn['Turns'] = df_len_turn['Turns'].str[:2].astype(int)

    # Merge Length & Turns into df_circuits
    df_circuits = df_circuits.merge(df_len_turn, on='circuitId', how='left')
    return df_circuits
