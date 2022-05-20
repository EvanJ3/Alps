import pandas as pd
from dash import Dash, dcc, html
import numpy as np
import pathlib
from apps.Dash_Utilities import *

##############################################################
# Dash requires pathlib standardized paths for reading
# instructions the primary used directories are declared here
# ############################################################ 
PATH = pathlib.Path(__file__).parent
DATA_PATH = PATH.joinpath("../Data").resolve()


##############################################################
# Generates all of a given methodology section's cards
# ############################################################ 
def Generate_Methodology_Card_Section(section_identifier='INDV'):
    """Generates a methodology card section given a filtering
        condition corresponding to the section name

    Args:
        section_identifier (str): The section filter string used to 
        seperate cards from a given section in the methodology datafile

    Returns:
        dash object: a div container with all methodolgy cards generated
    """
    methodology_df = pd.read_csv(DATA_PATH.joinpath('Methodology Data/Methodology_Components.csv'))
    methodology_df = methodology_df[methodology_df['Dashboard'] == section_identifier]
    cards = []
    for i in range(0,methodology_df.shape[0]):
        title_i = methodology_df['Component_Name'].iloc()[i]
        text_i = methodology_df['Component_Description'].iloc()[i]
        image_path_i = methodology_df['Component_Image_Path'].iloc()[i]
        card_i = Generate_HTML_Card(image_path=image_path_i,title=title_i,text=text_i)
        cards.append(card_i)
    card_container = html.Div(children=cards,className='card-container')
    card_section = html.Div([card_container],className='card-section')
    return card_section

##############################################################
# Generates a pdf downloadable card
# ############################################################ 
def Generate_PDF_Card(title,text,pdf_src,pdf_preview_src):
    """takes as input card generation sections and outputs
        a pdf card with download option

    Args:
        title (str): card title
        text (str): card body text
        pdf_src(str): file path for download
        pdf_preview_src(str): preview image path

    Returns:
        dash object: returns a downloadable dash card for displaying pdfs
    """
    pdf_download_button = Generate_HTML_Card_Button(button_text='Download',button_href=pdf_src)
    pdf_card = Generate_HTML_Button_Card(image_path=pdf_preview_src,title=title,text=text,button_elements=pdf_download_button)
    return pdf_card
