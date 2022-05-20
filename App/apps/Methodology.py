import pandas as pd
import plotly.express as px 
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, callback
import numpy as np
from apps.Dash_Utilities import *
from apps.Methodology_Dash_Utilities import *
import pathlib

##############################################################
# Dash requires pathlib standardized paths for reading
# instructions the primary used directories are declared here
# ############################################################
PATH = pathlib.Path(__file__).parent
DATA_PATH = PATH.joinpath("../Data").resolve()
DELIVERABLE_PATH = PATH.joinpath("../Deliverables").resolve()
REPORT_PATH = DELIVERABLE_PATH.joinpath("Report/United_States_Olympic_Committee_Practicum_Final.pdf")

######################################################################
# Here we read in the neccecary datafiles
# ###################################################################
method_df = pd.read_csv(DATA_PATH.joinpath('Methodology Data/Methodology_Components.csv'))


##############################################################
# here we create and store subtext for our section titles
# ############################################################
methodology_overview_title_subtext = '''
In this dashboard we present a deeper look into the underlying methodology and approaches utilized in the creation of this application.
Included in the sections below are summary of each of the components used within the application. These summaries are inlucded to provide
users with a better idea of how and what each figure represents. Additionally, we inlcude links to our final report which presents all 
of our findings related to this project and application.

'''

deliverables_title_subtext = '''
The following documents section is intended to provide users with downloads and
access to each of the document and presentation materials neccecary to understand
our application and analysis.
'''


indv_dash_title_subtext = '''
The Individual athlete dashboard is intend to provide users
with all of the information they could possibly need in evaluating
a given Alpine skiing athlete. The dashboard itself consists
of several interactive visualizations each depicting an
important dimension of better understanding athletic performance.

'''

event_dash_title_subtext = '''
The event dashboard represents the primary entry point
for additional modeling, analysis, and visualization of our
event level modeling methodology. This dashboard allows
far more in depth modeling and cross-validation of all KDE
models discussed throughout our report, and much more.

'''


country_dash_title_subtext = '''
The final and probably most important dashboard within
our web application is the country dashboard which aims
to provide users with a complete view of a given country’s
position in the race to Olympic gold. From pipeline analysis to 
prospect analysis this dashboard has everything a end user could 
think to analyze at the country and olympic perspectives of alpine skiing.
'''

##############################################################
# here we create our card text for the report card which
# is generated manually unlike the other component cards 
# due to the presence of a download button
# ############################################################
report_card_text = '''
Comprehensive report on all of our findings throughout the process.
The report covers a significant amount of detail not captured within the 
methodology page so we reccomend consulting it for any detailed clarifications or queries
'''

##############################################################
# here we create our card text for the presentation card which
# like the report card is generated manually 
# ############################################################
presentation_card_text = '''
The final presentation serves as a high level synopsis of the report findings.
It briefly covers key aspects of the analytical and modeling approach as well as
some of the additional tools developed for the project.
'''

######################################################################
# Here we create the full length title shelves for our 
# primary sections 
# ###################################################################

methodology_overview_title = TitleShelf(title='Methodology Overview',title_sub_text=methodology_overview_title_subtext)
deliverables_title = TitleShelf(title='Final Report & Presentation',title_sub_text=deliverables_title_subtext)
indv_dash_title = TitleShelf(title='Individual Dash Components',title_sub_text=indv_dash_title_subtext)
event_dash_title = TitleShelf(title='Event Dash Components',title_sub_text=event_dash_title_subtext)
country_dash_title = TitleShelf(title='Country Dash Components',title_sub_text=country_dash_title_subtext)

##############################################################
# here we generate the report and presentation pdf cards
# ############################################################
final_report_card = Generate_PDF_Card(title='Final Report',text=report_card_text,pdf_src='../Deliverables/Report/United_States_Olympic_Committee_Practicum_Final.pdf',pdf_preview_src='../assets/Methodology Dash Images/Final_Report_PDF_Preview_Image.png')
final_presentation_card = Generate_PDF_Card(title='Final Presentation',text=presentation_card_text,pdf_src='../Deliverables/Presentations/Evan_Jones_USOPC_Final_Presentation.pptx',pdf_preview_src='../assets/Methodology Dash Images/Final_Presentation_Preview.png')
deliverable_cards_container = html.Div([final_report_card,final_presentation_card],className='card-container')
report_components_section = html.Section([deliverable_cards_container],className='card-section')


##############################################################
# here we generate the other section cards and containers
# ############################################################
indv_dash_components_section = Generate_Methodology_Card_Section(section_identifier='INDV')
event_dash_components_section = Generate_Methodology_Card_Section(section_identifier='EVENT')
country_dash_components_section = Generate_Methodology_Card_Section(section_identifier='COUNTRY')


######################################################################
# Here we take each of our section shelf elements and append them
# and post them to the application's layout in order of appearance
# ################################################################### 
layout = html.Div(children=[
    methodology_overview_title,
    indv_dash_title,
    indv_dash_components_section,
    event_dash_title,
    event_dash_components_section,
    country_dash_title,
    country_dash_components_section,
    deliverables_title,
    report_components_section,
])