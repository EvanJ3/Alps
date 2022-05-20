import dash
from dash import Dash, dcc, html, Input, Output, ClientsideFunction
from apps import Individual_Athlete_Dash, Country_Dash, Event_Dash, Methodology, Uploads
from apps.Dash_Utilities import Generate_Dash_Nav, Generate_Dash_Footer
import getopt, sys

###########################################################################################
# load external tableau and latex style sheets/js
# ########################################################################################### 
external_scripts = [
    {'src':'https://public.tableau.com/javascripts/api/viz_v1.js'},
    {'src':'https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-MML-AM_CHTML'}
]


###########################################################################################
# Create the application and server
# ########################################################################################### 
app = dash.Dash(__name__,suppress_callback_exceptions=True,
                        meta_tags=[{'name':'viewport',
                        'content':'width=device-width, initial-scale=1.0'}],external_scripts=external_scripts)

server = app.server

#generate the application nav bar
dash_nav = Generate_Dash_Nav()

#generate the application footer
footer = Generate_Dash_Footer()

###########################################################################################
# Create the application's content entry point representing all page contents
# ########################################################################################### 
application_content_container = html.Div(children=[],id='page-content')

###########################################################################################
# append content entrypoint, footer and nav to base application layout
# which will remain constant across all application pages
# ########################################################################################### 
app.layout = html.Div(children=[
    dash_nav,
    application_content_container,
    footer
])

###########################################################################################
# Begin callback section
# ########################################################################################### 

###########################################################################################
# Updates the footer style from natural bottom to fixed bottom for pages without
# full length page content; ensures the proper footer placement on all pages
# ########################################################################################### 
@app.callback(
    Output(component_id='dash_footer',component_property='className'),
    Input(component_id='url',component_property='pathname'))
def Update_Footer_Class(pathname):
    if pathname == '/apps/Uploads':
        class_out = 'footer-fixed-bottom'
    else:
        class_out = 'footer-bottom'
    return class_out

###########################################################################################
# Callback updates the dash page layout based upon user selection of dashboard layout
# ########################################################################################### 
@app.callback(Output(component_id='page-content',component_property='children'),
                [Input(component_id='url',component_property='pathname')])

def display_page(pathname):
    if pathname == '/apps/Country_Dash':
        return Country_Dash.layout
    
    elif pathname == '/apps/Event_Dash':
        return Event_Dash.layout

    elif pathname == '/apps/Individual_Athlete_Dash':
        return Individual_Athlete_Dash.layout

    elif pathname == '/apps/Methodology':
        return Methodology.layout

    elif pathname == '/apps/Uploads':
        return Uploads.layout
    
    else:
        return Individual_Athlete_Dash.layout
    


###########################################################################################
# Running instructions specific to regular local environement running from terminal
# and running within a docker container under the -docker optional run argument
# ########################################################################################### 

if __name__ == '__main__':
    argumentList = sys.argv[1:]
    if argumentList == []:
        app.run_server(debug=True)
    elif (argumentList[0] in ['-docker','-d']):
        app.run_server(debug=True,port=8000,host='0.0.0.0')
    else:
        app.run_server(debug=True)




