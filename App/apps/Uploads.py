from dash import Dash, dcc, html, Input, Output, callback, dash_table, State
import numpy as np
from apps.Dash_Utilities import *
from apps.Uploads_Dash_Utilities import *
import pathlib
from plotly.validator_cache import ValidatorCache

##############################################################
# Dash requires pathlib standardized paths for reading
# instructions the primary used directories are declared here
# ############################################################

PATH = pathlib.Path(__file__).parent
DATA_PATH = PATH.joinpath("../Data").resolve()
EXAMPLE_PATH = PATH.joinpath(("../Data/Data Resources"))
UPLOAD_DATA_PATH = PATH.joinpath("../Data/User Uploaded Data").resolve()

######################################################################
# Here we add some text used as the subtitle for the upload section
# ###################################################################
upload_title_text = '''
Please use the dropdown box below to submit new data to add to our existing
data offerings. Please ensure that your uploaded file conforms with the column
structure shown in the example file below. After submitting your upload, you should see
a preview of your uploaded data appear as a table below the submission box. A cleaning
and reconsiliation script will then run to ensure the new data matches the applications
stored format of the existing data. This will take awhile to run so please be patient.
After it is finished, the loading spinner should stop and alert you of the success or failure of the process.
For changes to the dataset to take effect a restart of the application is neccecary.
'''
######################################################################
# Create upload title shelf
# ###################################################################
Upload_title = TitleShelf(title='Upload New Data',title_sub_text=upload_title_text)

######################################################################
# Here we create a download link and text for the example data file
# ###################################################################
#example_link_element = html.Div([html.A(['Example Data Format'],target='_blank',download="USOPC_Example_Data_Upload_Format.csv",href='App/Data/Data Resources/USOPC_Example_Data_Upload_Format.csv',style={"color":"blue"})],style={"width":"100%","display":"flex","flex-direction":"column","justify-content":"center","align-items":"center","color":"blue","text-decoration":"underline"})
example_link_element = html.Div([html.Button("Download Example Format", id="upload-example-download-button",className='card-button'), dcc.Download(id="upload-example-download")],className='dash-container shelf jc-center ai-center ac-center w100')
######################################################################
# Here create the actual upload element
# ###################################################################
Upload_element = dcc.Loading(UploadBox(upload_id='upload_data',multiple=True))

######################################################################
# Here we append all elements to the webpage body
# ###################################################################
layout = html.Div(children=[
    Upload_title,
    example_link_element,
    Upload_element
])

######################################################################
# Begin Application callback section
# ################################################################### 

@callback(
    Output("upload-example-download", "data"),
    Input("upload-example-download-button", "n_clicks"),
    prevent_initial_call=True,
)
def func(n_clicks):
    return dcc.send_file(
        "App/Data/Data Resources/USOPC_Example_Data_Upload_Format.csv"
    )

######################################################################
# callback runs the cleaning and concat of new data script
# returns a message for successful or unsuccessful outcome
# ################################################################### 

@callback(
    Output('output_upload_data', 'children'),
    Input('upload_data', 'contents'),
    State('upload_data', 'filename'),
    State('upload_data', 'last_modified'))
def update_output(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        children = [
            parse_contents(c, n, d) for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates)]
        return children



