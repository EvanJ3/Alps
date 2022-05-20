from dash import Dash, dcc, html
import pandas as pd
import numpy as np

def Generate_Dash_Nav():
    """Generates the applications dash html nav bar representation

    Returns:
        dash object: application navbar
    """
    company_logo_element = dcc.Link(html.Img(src='../assets/images/USOPC_Nav_Logo.webp',className='nav-logo'),href='/apps/Methodology')
    indv_dash_element = dcc.Link('Individual Dashboard',href='/apps/Individual_Athlete_Dash')
    event_dash_element = dcc.Link('Event Dashboard',href='/apps/Event_Dash')
    country_dash_element = dcc.Link('Country Dashboard',href='/apps/Country_Dash')
    methodology_dash_element = dcc.Link('Methodology',href='/apps/Methodology')
    uploads_dash_element = dcc.Link('Uploads',href='/apps/Uploads')
    dash_post_listener = dcc.Location(id='url',refresh=False,pathname='')
    nav_container = html.Nav(children=[company_logo_element,
                                    indv_dash_element,
                                    event_dash_element,
                                    country_dash_element,
                                    methodology_dash_element,
                                    uploads_dash_element,
                                    dash_post_listener
                                    ],className="container nav-flex")
    dash_nav = html.Header(children=[nav_container],className="header")
    return dash_nav

def Generate_Dash_Footer():
    """Generates the applications dash html footerrepresentation

    Returns:
        dash object: application footer
    """

    github_svg = html.Img(src='../assets/icons/github.svg',className='footer-svg')
    footer_github_link = html.A([github_svg],href='https://gitfront.io/r/EvanJ03/3a883b3855accaf614b2cdd9e000619aadf9a36e/USOPC-Alpine-Skiing/',className='footer-link-black')

    linkedin_svg = html.Img(src='../assets/icons/linkedin.svg',className='footer-svg')
    footer_linkedin_link = html.A([linkedin_svg],href="https://linkedin.com/in/evan-jones-b57a7512b",className='footer-link-black')

    inmail_svg = html.Img(src='../assets/icons/inmail.svg',className='footer-svg')
    footer_mail_link = html.A([inmail_svg],href="mailto:ej@evanjones.ai",className='footer-link-black')

    usopc_svg = html.Img(src='../assets/images/USOC_Logo.png',className='footer-svg')
    footer_usopc_link = html.A([usopc_svg],href='https://teamusa.org',className='footer-link-black')


    footer_icon_container = html.Div([footer_github_link,footer_linkedin_link,footer_mail_link,footer_usopc_link],className='footer-container',style={"color":"black"})
    footer_text = html.P('Evan Jones © 2022 | Atlanta, GA | Georgia Institute of Technology | United States Olympic & Paralympic Committee',className='footer-text-black')
    footer_text_container = html.Div([footer_text],className='footer-container')

    footer = html.Footer(children=[footer_icon_container,footer_text_container],className='footer-bottom',id='dash_footer')
    return footer

def LoadingGraph(graph_id,className):
    """Gets and prints the spreadsheet's header columns

    Args:
        file_loc (str): The file location of the spreadsheet
        print_cols (bool): A flag used to print the columns to the console
            (default is False)

    Returns:
        list: a list of strings representing the header columns
    """
    figure_style = {"height":"100%","width":"100%"}
    figure = dcc.Graph(id=graph_id,figure={},style=figure_style)
    loading_id = graph_id +'_loading'
    loading_element = dcc.Loading(id=loading_id,children=[figure],parent_className=className,parent_style=figure_style)
    chart_container = html.Div([loading_element],className=className)
    return chart_container

def TitleShelf(title,title_sub_text):
    """Gets and prints the spreadsheet's header columns

    Args:
        file_loc (str): The file location of the spreadsheet
        print_cols (bool): A flag used to print the columns to the console
            (default is False)

    Returns:
        list: a list of strings representing the header columns
    """
    title_head = html.H2(title,className='title-shelf-heading')
    title_text = html.P(title_sub_text,className='title-shelf-text')
    title_text_container = html.Div([title_text],className='title-shelf-text-container')
    title_shelf = html.Div([title_head,title_text_container],className='title-shelf')
    return title_shelf

def UploadBox(upload_id,multiple=True):
    """Gets and prints the spreadsheet's header columns

    Args:
        file_loc (str): The file location of the spreadsheet
        print_cols (bool): A flag used to print the columns to the console
            (default is False)

    Returns:
        list: a list of strings representing the header columns
    """
    upload_text = html.A(children=['Drag and Drop or Select Files'],className='upload-text')
    upload_text_container = html.Div(children=[upload_text],className='upload-text-container')
    upload_element = dcc.Upload(id=upload_id,children=[upload_text_container],multiple=multiple)
    output_id = 'output_'+ upload_id
    upload_output_element = html.Div(id=output_id,className='upload-output')
    upload_container = html.Div(children=[upload_element],className='upload-container')
    upload_shelf = html.Div(children=[upload_container,upload_output_element],className='upload-shelf')
    return upload_shelf

def DashGraph(graph_id,className):
    """Gets and prints the spreadsheet's header columns

    Args:
        file_loc (str): The file location of the spreadsheet
        print_cols (bool): A flag used to print the columns to the console
            (default is False)

    Returns:
        list: a list of strings representing the header columns
    """
    figure_style = {"height":"100%","width":"100%"}
    figure = dcc.Graph(id=graph_id,figure={},style=figure_style)
    chart_container = html.Div([figure],className=className)
    return chart_container

def array_to_table_row(array,classNameRoot='dash-df',is_header=False):
    """Gets and prints the spreadsheet's header columns

    Args:
        file_loc (str): The file location of the spreadsheet
        print_cols (bool): A flag used to print the columns to the console
            (default is False)

    Returns:
        list: a list of strings representing the header columns
    """
    table_row_class_name = classNameRoot + '-row'
    table_data_class_name = classNameRoot + '-data'
    table_header_class_name = classNameRoot + '-header'
    table_header_row_class_name = classNameRoot + '-header-row'
    table_data_row_class_name = classNameRoot + '-data-row'
    if is_header:
        table_row = html.Tr(children=[html.Th(children=x,className=table_header_class_name) for x in array],className=table_header_row_class_name)
    else:
        table_row = html.Tr(children=[html.Td(children=x,className=table_data_class_name) for x in array],className=table_data_row_class_name)
    return table_row

def DataFrame_To_HTML_Table(df,classNameRoot='dash-df'):
    """Gets and prints the spreadsheet's header columns

    Args:
        file_loc (str): The file location of the spreadsheet
        print_cols (bool): A flag used to print the columns to the console
            (default is False)

    Returns:
        list: a list of strings representing the header columns
    """
    column_names = df.columns
    table_class_name = classNameRoot + '-table'
    table_row_class_name = classNameRoot + '-row'
    table_data_class_name = classNameRoot + '-data'
    table_header_class_name = classNameRoot + '-header'
    table_header_row_class_name = classNameRoot + '-header-row'
    table_data_row_class_name = classNameRoot + '-data-row'
    table_header_row = array_to_table_row(column_names,classNameRoot=classNameRoot,is_header=True)
    table_rows = [table_header_row]
    for i in range(0,df.shape[0]):
        data_row_i = list(df.iloc()[i])
        table_row_i = array_to_table_row(data_row_i,classNameRoot=classNameRoot,is_header=False)
        table_rows.append(table_row_i)
    table_element = html.Table(children=table_rows,className=table_class_name)
    return table_element

def Array_To_HTML_Table(array,classNameRoot='dash-df'):
    """Gets and prints the spreadsheet's header columns

    Args:
        file_loc (str): The file location of the spreadsheet
        print_cols (bool): A flag used to print the columns to the console
            (default is False)

    Returns:
        list: a list of strings representing the header columns
    """
    table_class_name = classNameRoot + '-table'
    table_row_class_name = classNameRoot + '-row'
    table_data_class_name = classNameRoot + '-data'
    table_header_class_name = classNameRoot + '-header'
    table_header_row_class_name = classNameRoot + '-header-row'
    table_data_row_class_name = classNameRoot + '-data-row'
    table_rows = []
    if type(array) == list:
        array_length = len(array)
    else:
        array_length = array.shape[0]
    
    for i in range(0,array_length):
        data_row_i = array[i]
        table_row_i = array_to_table_row(data_row_i,classNameRoot=classNameRoot,is_header=False)
        table_rows.append(table_row_i)
    table_element = html.Table(children=table_rows,className=table_class_name)
    return table_element


def Multi_Index_DataFrame_To_HTML_Table(df,index_cols,classNameRoot='dash-df',header=None):
    """Gets and prints the spreadsheet's header columns

    Args:
        file_loc (str): The file location of the spreadsheet
        print_cols (bool): A flag used to print the columns to the console
            (default is False)

    Returns:
        list: a list of strings representing the header columns
    """
    df_copy = df.copy()
    col_names = list(df_copy.columns)
    table_class_name = classNameRoot + '-table'
    table_row_class_name = classNameRoot + '-row'
    table_data_class_name = classNameRoot + '-data'
    table_header_class_name = classNameRoot + '-header'
    table_header_row_class_name = classNameRoot + '-header-row'
    table_data_row_class_name = classNameRoot + '-data-row'
    if header is None:
        table_header_row = array_to_table_row(col_names,classNameRoot=classNameRoot,is_header=True)
        table_rows = [table_header_row]
    else:
        table_rows = [header]
    previous_index_values = [None]*len(index_cols)
    total_index_elements = len(list(np.concatenate([[x] if type(x) != list else x for x in index_cols]).flat))
    index_element_lens = [1 if type(x) != list else len(x) for x in index_cols]
    for i in range(0,df_copy.shape[0]):
        data_row_i = df_copy.iloc()[i]
        index_row_elements_i = [list(data_row_i[x]) if (isinstance(data_row_i[x], pd.DataFrame)or isinstance(data_row_i[x], pd.Series)) else data_row_i[x] for x in index_cols]
        for j in range(0,len(previous_index_values)):
            if previous_index_values[j] == index_row_elements_i[j]:
                data_row_i[index_cols[j]] = ''
            else:
                previous_index_values[j] = index_row_elements_i[j]
                pre_row_break_count = sum(index_element_lens[:j])
                pre_row_break_data = ['']*pre_row_break_count

                if type(index_row_elements_i[j]) == list:
                    previous_index_values[previous_index_values.index(index_row_elements_i[j])+1:] = [None for x in previous_index_values[previous_index_values.index(index_row_elements_i[j])+1:]]
                    break_value_data = [x for x in previous_index_values[j]]
                else:
                    break_value_data = [previous_index_values[j]]

                post_row_break_count = len(list(data_row_i))-(pre_row_break_count+len(break_value_data))
                post_row_break_data = ['']*post_row_break_count
                break_row = list(np.concatenate([pre_row_break_data,break_value_data,post_row_break_data]).flat)
                blank_row = array_to_table_row(break_row,classNameRoot=classNameRoot,is_header=False)
                table_rows.append(blank_row)
                data_row_i[index_cols[j]] = ''

        all_row_elements_i = list(data_row_i)
        all_row_elements_i = [x if x != 0 else '--' for x in all_row_elements_i]
        table_row_i = array_to_table_row(all_row_elements_i,classNameRoot=classNameRoot,is_header=False)
        table_rows.append(table_row_i)
    table_element = html.Table(children=table_rows,className=table_class_name)
    return table_element

def Generate_HTML_Card(image_path,title,text):
    """Gets and prints the spreadsheet's header columns

    Args:
        file_loc (str): The file location of the spreadsheet
        print_cols (bool): A flag used to print the columns to the console
            (default is False)

    Returns:
        list: a list of strings representing the header columns
    """
    card = html.Div([
        html.Img(src=image_path,className='card-img'),
        html.Div([
            html.P(title,className='card-title')
        ],className='card-title-container'),
        html.Div([
            html.P(text,className='card-text')
        ],className='card-text-container')
    ],className='card card-fade-in')
    return card

def Generate_HTML_Card_Button(button_text,button_href):
    """Gets and prints the spreadsheet's header columns

    Args:
        file_loc (str): The file location of the spreadsheet
        print_cols (bool): A flag used to print the columns to the console
            (default is False)

    Returns:
        list: a list of strings representing the header columns
    """
    file_name = str(button_href).split('/')[-1]
    button = html.A(
        children=[
        html.Button(button_text,className='card-button')
        ],
        href = button_href,
        target="_blank",
        download=file_name,
        rel="noopener noreferrer")
    return button
    
def Generate_HTML_Button_Card(image_path,title,text,button_elements):
    """Gets and prints the spreadsheet's header columns

    Args:
        file_loc (str): The file location of the spreadsheet
        print_cols (bool): A flag used to print the columns to the console
            (default is False)

    Returns:
        list: a list of strings representing the header columns
    """
    card = html.Div([
        html.Img(src=image_path,className='card-img'),
        html.Div([
            html.P(title,className='card-title')
        ],className='card-title-container'),
        html.Div([
            html.P(text,className='card-text')
        ],className='card-text-container'),
        html.Div([button_elements],className='card-button-container')
    ],className='card card-fade-in')
    return card


