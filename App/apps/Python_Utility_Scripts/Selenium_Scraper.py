import pandas as pd
import numpy as np
import selenium
from selenium import webdriver
from selenium.webdriver.support.ui import Select
from selenium.webdriver.common.keys import Keys
import unidecode
import time
import pathlib
import os


##############################################################
# Dash requires pathlib standardized paths for reading
# instructions the primary used directories are declared here
# ############################################################
PATH = pathlib.Path(__file__).parent
DATA_PATH = PATH.joinpath("../../Data").resolve()
SCRAPING_PATH = PATH.joinpath("../../Data/Scraping Results").resolve()
GEKKO_PATH = PATH.joinpath("../../Selenium/gecko").resolve()

def Generate_Athlete_Search_Aliases(athlete_full_name):
    """Generates and returns an array of potential athlete 
    aliases derived from their input full name

    Args:
        athlete_full_name(str): full name of athlete you wish
        to generate aliases for

    Returns:
        np.array: array of generate possible athelete aliases
    """
    result = []
    full_name = athlete_full_name.lower()
    first_name = full_name.split(' ')[0].lower()
    last_name = full_name.split(' ')[-1].lower()
    result.append([first_name,last_name])
    result.append(["",last_name])
    result.append([first_name,""])
    decoded_first_name = unidecode.unidecode(first_name).lower()
    decoded_last_name = unidecode.unidecode(last_name).lower()
    if (decoded_last_name != last_name) or (decoded_first_name != first_name):
        result.append([decoded_first_name,decoded_last_name])
        result.append(["",decoded_last_name])
        result.append([decoded_first_name,""])
    result_array = np.array(result)
    return result_array


def levenshtein_ratio(s, t):
    """Calculates and returns levenshtein edit distance
    between two strings

    Args:
        s (str): string 1
        t (str): string 2

    Returns:
        float: levenshtein distance
    """
    rows = len(s)+1
    cols = len(t)+1
    distance = np.zeros((rows,cols),dtype = int)
    for i in range(1, rows):
        for k in range(1,cols):
            distance[i][0] = i
            distance[0][k] = k
            
    for col in range(1, cols):
        for row in range(1, rows):
            if s[row-1] == t[col-1]:
                cost = 0 
            else:
                cost = 2
        
            distance[row][col] = min(distance[row-1][col] + 1,distance[row][col-1] + 1,distance[row-1][col-1] + cost)
    Ratio = ((len(s)+len(t)) - distance[row][col]) / (len(s)+len(t))
    return Ratio

def Get_Athlete_Bio(driver,athlete_full_name,athlete_country,athlete_gender,verbose=True):
    """Scapres FIS for a given athlete's biographical information

    Args:
        driver (selenium driver object): currenlty open selenium driver object
        athlete_full_name(str): full name of searched athlete
        athlete_country(str): USOPC noc designation for country they compete for
        athlete_gender(str): USOPC gender designation
        verbose(bool): if true outputs text feedback on the scraping process


    Returns:
        str: best_entry_name found to match the search according to leven distance
        float or int: best_entry_age identified age of the best matching entry
        datetime: best_entry_birthday identified age of the best matching birthdate
    """
    best_entry_name = 'Unknown'
    best_entry_age = 'Unknown'
    best_entry_birthday = 'Unknown'
    similarity_threshold = 0.75
    alias_array = Generate_Athlete_Search_Aliases(athlete_full_name)
    page_scores = []
    page_entries = []
    for i in range(0,alias_array.shape[0]):
        if i > 2:
            name_to_compare = alias_array[3,0]+' '+alias_array[3,1]
            name_to_compare_transposed = alias_array[3,1]+' '+alias_array[3,0]
        else:
            name_to_compare = alias_array[0,0]+' '+alias_array[0,1]
            name_to_compare_transposed = alias_array[0,1]+' '+alias_array[0,0]
            
        search_result_dict = Search_Athlete(driver=driver,first_name=alias_array[i,0],last_name=alias_array[i,1],athlete_country=athlete_country,athlete_gender=athlete_gender)
        search_result_keys = list(search_result_dict.keys())
        if len(search_result_keys) != 0:
            for j in range(0,len(search_result_keys)):
                cleaned_search_result_name = search_result_keys[j].lower()
                similarity_score_j = levenshtein_ratio(cleaned_search_result_name,name_to_compare)
                similarity_score_j_transposed = levenshtein_ratio(cleaned_search_result_name,name_to_compare_transposed)
                similarity_score_j = max(similarity_score_j,similarity_score_j_transposed)
                page_scores.append(similarity_score_j)
                page_entry_dict = dict()
                page_entry_dict[search_result_keys[j]] = search_result_dict[search_result_keys[j]]
                page_entries.append(page_entry_dict)
                if similarity_score_j >= .99:
                    break
        if len(page_scores) != 0:
            if np.max(page_scores) >= .99:
                break
        time.sleep(2+np.random.rand())
    if len(page_scores) != 0:
        best_similarity_score = np.max(page_scores)
        if best_similarity_score >= similarity_threshold:
            max_page_index = np.argmax(page_scores)
            best_entry_dict = page_entries[max_page_index]
            best_entry_name = list(best_entry_dict.keys())[0]
            best_entry_age = best_entry_dict[best_entry_name]['age']
            best_entry_birthday = best_entry_dict[best_entry_name]['birthdate']
            if verbose:
                print(f'found entry for {athlete_full_name} under name {best_entry_name} with {round(best_similarity_score*100,2)}% similarity')
            

    return best_entry_name,best_entry_age,best_entry_birthday


def Search_Athlete(driver,first_name,last_name,athlete_country,athlete_gender):
    """mid-level function which does the actual scraping of the athlete bio

    Args:
        driver (selenium driver object): currenlty open selenium driver object
        athlete_first_name(str): first name of searched athlete
        athlete_last_name(str): last name of serched athlete
        athlete_country(str): USOPC noc designation for country they compete for
        athlete_gender(str): USOPC gender designation

    Returns:
        dict: returns a dict of the given ahtletes scraped result
    """
    output_dict = dict()
    base_url = 'https://www.fis-ski.com/DB/general/biographies.html'
    driver.get(base_url)
    time.sleep(6+np.random.rand())
    last_name_entry_box = driver.find_element_by_id('form_lastname')
    last_name_entry_box.send_keys(last_name)
    time.sleep(2+np.random.rand())
    first_name_entry_box = driver.find_element_by_id('form_firstname')
    first_name_entry_box.send_keys(first_name)
    time.sleep(2+np.random.rand())
    country_entry_box = driver.find_element_by_id('form_nationcode')
    country_entry_box.send_keys(athlete_country)
    time.sleep(2+np.random.rand())
    gender_select_div = driver.find_element_by_id('select_gendercode')
    drop_down_selection_div = gender_select_div.find_element_by_class_name('selectric')
    open_drop_down_button = drop_down_selection_div.find_element_by_tag_name('b')
    open_drop_down_button.click()
    time.sleep(2+np.random.rand())
    drop_down_selection_items = gender_select_div.find_element_by_class_name('selectric-items')
    drop_down_list_items = drop_down_selection_items.find_elements_by_tag_name('li')
    if athlete_gender == 'Male':
        drop_down_list_items[2].click()
    else:
        drop_down_list_items[1].click()
        
    time.sleep(1+np.random.rand())
    form_inner = driver.find_element_by_class_name('form__inner')
    search_button = form_inner.find_element_by_class_name('btn_yellow')
    search_button.click()
    time.sleep(2+np.random.rand())
    athlete_results_div = driver.find_element_by_id('athletes-search')
    table_body = athlete_results_div.find_element_by_class_name("tbody")
    table_rows = table_body.find_elements_by_tag_name('a')
    for i in range(0,len(table_rows)):
        row_i = table_rows[i]
        table_div_container = row_i.find_element_by_class_name('container')
        row_divs = table_div_container.find_elements_by_tag_name('div')
        table_athlete_name = row_divs[4].text
        table_athlete_age = row_divs[7].text
        table_athlete_birthdate = row_divs[8].text
        if table_athlete_age == ' ':
            table_athlete_age = 'Unknown'
        if table_athlete_birthdate == ' ':
            table_athlete_birthdate = 'Unknown'
        inner_dict = dict()
        inner_dict['age'] = table_athlete_age
        inner_dict['birthdate'] = table_athlete_birthdate
        output_dict[table_athlete_name] = inner_dict
    return output_dict

def scrape_bios_from_df(driver,df):
    """Top level scraping function which scrapes all entires
    in the missing data dataframe

    Args:
        df (pd.dataframe): dataframe of missing athletes compiled from the data cleaning script

    Returns:
        dataframe: results of the datascraping process as dataframe
    """
    df_copy = df.copy()
    athlete_names = np.array(df_copy['Athlete_Name'])
    athlete_countries = np.array(df_copy['Country_Abbv'])
    athlete_genders = np.array(df_copy['Athlete_Gender'])
    recorded_birthdays = []
    recorded_ages = []
    recorded_names = []
    athletes_with_no_close_entry = 0
    athletes_still_unknown = 0
    athletes_found = 0
    athletes_completed = 0
    athletes_total = athlete_names.shape[0]
    for i in range(0,athlete_names.shape[0]):
        athlete_name_i = athlete_names[i]
        athlete_country_i = athlete_countries[i]
        athlete_gender_i = athlete_genders[i]
        try:
            best_entry_name,updated_age_i,updated_birthdate_i = Get_Athlete_Bio(driver=driver,athlete_full_name=athlete_name_i,athlete_country=athlete_country_i,athlete_gender=athlete_gender_i,verbose=True)
        except:
            print(f'**Scraper failed with an error for athlete {athlete_name_i}**')
            best_entry_name = 'Unknown'
            update_age_i = 'Unknown'
            updated_birthdate_i = 'Unknown'
            
        if best_entry_name == 'Unknown':
            print(f'Unable to find any close entry to athlete {athlete_name_i}')
            athletes_with_no_close_entry+=1
        
        if (updated_birthdate_i == 'Unknown') and (updated_age_i == 'Unknown'):
            athletes_still_unknown +=1
        else:
            athletes_found +=1
            
        athletes_completed +=1
        recorded_birthdays.append(updated_birthdate_i)
        recorded_ages.append(updated_age_i)
        recorded_names.append(best_entry_name)
        if athletes_completed % 10 == 0:
            print(f'Iteration Number {athletes_completed}')
            print(f'Total Athletes Found = {athletes_found}')
            print(f'Total Athletes Failed = {athletes_with_no_close_entry}')
            print(f'Total Athletes Located With No Age or Date = {athletes_still_unknown}')
            print(f'Success Percentage = {round(athletes_found/athletes_completed,2)}')
            print(f'Total Progress Complete = {round(athletes_completed/athletes_total,2)}')
            print('----------------------------------------------------------------------')
        time.sleep(2)
    
    
    df_copy['Athlete Scraped Birthdays'] = np.array(recorded_birthdays)
    df_copy['Athlete Scraped Ages'] = np.array(recorded_ages)
    df_copy['Athlete Scraped Name'] = np.array(recorded_names)
    driver.close()
    return df_copy

##############################################################
# when the scraper returns just a year for value
# then the unbiased estimate of birthdate is the midpoint date
# ############################################################
def format_year_only_dates_to_midpoint(df):
    df_copy = df.copy()
    birthdates_array = np.array(df_copy['Athlete Scraped Birthdays'])
    for i in range(0,birthdates_array.shape[0]):
        if len(birthdates_array[i]) == 4:
            birthdates_array[i] = '1/7/'+birthdates_array[i]
        else:
            pass
    df_copy['Athlete Scraped Birthdays'] = birthdates_array
    return df_copy


##############################################################
# converts scraped entires into format consisitent with the 
# existing data entries
# ############################################################ 
def format_scraped_athlete_results(scraping_results):
    df_success = scraping_results.loc[(scraping_results['Athlete Scraped Ages'] != 'Unknown')|(scraping_results['Athlete Scraped Birthdays'] != 'Unknown')]
    df_success = format_year_only_dates_to_midpoint(df_success)
    df_success['Athlete Scraped Birthdays'] = pd.to_datetime(df_success['Athlete Scraped Birthdays'], infer_datetime_format=True)
    df_output_to_merge = df_success[['Athlete_ID','Athlete Scraped Birthdays']]
    return df_output_to_merge

def run():
    df = pd.read_csv(SCRAPING_PATH.joinpath('Missing_Athlete_Birthdays.csv'))
    gecko_path = GEKKO_PATH.joinpath("geckodriver.exe")
    driver = webdriver.Firefox(executable_path=gecko_path)
    output = scrape_bios_from_df(driver,df)
    output.to_csv(SCRAPING_PATH.joinpath('Alpine_Skiing_Scraping_Results.csv'),index=False,encoding='utf-8')
    formatted_output = format_scraped_athlete_results(output)
    formatted_output.to_csv(SCRAPING_PATH.joinpath('Alpine_Skiing_Scraping_Entries_to_Merge.csv'),index=False)

if __name__ == '__main__':
    run()
