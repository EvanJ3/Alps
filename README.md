<div style='display:flex;flex-direction:column;width:100%;justify-content:center;align-items:center'>
    <h1 style='text-align:center'>
        Alps
    </h1>
    <h3 style='text-align:center'>
        An Age Focused Analysis, Framework, & Application for Modeling Olympic Medal Achievement & Age
    </h3>
</div>

![Alpine Dashboard Logo](./App/assets/images/Github_Readme_Lead.png)

Alps is a powerful web-application that allows users to better understand and analyze alpine skiing through the primary lens of age. This application leverages several interactive visualization tools and three unique dashboard pages to provide a comprehensive analysis of olympic medal achievement and age.

## Features

Here are a few of the things our application offers:

- Query detailed alpine athlete biographical information
- Examine individual athlete seasonal performance
- Explore a given athlete's frequency of compeititon by alpine dicipine and season
- Analyze alpine athlete historical podiums
- Evaluate competition difficulty
- View historcal world rankings for athletes by discipline
- Prototype and evaluate KDE models for three key perspectives of age modeling in alpine skiing
- Visualize peak athlete age ranges by dicsipline and gender
- Train and plot smoothing models for athlete career trajectories
- Perform KDE cross-validation on a variety of age based models
- Compare and chart identified peak ages for a given event and gender
- Analyze a countries olympic performance over time
- Contrast two olympic nations and their respecitve athletetic pipelines
- Aggregate potential olympic prospect tables for a given country and olympic games by gender and event

## Installation

The installation and setup for our application has two possible approaches each with their own benfits. The first and reccomended option is setting up the application through docker. The second approach is to download the repository directly and run in your own cloned local enviornment. We will provide detailed instructions for both below:

### Installing via Docker

1. Clone and unzip the repository from github

2. Install [Docker Desktop](https://www.docker.com/products/docker-desktop/). Note, Windows users will also need to install Windows [Subsystem for Linux (WSL2)](https://docs.microsoft.com/en-us/windows/wsl/install) which is a dependency to run docker on Windows machines.

3. Build the image from source via running the following docker command in command line or terminal from the top level project directory:

```bash
docker build -t alps:latest .
```

4. After successfully building the local docker container from source, you can run the container via docker desktop or by running the following docker commmand in command line or terminal

```bash
docker run -p 8080:8000 -ti alps:latest
```

5. After startup is complete the web application will automatically spin up. In order to access the local running web application you will need to navigate to the localhost port the application is being served from. To do this enter the following address into your addressbar http://127.0.0.1:8080/ . After navigating to this link the application should be viewable

6. Once you are finished using the application be sure to shutdown your docker container. This can be accomplished within docker desktop by navigaing to "Containers" and selecting the "kill" option for our image "usopc-alpine-dashabord". Additionally, you can also kill the process by running the following docker command in command line or terminal:

```bash
docker kill container-name alps:latest
```

### Installing via Github

1. Download and unzip the repository in your desired local directory
2. Install [anaconda/miniconda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) locally
3. From the project's top level directory run the following command in anaconda terminal

```bash
conda env create --file environment.yml
```

4. After successful installation, activiate this new Conda environment with the following command:

```bash
conda activate UsopcAlpine 
```

5. Navigate to the top level project dirctory and run the app.py file via the following command

```bash
cd App #navigate to top level project directory
python app.py 
```

6. After the application launches navigate to the default launch port via your web browser. http://127.0.0.1:8050/

7. After you are finished using the application be sure to kill the process by navigating to the application's terminal window and pressing ctrl+C to kill the process. Then exit the terminal session via running.

```bash
exit()
```

## Directory Structure & File Usage

``` bash

USOPC-ALPINE-SKIING
|   
│   README.md: Project overview and installation instructions
│   Dockerfile: Dockerfile used to build the local docker image for the repository from source
│   environment.yml: YAML file used to store the commands neccecary to clone our local conda enviornment
│   
└── App: The top-level project directory which contains all files neccecary to run the application
    |   
    |   __pycache__: pre-compiled python script cython/binaries
    |   app.py: Entry-point file when run will launch the dash application on a local flask webserver
    |
    └── apps
        |
        |   __init__.py: Tells dash to treat our directory as an importable module (neccecary for multi-page apps)
        |   Individual_Dash.py: Individual dashboard page layout used by app.py
        |   Event_Dash.py: Event dashboard page layout used by app.py
        |   Country_Dash.py: Country dashboard page layout used by app.py
        |   Methodology.py: Methodology dashboard page layout used by app.py
        |   Uploads.py: Uploads dashboard page layout used by app.py
        |   Dash_Utilities.py: Custom utlity functions and routines for the creation of reusable dash components
        |   Individual_Dash_Utilities.py: Unique utility functions and routines used by the individual dash page
        |   Event_Dash_Utilities.py: Unique utility functions and routines used by the event dash page
        |   Country_Dash_Utilities.py: Unique utility functions and routines used by the country dash page
        |   Methodology_Dash_Utilities.py: Unique utility functions and routines used by the methodology dash page
        |   Upload_Dash_Utilities: Unique utility functions and routines used by the uploads dash page
        |   Data_Cleaning_Utility_Functions.py: Script used for the cleaning and reconsiliation of user uploaded data
        |   Splines.py: Python implementations of several spline fitting and cross-validation routines
        |
        ├── pycache__: pre-compiled python script cython/binaries
        |       └───...
        |
        ├── Python Utility Scripts: Directory contains several utiilty scripts
        |       Competition_Rankings.py: Calculates and writes competition difficulty ratings
        |       Create_ISO_FLAG_XREF.py: Generate the ISO flag xref table
        |       Event_Age_Signficance_CV.py: Runs cross-validation across all event gender pairs for event peak optimal ages
        |       Format_Scraping_Results.py: Formats scraped data into format fit to merge with base dataframe
        |       Generate_Posthoc_Olympic_Results.py: Aggregate and write known olympic medal results by country, event, games
        |       Medal_Prediction_CV.py: Run Medal Prediction Crossvalidation for all models
        |       Merge_FIS_Ids.py: Merges FIS ids with athlete names generates FIS xref table
        |       Rank_Age_Significance_CV.py: Runs cross-validation across all event gender pairs for indv peak optimal ages
        |       Selenium_Scraper.py: Data Scraper entry point
        |       Splines.py: Utility functions for fitting and cross-valiation of smoothing splines
        |       Tabulate_Olympic_Rankings.py: Tabulates and writes olympic rankings
        |       WR_Age_Significance_CV.py: Runs cross-validation across all event gender pairs for indv wr optimal ages
        |
        ├── assets: folder containing all our dash styling sheets, icons, and images. (name and location cannot be altered)
        |       |
        |       |   _animations.scss: SCSS utility functions for globally imported animations
        |       |   _cards.scss: SCSS utility functions for globally imported card styles
        |       |   _dash.scss: SCSS utility functions for globally imported dash component styles
        |       |   _footer.scss: SCSS utility functions for globally imported footer styles
        |       |   _globals.scss: SCSS utility functions for globally imported container styles
        |       |   _header.scss: SCSS utility functions for globally imported navigation bar styles
        |       |   _mixins.scss: SCSS utility functions for globally imported mixin breakpoints
        |       |   _variables.scss: SCSS utility functions for globally imported color and font variables
        |       |   Individual_Dash.scss: Individual dashboard unique SCSS style sheet
        |       |   Individual_Dash.css: Compiled css styles for the Individual Dash derived from SCSS representation
        |       |   Individual_Dash.css.map: Source mapping from SCSS to CSS for Individual Dash Styles
        |       |   Event_Dash.scss: Event dashboard unique SCSS style sheet
        |       |   Event_Dash.css: Compiled css styles for the Event Dash derived from SCSS representation
        |       |   Event_Dash.css.map: Source mapping from SCSS to CSS for Event Dash Styles
        |       |   Country_Dash.scss: Country dashboard unique SCSS style sheet
        |       |   Country_Dash.css: Compiled css styles for the Country Dash derived from SCSS representation
        |       |   Country_Dash.css.map: Source mapping from SCSS to CSS for Country Dash Styles
        |       |   Methodology_Dash.scss: Methodology dashboard unique SCSS style sheet
        |       |   Methodology_Dash.css: Compiled css styles for the Methodology Dash derived from SCSS representation
        |       |   Methodology_Dash.css.map: Source mapping from SCSS to CSS for Methodology Dash Styles
        |       |
        |       ├── flags:directory containing svgs of all world flags named according to their ISO country code
        |       |       └───...
        |       |
        |       ├── icons: directory contains svg icons for a variety of visual elements used in the web application
        |       |       └───...
        |       | 
        |       ├── images: directory contains several png and jpeg images used in the web application
        |       |       └───...
        |       | 
        |       └─── Methodology Dash Images: contains preview images of the application components for methodlolgy dash
        |               └───...
        |       
        ├── Data:
        |    |
        |    | Alpine_Skiing_Cleaned.csv: The most up-to-date cleaned representation of the alpine skiing data
        |    | AlpineSkiing.csv: The most up-to-date pre-cleaned alpine data which includes sucessful user uploaded rows
        |    | 
        |    ├── Backup Alpine Data: Folder contains databack ups for application critical data for use in case of corruption
        |    |       └───...
        |    |
        |    ├── Data Resources: Contains several data dictionary and example files for better understanding data sources
        |    |       └───...
        |    |
        |    ├── Derived Views: Directory contains several tables pre-calcuated for efficent application processing
        |    |       
        |    |       ML_Input_Data.csv: pre-calculated and formatted for direct use with our medal prediction models
        |    |       Olympic_Athlete_Data.csv: pre-aggregated potential athletes for a given olympic games
        |    |       Olympic_Posthoc_Results.csv: pre-calculated olympic medal scores and frequencies by country and olympics
        |    |       Timed_Competition_Difficulty.csv: pre-calculated competition difficulty ratings and scores
        |    |       Olympic_Ranking_Data: pre-calculated olympic rankings by country event and gender
        |    | 
        |    ├── FIS Data: Bi-yearly FIS athlete point sheets for use in identifying athlete FIS for image scraping
        |    |       └───...
        |    |
        |    ├── Methodoloy Data: Folder containing data neccecary for the generation of methodology dash cards
        |    |
        |    |      Methodology_Components.csv: dash component description data for use in methodology dash cards
        |    |    
        |    ├── Modeling Results: Folder contains the data resulting from model training, evaluation, and testing
        |    |      
        |    |      Event_KDE_Result_DF.csv: Event kde selected peak rank age ranges by sport and gender
        |    |      Indv_KDE_Result_DF.csv: Individual kde selected peak rank age ranges by sport and gender
        |    |      Indv_WR_KDE_Result_DF.csv: Individual kde selected world record peak age ranges by sport and gender
        |    |      Medal_Prediction_CV_Results.csv: training, evaluation, and testing aggregated for all not nn models
        |    |      NN_Model_Histories.csv: Neural Network training, evaluation, and testing aggregated for models 0-4
        |    |      Regression Model Stats.csv: ANOVA and error statistics for all regression modeling approaches
        |    |      
        |    ├── Scraping Results: Directory contains data files for input/output of the data scrapper
        |    |      
        |    |      Alpine_Skiing_Scraping_Entries_to_Merge.csv: Formated scraping results ready to merge with base dataset
        |    |      Alpine_Skiing_Scraping_Results.csv: Scraper output results (scraper output)
        |    |      Missing_Athlete_Birthdays.csv: Entries identified during cleaning process to scrape FIS (scraper input)
        |    | 
        |    ├── Xref: Directory contains several reference mappings used in the webapplication
        |    | 
        |    |      country.json: ISO Country Names -> ISO Country Code (used in flag image selection)
        |    |      ISO_NOC_Flag_XREF.csv: NOC Country Names -> ISO Country Names (used in flag image selection)
        |    |      FIS_CompID_XREF.csv: USOPC Person ID -> FIS Person ID (used in athlete bio image selection)
        |    |      
        |    └─── User Uploaded Data: Directory stores all successful user data uploads from the upload dashboard
        |           └───...   
        |
        ├── Deliverables: Contains all deliverables associated with the project
        |    |
        |    |
        |    ├── Bi-weekly Updates: contains all bi-weekly updates and corresponding schedules submitted
        |    |       └───...  
        |    |
        |    ├── Modeling Images: contains all modeling images utilized in presenting final reports and slides
        |    |       └───... 
        |    |
        |    ├── Presentations: contains all midterm and final report presentations as pptx files
        |    |       └───... 
        |    |
        |    ├── Report: contains the final report and associated files
        |    |       └───... 
        |    |
        |    └─── Research Reference: contains all papers cited in the final report for future reference
        |            └───... 
        |    
        ├── Models:
        |    |
        |    |
        |    ├── Lasso: Contains the pickeled optimal lasso model used in medal prediction modeling
        |    |       └───... 
        |    |
        |    |
        |    ├── Linear Regression: Contains the pickeled optimal linear regression model used in medal prediction modeling
        |    |       └───... 
        |    |
        |    |
        |    ├── Ridge: Contains the pickeled optimal ridge model used in medal prediction modeling
        |    |       └───... 
        |    |
        |    ├── Neural Networks: Contains the H5 trained optimal tensorflow model used in medal prediction modeling
        |    |       └───... 
        |    |
        |    └─── Regression Trees: Contains the pickeled optimal regression tree model used in medal prediction modeling
        |            └───...
        |  
        ├── Selenium: contains drivers neccecary to run selenium data scraper
        |            |
        |            └─── Gecko Driver for your OS needs to go here
        |
        ├── Tableau: contains files used in the creation of the embedded tableau visualization in country dash
        |      
        |       Alpine_World_Medal_Counts.twb: tableau workbook used to generate embedded tableau visualization
        |       Tableau_World_Medal_Counts_Embedding_Link.txt: text file with the dashboards embedable tabeau public url
        |      
        └─── Jupyter: contains various jupyter notebooks used witin the project
```

### Running Python Utilities

Contained within the application are several python files (most under "Python Utility Scripts" directory). Each of these files is responsible for a unqiue aspect of the application. Most of these files won't need to be used as the application handles calling them as needed. However, several scrips such as the data scraper and all modeling scrips may need to be run on demand by users in order to update or re-use them. With the excpetion of the data scraper all files within this utility class of scripts can be run via the following command in terminal or command line from the parent directory of the script:

```bash
python filename.py
```

### Running the Scraper

In order to used the provided data scraper for augmenting our alpine dataset with FIS entries you will need to accomplish a few setup steps and additional installation. 

1. Install and update Firefox or Google Chrome to the latest version
2. Install Selenium drivers. Selenium requires a driver to interface with the chosen browser. Firefox, for example, requires geckodriver, which needs to be installed before the below examples can be run. Make sure it’s in your PATH, e. g., place it in /usr/bin or /usr/local/bin.

Failure to observe this step will give you an error selenium.common.exceptions.WebDriverException: Message: ‘geckodriver’ executable needs to be in PATH.

Other supported browsers will have their own drivers available. Links to some of the more popular browser drivers follow.
```
Chrome: 	https://sites.google.com/chromium.org/driver/
Edge: 	    https://developer.microsoft.com/en-us/microsoft-edge/tools/webdriver/
Firefox: 	https://github.com/mozilla/geckodriver/releases
Safari: 	https://webkit.org/blog/6900/webdriver-support-in-safari-10/
```
3. After verifying that your driver and selenium installations are communicating with the web you are ready to run the web scraper. The web scraper targets athlete entries according to the missing athletes table in the repository's data folder ensure that your search targets are included in this csv. Running the web scraper for even a small number of entries will take a LONG time. This is intentional due to the testing of various combinations of names, but also to ensure that our query server won't be overloaded and lock out the scraper due to DDOS suspicion. 



## Tutorials

Although doccumentation is provided at various levels of detail in the individual files the application runs on, there may be some level of background knowlege needed to fully understand the code. If you aren't familiar with some of the technologies or languages used within our project, that's perfectly fine. Below we have listed a few links to popular resources to better understand, utilize, and operate all aspects of our repository.

- [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html)
- [Python](https://docs.python.org/3/tutorial/)
- [JupyterLab/Jupyter Notebooks](https://jupyterlab.readthedocs.io/en/stable/)
- [Flask](https://flask.palletsprojects.com/en/2.0.x/tutorial/index.html)
- [Sklearn](https://scikit-learn.org/stable/tutorial/index.html)
- [SASS](https://sass-lang.com/guide)
- [Plotly Dash](https://dash.plotly.com/installation)
- [Pandas](https://pandas.pydata.org/pandas-docs/stable/getting_started/tutorials.html)
- [Numpy](https://numpy.org/doc/stable/user/quickstart.html)
- [Tensorflow](https://www.tensorflow.org/tutorials)
- [Docker](https://www.docker.com/101-tutorial/)
- [Selenium](https://selenium-python.readthedocs.io/getting-started.html)