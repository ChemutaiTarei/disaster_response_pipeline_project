# Disaster Response Pipeline Project

## Summary
The project involved analysing disaster data from Appen (formally Figure 8) to and building a model for an API that classifies disaster messages. A machine learning pipeline was created to categorize these disaster events. A web app was also created where an emergency worker can input a new message and get classification results in several categories. The web app also displays visualizations of the data.


### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterReponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterReponse.db models/model.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/


### files in the repository
1.	run.py: web application file. It contains the code that starts the Flask server and handles the routing for different pages.
2.	README.md: provides documentation and instructions for the project
3.	process_data.py: contains the data preprocessing step
4.	model.pkl: contains trained machine learning model
5.	master.html and go.html: HTML templates used by Flask application to render the main page and the classification result page, respectively
6.	DisasterResponse.db: the SQLite database file where the preprocessed data is stored
7.	.gitignore: specifies which files should be ignored by Git
8.	train_classifier.py: This file contains the machine learning model using the preprocessed data 

