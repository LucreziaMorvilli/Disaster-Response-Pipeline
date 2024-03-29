# Disaster Response Pipeline Project
Part of Data Scientist Nanodegree (Udacity)

### Aim:
Analyze disaster data to build a model for an API that classifies disaster messages.

### Project Description:
I used a data set containing real messages that were sent during disaster events and then created a machine learning pipeline to categorize these events so that messages can be sent to an appropriate disaster relief agency.

The project includes a web app where an emergency worker can input a new message and get classification results in several categories. The web app will also display visualizations of the data.

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

### Screenshots Web App

![Sample Input](screenshots/one.png)

#### Example:
![Sample Output](screenshots/five.png)

#### Visualizations of the data:
![two](screenshots/two.png)
![three](screenshots/three.png)
![four](screenshots/four.png)



