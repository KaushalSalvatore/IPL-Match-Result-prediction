from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

# Load the Random Forest CLassifier model
score_predict_model_name = 'artifacts/score_prediction_model.pkl'
win_probability_predict_name = 'artifacts/win_probability_model.pkl'
score_predict_regressor = pickle.load(open(score_predict_model_name, 'rb'))
win_probability_regressor = pickle.load(open(win_probability_predict_name, 'rb'))


app = Flask(__name__)


@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    temp_array = list()
    
    if request.method == 'POST':
        
        batting_team = request.form['batting-team']
        if batting_team == 'Chennai Super Kings':
            temp_array = temp_array + [1,0,0,0,0,0,0,0,0,0]
        elif batting_team == 'Delhi Daredevils':
            temp_array = temp_array + [0,1,0,0,0,0,0,0,0,0]
        elif batting_team == 'Kings XI Punjab':
            temp_array = temp_array + [0,0,1,0,0,0,0,0,0,0]
        elif batting_team == 'Kolkata Knight Riders':
            temp_array = temp_array + [0,0,0,1,0,0,0,0,0,0]
        elif batting_team == 'Mumbai Indians':
            temp_array = temp_array + [0,0,0,0,1,0,0,0,0,0]
        elif batting_team == 'Rajasthan Royals':
            temp_array = temp_array + [0,0,0,0,0,1,0,0,0,0]
        elif batting_team == 'Royal Challengers Bangalore':
            temp_array = temp_array + [0,0,0,0,0,0,1,0,0,0]
        elif batting_team == 'Sunrisers Hyderabad':
            temp_array = temp_array + [0,0,0,0,0,0,0,1,0,0]
        elif batting_team == 'Pune Warriors':
            temp_array = temp_array + [0,0,0,0,0,0,0,0,1,0]
        elif batting_team == 'Gujarat Lions':
            temp_array = temp_array + [0,0,0,0,0,0,0,0,0,1]
            
            
        bowling_team = request.form['bowling-team']
        if bowling_team == 'Chennai Super Kings':
            temp_array = temp_array + [1,0,0,0,0,0,0,0,0,0]
        elif bowling_team == 'Delhi Daredevils':
            temp_array = temp_array + [0,1,0,0,0,0,0,0,0,0]
        elif bowling_team == 'Kings XI Punjab':
            temp_array = temp_array + [0,0,1,0,0,0,0,0,0,0]
        elif bowling_team == 'Kolkata Knight Riders':
            temp_array = temp_array + [0,0,0,1,0,0,0,0,0,0]
        elif bowling_team == 'Mumbai Indians':
            temp_array = temp_array + [0,0,0,0,1,0,0,0,0,0]
        elif bowling_team == 'Rajasthan Royals':
            temp_array = temp_array + [0,0,0,0,0,1,0,0,0,0]
        elif bowling_team == 'Royal Challengers Bangalore':
            temp_array = temp_array + [0,0,0,0,0,0,1,0,0,0]
        elif bowling_team == 'Sunrisers Hyderabad':
            temp_array = temp_array + [0,0,0,0,0,0,0,1,0,0]
        elif batting_team == 'Pune Warriors':
            temp_array = temp_array + [0,0,0,0,0,0,0,0,1,0]
        elif batting_team == 'Gujarat Lions':
            temp_array = temp_array + [0,0,0,0,0,0,0,0,0,1]
           
        overs = float(request.form['overs'])
        runs = int(request.form['runs'])
        wickets = int(request.form['wickets'])
        runs_in_prev_5 = int(request.form['runs_in_prev_5'])
        wickets_in_prev_5 = int(request.form['wickets_in_prev_5'])
       
        temp_array = temp_array + [overs, runs, wickets, runs_in_prev_5, wickets_in_prev_5]
       

        data = np.array([temp_array])
        print(data)
        my_prediction = int(score_predict_regressor.predict(data)[0])
              
        return render_template('result.html', lower_limit = my_prediction-10, upper_limit = my_prediction+5)
    


@app.route('/winprediction', methods=['GET','POST'])
def win_probability_predict():
    if request.method == 'GET':
        return render_template('win_probability.html')
    else:
        batting_team = request.form['batting-team']
        bowling_team = request.form['bowling-team']
        city = request.form['city']
        target = int(request.form['target'])
        score = int(request.form['score'])
        overs = float(request.form['overs'])
        wickets = int(request.form['wickets']) 

        runs_left = target - score
        balls_left = 120 - (overs*6)
        wickets = 10 - wickets
        crr = score/overs
        rrr = (runs_left*6)/balls_left
        input_df = pd.DataFrame({'batting_team':[batting_team],
                                 'bowling_team':[bowling_team],
                                 'city':[city],
                                 'runs_left':[runs_left],
                                 'balls_left':[balls_left],
                                 'wickets':[wickets],
                                 'total_runs_x':[target],
                                 'crr':[crr],
                                 'rrr':[rrr]})
        

        result = win_probability_regressor.predict_proba(input_df)
        loss = result[0][0]
        win = result[0][1]
        print({loss})
        print({win})
        return render_template('win_probability.html')

    


if __name__ == '__main__':
	app.run(debug=True)