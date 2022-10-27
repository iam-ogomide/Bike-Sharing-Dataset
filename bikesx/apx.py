import flask
import pandas as pd
from joblib import dump, load


with open('line.joblib', 'rb') as f:
    model = load(f)


app = flask.Flask(__name__, template_folder='templates')


@app.route('/', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'GET':
        return (flask.render_template('main.html'))

    if flask.request.method == 'POST':
        season = flask.request.form['season']
        yr = flask.request.form['yr']
        mnth = flask.request.form['mnth']
        holiday = flask.request.form['holiday']
        weekday = flask.request.form['weekday']
        workingday = flask.request.form['workingday']
        weathersit = flask.request.form['weathersit']
        temp = flask.request.form['temp']
        atemp = flask.request.form['atemp']
        hum = flask.request.form['hum']
        windspeed = flask.request.form['windspeed']
        casual = flask.request.form['casual']
        registered = flask.request.form['registered']

        input_variables = pd.DataFrame([[season, yr, mnth, holiday, weekday, workingday, weathersit, temp, atemp, hum, windspeed, casual, registered]],
                                       columns=['season', 'yr', 'mnth', 'holiday', 'weekday',
                                                'workingday', 'weathersit', 'temp', 'atemp', 'hum', 'windspeed', 'casual', 'registered'],
                                       dtype='float',
                                       index=['input'])

        predictions = model.predict(input_variables)[0]
        print(predictions)

        return flask.render_template('main.html', original_input={'season': season, 'yr': yr, 'mnth': mnth, 'holiday': holiday, 'weekday': weekday, 'workingday': workingday, 'weathersit': weathersit, 'temp': temp, 'atemp': atemp, 'hum': hum, 'windspeed': windspeed , 'casual': casual , 'registered': registered},
                                     result=predictions)


if __name__ == '__main__':
    app.run(debug=True)
