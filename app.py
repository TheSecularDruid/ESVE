from dash import html
from dash import dcc
from dash import Dash
import pandas as pd


data = pd.read_csv(".\events.csv    ")

#quick loop to only select a couple sequences
seq1 = []
seq2 = []
seq3 = []
seq1_ID = data['ID'][0]
seq2_ID = data['ID'][1]
seq3_ID = data['ID'][3]

for k in range(len(data['ID'])):
    if data['ID'][k] == seq1_ID:
        seq1.append([data['time'][k],data['Venue'][k]])
    elif data['ID'][k] == seq2_ID:
        seq2.append([data['time'][k],data['Venue'][k]])
    elif data['ID'][k] == seq3_ID:
        seq3.append([data['time'][k],data['Venue'][k]])


seq1.sort(key=lambda x:x[0])
seq2.sort(key=lambda x:x[0])
seq3.sort(key=lambda x:x[0])

venues = list(set(data['Venue']))
external_stylesheets = [{"rel":"stylesheet"}]

app = Dash(__name__,external_stylesheets=external_stylesheets)

sequences = [seq1,seq2,seq3]

app.layout = html.Div(
    [html.Div(
        [dcc.Dropdown(venues, event[1],className="event") for event in seq],
        className = "sequence"
    )
    for seq in sequences]
)

####separation
if __name__=='__main__':
    app.run_server(debug=True)