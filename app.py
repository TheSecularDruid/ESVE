from dash import html
from dash import dcc
from dash import Dash
import pandas as pd


data = pd.read_csv(".\events.csv    ")

#quick loop to only select a couple sequences
n_attr = len(data.columns) - 2 #removing ID and time attribute for now

n_seq = 5
sequences = [[-1,[]] for _ in range(n_seq)]   #First element is sequence ID, second element is a list of events

index = 0
for seq in sequences:
    if data['ID'][index] not in [s[0] for s in sequences]:
        seq[0] = data['ID'][index]
    index+=1

for index in range(data.shape[0]):
    for seq in sequences:
        if data['ID'][index]==seq[0]:
            seq[1].append([data[attr][index] for attr in data.columns[1:]])


for seq in sequences:
    seq[1].sort(key=lambda x:x[0])

attributes_values = [list(set(data[attr])) for attr in data.columns[1:]]
n_attributes = len(attributes_values)
external_stylesheets = [{"rel":"stylesheet"}]

app = Dash(__name__,external_stylesheets=external_stylesheets)


venues = attributes_values[1]

app.layout = html.Div(
    [html.Div(
        [dcc.Dropdown(venues, event[k],className="event")
            for k in range(n_attributes)
        for event in seq[1]],
        className = "sequence"
    )
    for seq in sequences]
)

####separation
if __name__=='__main__':
    app.run_server(debug=True)