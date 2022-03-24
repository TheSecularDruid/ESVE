from dash import html
from dash import dcc
from dash import Dash
import csv



##Data loading
sequences = {}
attr_values = {}


with open("./events.csv",'r') as f:
    r = csv.reader(f,csv.QUOTE_NONNUMERIC)
    attributes = r.__next__()
    for attr in attributes:
        attr_values[attr] = []

    for row in r:
        if row[0] not in sequences.keys():
            sequences[row[0]] = []
        sequences[row[0]].append(row[1:])
        for val_id in range(2,len(row)):
            if row[val_id] not in attr_values[attributes[val_id]]:
                attr_values[attributes[val_id]].append(row[val_id])



for seq in sequences.values():
    seq.sort(key=lambda x:x[0])

cur_seq_ID = list(sequences.keys())[0]

attr_types = {}
for attr in attributes:
    attr_types[attr] = 'Categorical'


##Display
external_stylesheets = [{"rel":"stylesheet"}]
app = Dash(__name__,external_stylesheets=external_stylesheets)

def dropdown_list_or_graph(attr_index):
    if attr_types[attr]=='Categorical':
        DDList = []
        for event in sequences[cur_seq_ID]:
            cur_attribute = attributes[attr_index]
            options = attr_values[cur_attribute]
            value = event[attr_index-1]    # -1 cuz events don't have an "ID" field
            DDList.append(dcc.Dropdown(options=options,value=value))
        return DDList
    if attr_types[attr]=='Numerical':
        pass


def horizontal_box(attr_index):
    return html.Div(children=dropdown_list_or_graph(attr_index),className = 'horizontal_box')

vertical_box = html.Div(children=[horizontal_box(attr_index) for attr_index in range(2,len(attributes))],#not displaying time, ID
                        className = 'vertical_box')


app.layout = html.Div(children = [
                    html.Div(children=[
                        html.P("Choose the sequence"),
                        dcc.Dropdown(options=list(sequences.keys()),value=cur_seq_ID,id='sequence_chooser')],
                        className="sequence_chooser"),
                    vertical_box]
                    )


##Callbacks
#TODO

####Run
if __name__=='__main__':
    app.run_server(debug=True)