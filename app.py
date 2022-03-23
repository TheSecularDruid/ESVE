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


##App setup
external_stylesheets = [{"rel":"stylesheet"}]
app = Dash(__name__,external_stylesheets=external_stylesheets)


# app.layout = html.Div(
#     dcc.Dropdown(sequences.keys(),cur_seq_ID,id='sequence_chooser'),
#     html.Div([
#             html.Div(
#                 [dcc.Dropdown(
#                     attr_values[attr],
#                     sequences[cur_seq_ID][event_index],
#                     className='event')
#                 for event_index in range(len(sequences[cur_seq_ID]))],
#                 className='sequence')
#         for attr_index,attr_type in enumerate(attr_types.values())]
#
#     )
# )
#
#
# [[(attr_values[attr],sequences[cur_seq_ID][event_index])
#                 for event_index in range(len(sequences[cur_seq_ID]))]
#         for attr_index,attr_type in enumerate(attr_types.values())]
#



####Run
if __name__=='__main__':
    app.run_server(debug=True)