from dash import html
from dash import dcc
from dash import Dash
from dash import Input
from dash import Output
from dash import State
from dash import callback_context
from dash import ALL
from dash.exceptions import PreventUpdate
from plotly import express as px
import pandas as pd
import csv
import requests
from json import loads


##Data loading
def load_data_from_disk():
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
        f.close()


    for seq in sequences.values():
        seq.sort(key=lambda x:x[0])

    attr_types = {}
    for attr in attributes:
        attr_types[attr] = 'Categorical'

    attributes_data = {'names':attributes,'values':attr_values,'types':attr_types}
    return sequences,attributes_data

def load_data_from_MAQUI():
    try:
        context = requests.get('localhost:5000/getCurrentContext')
        if not context["upToDate"]:
            raise preventUpdate

        context.raise_for_status()

        params = {"panelID":context["Pannel"],'ForSID':context['Focus'],'outputType':context['Output']}
        data = requests.get('localhost:5000/getCurrentData',params=params)
        data.raise_for_status()

        return data
    except requests.exceptions.RequestException as e:
        print(e)
        print(e.text)
        print("ya fucked up bra")

##Display
external_stylesheets = [{"rel":"stylesheet"}]
app = Dash(__name__,external_stylesheets=external_stylesheets)

def dropdown_list_or_graph(attr_index,cur_seq_ID,attr_data,sequences):
    attr_name = attr_data["names"][attr_index]
    id_list = []
    if attr_data["types"][attr_name]=='Categorical':
        DDList = []
        for event_id, event in enumerate(sequences[cur_seq_ID]):
            options = attr_data['values'][attr_name]
            value = event[attr_index-1]    # -1 cuz events don't have an "ID" field
            dd_id = {'place':str(attr_index) + 'c' + str(event_id),'type':'attr_value_dd'}
            DDList.append(dcc.Dropdown(options=options,value=value,className="categ_event_dd",id=dd_id,clearable=False))
            id_list.append(dd_id)
        return DDList
    if attr_data["types"][attr_name]=='Numerical':
        values = dict({
            'Time':[event[0] for event in sequences[cur_seq_ID]],
            'Value':[float(event[attr_index-1]) for event in sequences[cur_seq_ID]]
            })
        fig = px.scatter(values,x='Time',y='Value')
        id = str(attr_index)
        id_list.append(id)
        return [dcc.Graph(figure=fig,id=id)]

app.layout = html.Div(children = [
                    html.Div(children=[
                        html.Div(children=[
                            html.P("Choose the sequence"),
                            dcc.Dropdown(id='sequence_chooser_dd',clearable=False)],
                            className="sequence_chooser"),
                        html.Button("Load data from MAQUI current context",id="context-fetch",n_clicks=0),
                        dcc.ConfirmDialog(message='The MAQUI context is not up to date, continue ?',id="MAQUI-uptodate"),
                        html.Button("Load data from disk",id='disk-data-fetch',n_clicks=0),
                        html.Button("Add an event to the sequence",id='add-event',n_clicks=0),
                        dcc.Store(id='sequences_data'),
                        dcc.Store(id='attributes_data'),
                        html.Button("Save data to disk",id='data-save')],
                        className='horizontal_box'),
                    html.Div(id='events-disp')]
                    )


##Callbacks
@app.callback(
    Output('events-disp','children'),
    Input('sequence_chooser_dd','value'),
    Input("attributes_data","data"),
    State("sequences_data",'data'),
    State({'type':'type_chooser_radio','place':ALL},'value'))
def update_current_sequence_display(cur_seq_ID,attributes_data,sequences,types):
    children = []
    for attr_index in range(2,len(attributes_data["names"])):   #not displaying time, ID
        line = []
        #title part
        attr_name = html.P(attributes_data['names'][attr_index],className='attr_title')
        line.append(attr_name)
        #picking categorical/numerical display
        options = ['Categorical','Numerical']
        radio_id = {'place':attr_index,'type':'type_chooser_radio'}

        if not types: #list is empty
            value = 'Categorical'
        else:
            value = types[attr_index-2]   #types is already cut off at initialization
        type_chooser = dcc.RadioItems(options=options,value=value,id=radio_id,inline=False)
        line.append(type_chooser)

        line = line + dropdown_list_or_graph(attr_index,cur_seq_ID,attributes_data,sequences)
        children.append(html.Div(line,className='horizontal_box'))

    return children


    return children
@app.callback(
            Output('sequence_chooser_dd','options'),
            Output('sequence_chooser_dd','value'),
            State('sequence_chooser_dd','options'),
            Input('sequences_data','data'),
            Input('context-fetch','n_clicks'),
            Input('disk-data-fetch','n_clicks'),
            )
def display_sequence_chooser(previous_options,sequences,_,r):
    call_id = callback_context.triggered[0]['prop_id']
    if sequences==None or call_id=='.':   #innit call, no access to data yet, read them directly from disk
        with open("./events.csv",'r') as f:
            r = csv.reader(f,csv.QUOTE_NONNUMERIC)
            r.__next__()  #title row
            first_row = r.__next__()
            options = [first_row[0]]
            value = options[0]
            f.close()
    elif call_id=='context-fetch.n_clicks' or call_id=='disk-data-fetch.n_clicks':
        options = list(sequences.keys())
        value = options[0]
    elif call_id=='sequences_data.data':
        if previous_options != list(sequences.keys()):
            options = list(sequences.keys())
            value = options[0]
        else:
            raise PreventUpdate

    return options, value

@app.callback(
    Output('sequences_data','data'),
    Output('attributes_data','data'),
    Input('context-fetch','n_clicks'),
    Input('disk-data-fetch','n_clicks'),
    Input({'type':'attr_value_dd', 'place': ALL}, 'value'),
    Input({'type':'type_chooser_radio','place':ALL},'value'),
    Input('add-event','n_clicks'),
    State('sequence_chooser_dd','value'),
    State('sequences_data','data'),
    State('attributes_data','data'),
    )
def update_data(a,b,c,d,e,seq_id,sequences,attr_data):  #alphabet letters are placeholder for dash auto-generated JS or smth
    calling = callback_context.triggered[0]
    if calling['prop_id']=='.' or calling['prop_id']=='disk-data-fetch.n_clicks':  #if calling is empty, ie init call
        return load_data_from_disk()

    elif calling['prop_id']=='context-fetch.n_clicks':
        return load_data_from_MAQUI()

    elif calling['prop_id']=='add-event.n_clicks':
        sequences[seq_id].append(sequences[seq_id][-1].copy())
        return sequences,attr_data

    else: #input id is a dictionnary
        call_id = loads(calling['prop_id'][:-6])       #removing the ".value" part of the string, idk why it's there
        call_value = calling['value']
        if call_id['type']=='type_chooser_radio':
            attr_index = call_id['place']
            attr_name = attr_data['names'][attr_index]
            attr_data['types'][attr_name] = call_value
            return sequences,attr_data

        elif call_id['type']=='attr_value_dd':
            place = call_id['place']
            attr_id, event_id = map(int,place.split('c'))
            sequences[seq_id][event_id][attr_id-1] = call_value   #events have no id hence -1
            return sequences,attr_data

        else:
            print('error updating data')
            print('unexpected callback Input :',calling['prop_id'])
            raise PreventUpdate

@app.callback(
    Output('data-save','n_clicks'),
    Input('data-save','n_clicks'),
    State('sequences_data','data'),
    State('attributes_data','data'),
    prevent_initial_call=True)
def save_data(n,sequences,attributes):
    with open("./events_saved.csv",'w',newline='') as f:
        w = csv.writer(f,csv.QUOTE_NONNUMERIC)
        w.writerow(attributes['names'])
        for seq_id,sequence in sequences.items():
            for event in sequence:
                w.writerow([seq_id]+event)
        f.close()
    return n


####Run
if __name__=='__main__':
    app.run_server(debug=True)