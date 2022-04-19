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
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import pickle

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
        context_rep = requests.get('http://localhost:5000/getCurrentContext')
        context = context_rep.json()
        if not context["UpToDate"]:
            raise PreventUpdate
            print('no context selected on MAQUI')

        context_rep.raise_for_status()

        params = {"panelID":context["Pannel"],'ForSID':context['Focus'],'outputType':context['Output'],'write':False}
        data_rep = requests.get('http://localhost:5000/saveLocalSequences',params=params)
        data_rep.raise_for_status()

        sequences = data_rep.json()

        attributes_rep = requests.get('http://localhost:5000/getAttributeList/all')
        attributes_rep.raise_for_status()

        attributes_maqui = attributes_rep.json()

        attributes = {'names':[],'values':{},'types':{}}
        for attr in attributes_maqui:
            attr_name = attr['attributeName']
            attributes['names'].append(attr_name)
            attributes['types'][attr_name] = attr['numericalOrCategorical']
            attributes['types'][attr_name][0].upper()
        #TODO finish here lol

        print(attributes)
        raise PreventUpdate
    except requests.exceptions.RequestException as e:
        print(e)
        raise PreventUpdate

##Display
external_stylesheets = [{"rel":"stylesheet"}]
app = Dash(__name__,external_stylesheets=external_stylesheets)

def dropdown_list_or_graph(attr_index,cur_seq_ID,attr_data,sequences):
    attr_name = attr_data["names"][attr_index]
    if attr_data["types"][attr_name]=='Categorical':
        DDList = []
        for event_id, event in enumerate(sequences[cur_seq_ID]):
            options = attr_data['values'][attr_name]
            value = event[attr_index-1]    # -1 cuz events don't have an "ID" field
            dd_id = {'place':str(attr_index) + 'c' + str(event_id),'type':'attr_value_dd'}
            DDList.append(dcc.Dropdown(options=options,value=value,className="categ_event_dd",id=dd_id,clearable=False))
        return DDList
    if attr_data["types"][attr_name]=='Numerical':
        values = dict({
            'Time':[event[0] for event in sequences[cur_seq_ID]],
            'Value':[float(event[attr_index-1]) for event in sequences[cur_seq_ID]]
            })
        fig = px.scatter(values,x='Time',y='Value')
        id = str(attr_index)
        graph = dcc.Graph(figure=fig,id=id)

        inputList = []
        for event_id,event in enumerate(sequences[cur_seq_ID]):
            input_id = {'type':'graph_input','place':str(attr_index)+'c'+str(event_id)}
            input_value = event[attr_index-1]

            inputList.append(dcc.Input(id=input_id,value=input_value,debounce=True,n_submit=0))

        return [html.Div([graph,html.Div(inputList)],className='vertical_box')]

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
                        dcc.Input(placeholder="ID of event to remove",debounce=True,id="remove-event"),
                        dcc.Store(id='sequences_data'),
                        dcc.Store(id='attributes_data'),
                        dcc.Store(id='prediction_model'),
                        dcc.Dropdown(value='sklearn.neighbors.KNeighborsClassifier',
                                     options=[
                                            {'label':'knn','value':'KNeighborsClassifier'},
                                            {'label':'SVN','value':'SVC'},
                                            {'label':'random forest','value':'RandomForestClassifier'}],
                                     id='model_type'),
                        html.Button('Train model on current data',id='train-model',n_clicks=0),
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
    State({'type':'type_chooser_radio','place':ALL},'value'),
    State('prediction_model','data'))
def update_current_sequence_display(cur_seq_ID,attributes_data,sequences,types,model):
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

    if model==None:
        print("no model has been trained yet")
    else:
        model = bytes(model,'ISO-8859-1')
        model = pickle.loads(model)
        scores = []
        for event in sequences[cur_seq_ID]:
            translated_event = [string_to_number(k) for k in event[:-1]]
            score = str(model.predict([translated_event]))
            scores.append(html.P(score))
        children.append(html.Div(scores,className='horizontal_box'))
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
    Input({'type':'graph_input','place':ALL},'value'),
    Input('add-event','n_clicks'),
    Input('remove-event','value'),
    State('sequence_chooser_dd','value'),
    State('sequences_data','data'),
    State('attributes_data','data'),
    )
def update_data(a,b,c,d,e,f,g,seq_id,sequences,attr_data):  #alphabet letters are placeholder for dash auto-generated JS or smth
    calling = callback_context.triggered[0]
    if calling['prop_id']=='.' or calling['prop_id']=='disk-data-fetch.n_clicks':  #if calling is empty, ie init call
        return load_data_from_disk()

    elif calling['prop_id']=='context-fetch.n_clicks':
        return load_data_from_MAQUI()

    elif calling['prop_id']=='add-event.n_clicks':
        sequences[seq_id].append(sequences[seq_id][-1].copy())
        return sequences,attr_data

    elif calling['prop_id']=='remove-event.value':
        sequences[seq_id].pop(int(calling['value']))
        return sequences,attr_data

    else: #input id is a dictionnary
        end = calling['prop_id'].index('}.')
        call_id = loads(calling['prop_id'][:end+1])       #removing the ".value" part of the string, idk why it's there
        call_value = calling['value']
        if call_id['type']=='type_chooser_radio':
            attr_index = call_id['place']
            attr_name = attr_data['names'][attr_index]
            attr_data['types'][attr_name] = call_value
            return sequences,attr_data

        elif call_id['type']=='attr_value_dd' or call_id['type']=='graph_input':
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

## Prediction model
str_to_nb = {}
max_str_to_nb = 0
def string_to_number(str):
    global str_to_nb,max_str_to_nb
    if str in str_to_nb.keys():
        return str_to_nb[str]
    else:
        max_str_to_nb += 1
        str_to_nb[str] = max_str_to_nb
        return max_str_to_nb

@app.callback(
    Output('prediction_model','data'),
    Input('train-model','n_clicks'),
    State('sequences_data','data'),
    State('model_type','value'))
def train_model(n,sequences,model_type):
    model = eval(model_type)()
    X = []
    y = []
    if sequences == None:
        raise PreventUpdate
    else:
        for seq in sequences.values():
            for event in seq:
                X.append(event[:-1])
                y.append(int(event[-1]))

        for event_id in range(len(X)):
            X[event_id] = [string_to_number(k) for k in X[event_id]]
        y = [int(k) for k in y]
        model.fit(X=X,y=y)

        return pickle.dumps(model).decode('ISO-8859-1')

####Run
if __name__=='__main__':
    app.run_server(debug=True)