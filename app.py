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
import numpy as np
import csv
import requests
from json import loads
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import pickle
from datetime import datetime

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
        attr_types[attr] = 'categorical'

    attributes_data = {'names':attributes,'values':attr_values,'types':attr_types}
    return sequences,attributes_data

def load_data_from_MAQUI():
    try:
        context_rep = requests.get('http://localhost:5000/getCurrentContext')
        context = context_rep.json()
        if not context["UpToDate"]:
            print('no context selected on MAQUI')
            raise PreventUpdate

        context_rep.raise_for_status()

        params = {"panelID":context["Pannel"],'ForSID':context['Focus'],'outputType':context['Output'],'write':False}
        data_rep = requests.get('http://localhost:5000/saveLocalSequences',params=params)
        data_rep.raise_for_status()

        sequences_maqui = data_rep.json()
        sequences = {}
        for seq_key,sequence_maq in sequences_maqui.items():
            sequence = []
            for event_maqui in sequence_maq:
                for x in event_maqui[1:]:
                    time = event_maqui[0]
                    event = [time]+x
                    sequence.append(event)
            sequences[seq_key]=sequence

        attributes_rep = requests.get('http://localhost:5000/getAttributeList/event')
        attributes_rep.raise_for_status()

        attributes_maqui = attributes_rep.json()
        attributes = {'names':['id','time'],'values':{},'types':{}}
        for attr in attributes_maqui:
            attr_name = attr['attributeName']
            attributes['names'].append(attr_name)
            attributes['types'][attr_name] = attr['numericalOrCategorical']

        for id,name in enumerate(attributes['names']):
            attributes['values'][name] = []

            for sequence in sequences.values():
                for event in sequence:
                    attributes['values'][name].append(event[id-1])  #events don't carry the id attribute
            attributes['values'][name] = list(set(attributes['values'][name]))
        return sequences,attributes
    except requests.exceptions.RequestException as e:
        print(e)
        raise PreventUpdate

def split_time(sequences,attributes):
    attributes['values']['date'] = []
    attributes['values']['hour'] = []
    for time in attributes['values']['time']:
        date,hour = time.split(' ',1)
        attributes['values']['date'].append(date)
        attributes['values']['hour'].append(hour)
    new_keys = [list(attributes['values'].keys())[0],'date','hour'] + list(attributes['values'].keys())[2:-2]
    new_attr_values = dict((key,attributes['values'][key]) for key in new_keys)
    attributes['values'] = new_attr_values

    attributes['names'] = new_keys

    attributes['types']['date'] = 'date'
    attributes['types']['hour'] = 'hour'

    for sequence in sequences.values():
        for event_id,event in enumerate(sequence):
            date,hour = event[0].split(' ',1)
            event = [date,hour] + event[1:]
            sequence[event_id] = event
    return sequences,attributes

##Display
external_stylesheets = [{"rel":"stylesheet"}]
app = Dash(__name__,external_stylesheets=external_stylesheets)

def dropdown_list_or_graph(attr_index,cur_seq_ID,attr_data,sequences):
    attr_name = attr_data["names"][attr_index]
    if attr_data["types"][attr_name]=='categorical':
        DDList = []
        for event_id, event in enumerate(sequences[cur_seq_ID]):
            options = attr_data['values'][attr_name]
            value = event[attr_index-1]    # -1 cuz events don't have an "ID" field
            dd_id = {'place':str(attr_index) + 'c' + str(event_id),'type':'attr_value_dd'}
            DDList.append(dcc.Dropdown(options=options,value=value,className="categ_event_dd",id=dd_id,clearable=False))
        return DDList

    if attr_data["types"][attr_name]=='numerical':
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

    if attr_data["types"][attr_name]=='masked':
        return [html.Div(children=[])]

    if attr_data["types"][attr_name]=='date':
        datelist = []
        for event_id, event in enumerate(sequences[cur_seq_ID]):
            date = event[attr_index-1]    # -1 cuz events don't have an "ID" field
            date_id = {'place':str(attr_index) + 'c' + str(event_id),'type':'date_picker'}
            datelist.append(dcc.DatePickerSingle(date,id=date_id))
        return datelist

    if attr_data['types'][attr_name]=='hour':
        hourlist = []
        hourinputlist = []
        for event_id,event in enumerate(sequences[cur_seq_ID]):
            hour_id = {'place':str(attr_index)+'c'+str(event_id),'type':'hour_input'}
            hourlist.append(dcc.Input(value=event[attr_index-1]))
        return hourlist


    else:
        print("error : unrecognized datatype "+attr_data["types"][attr_name]+ " in dropdown_list_or_graph")
        return [html.Div(children=[])]

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
                        dcc.Input(placeholder="ID of event to search",debounce=True,id="event-search-id"),
                        html.Button("Find opposite sequence",id='find-opposite-sequence'),
                        dcc.Input(placeholder="ID of event to remove",debounce=True,id="remove-event"),
                        dcc.Store(id='sequences_data'),
                        dcc.Store(id='attributes_data'),
                        dcc.Store(id='prediction_model'),
                        dcc.Store(id='prediction_data'),
                        dcc.Dropdown(value='sklearn.neighbors.KNeighborsClassifier',
                                     options=[
                                            {'label':'knn','value':'KNeighborsClassifier'},
                                            {'label':'SVN','value':'SVC'},
                                            {'label':'random forest','value':'RandomForestClassifier'}],
                                     id='model_type'),
                        html.Button('Train model on current data',id='train-model',n_clicks=0),
                        html.Button("Save data to disk",id='data-save')],
                        className='horizontal_box'),
                    html.Div(id='events-disp'),
                    html.Div(id='prediction-results'),
                    html.Div(id='opposite-event',className='horizontal_box'),
                    html.Div(id='opposite-sequence',className='horizontal_box')]
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
    for attr_index in range(1,len(attributes_data["names"])):   #not displaying time, ID
        line = []
        #title part
        attr_name = html.P(attributes_data['names'][attr_index],className='attr_title')
        line.append(attr_name)
        #picking categorical/numerical display
        options = ['categorical','numerical','masked']
        radio_id = {'place':attr_index,'type':'type_chooser_radio'}

        if not types: #list is empty
            value = 'categorical'
        else:
            value = types[attr_index-2]   #types is already cut off at initialization
        type_chooser = dcc.RadioItems(options=options,value=value,id=radio_id,inline=False)
        line.append(type_chooser)
        line = line + dropdown_list_or_graph(attr_index,cur_seq_ID,attributes_data,sequences)
        children.append(html.Div(line,className='horizontal_box'))
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
def update_sequences_and_attributes_data(a,b,c,d,e,f,g,seq_id,sequences,attr_data):  #alphabet letters are placeholders for dash auto-generated JS or smth
    calling = callback_context.triggered[0]
    if calling['prop_id']=='.' or calling['prop_id']=='disk-data-fetch.n_clicks':  #if calling is empty, ie init call
        sequences,attributes = load_data_from_disk()
        return split_time(sequences,attributes)

    elif calling['prop_id']=='context-fetch.n_clicks':
        sequences,attributes = load_data_from_MAQUI()
        return split_time(sequences,attributes)

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

        elif call_id['type']=='attr_value_dd' or call_id['type']=='graph_input' or call_id['type']=='date_picker' or call_id['type']=='hour_input':
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
def save_data(n,sequences,attributes):   #todo update to the new data formatting
    with open("./events_saved.csv",'w',newline='') as f:
        w = csv.writer(f,csv.QUOTE_NONNUMERIC)
        w.writerow(attributes['names'])
        for seq_id,sequence in sequences.items():
            for event in sequence:
                w.writerow([seq_id]+event)
        f.close()
    return n



@app.callback(
    Output('opposite-event','children'),
    Input('event-search-id','value'),
    State('attributes_data','data'),
    State('sequences_data','data'),
    State('sequence_chooser_dd','value'),
    State('prediction_model','data'),
    prevent_initial_call=True)
def display_closest_opposite_event(event_id,attributes,sequences,cur_seq_id,model):
    if int(event_id)<len(sequences[cur_seq_id]):
        cur_event = sequences[cur_seq_id][int(event_id)]
    else:
        print("selected out of range event ID")
        raise preventUpdate
    op_event,op_seq_id,op_ev_id,dist = find_closest_opposite_event(cur_event,sequences,attributes["types"],attributes["values"],model)
    if op_ev_id!=-1:
        intro = html.P("The closest opposite event is the "+str(op_ev_id)+"th event of the sequence "+str(op_seq_id)+" which is at a distance of "+str(dist))
        event_disp = []
        for attr_id,attr_val in enumerate(op_event):
            event_disp.append(html.Div([html.P(attributes["names"][attr_id+1]),html.P(str(attr_val))]))
        event_disp = html.Div(children=event_disp,className='horizontal_box')

        return html.Div(children=[intro,event_disp],className='vertical_box')

    else:
        return html.P('No event with opposite target in the current data')

@app.callback(
    Output('opposite-sequence','children'),
    Input('find-opposite-sequence','n_clicks'),
    State('sequences_data','data'),
    State('sequence_chooser_dd','value'),
    State('attributes_data','data'),
    State('prediction_model','data'),
    prevent_initial_call=True)
def display_closest_opposite_sequence(_,sequences,cur_seq_ID,attributes,model):
    closest_seq,closest_seq_id,min_dist = find_closest_opposite_sequence(sequences[cur_seq_ID],sequences,attributes,model,type_dist='edit')

    seq_disp = []
    for event in closest_seq:
        ev_disp = []
        for attr in event:
            ev_disp.append(html.P(attr))
        ev_disp = html.Div(ev_disp)
        seq_disp.append(ev_disp)
    seq_disp = html.Div(seq_disp,style={'display':'grid'})

    return seq_disp

@app.callback(
    Output('prediction_data','data'),
    Input('train-model','n_clicks'),
    Input('sequences_data','data'),
    Input('sequence_chooser_dd','value'),
    State('attributes_data','data'),
    State('prediction_model','data'))
def update_predictions(_,sequences,cur_seq_ID,attributes,model):
    if model==None:
        print("no model has been trained yet")
        return []

    model = bytes(model,'ISO-8859-1')
    model = pickle.loads(model)
    scores = []
    for event in sequences[cur_seq_ID]:
        translated_event = []
        for attr_id,attr_value in enumerate(event[:-1]):
            attr_name = attributes['names'][attr_id]
            attr_type = attributes['types'][attr_name]
            if attr_type=='numerical':
                translated_event.append(float(attr_value))
            else:  #todo cases for date,hour
                translated_event.append(string_to_number(attr_value))

        score = str(model.predict([translated_event]))
        scores.append(score)
    return scores

@app.callback(
        Output("prediction-results",'children'),
        Input('prediction_data','data'))
def display_predictions(scores):
    scores = [html.P(score) for score in scores]
    scores = html.Div(scores,className='horizontal_box')

    header = html.H6('ML predictions on whether each event is a fraud')

    return html.Div([header,scores],className = 'vertical_box')


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



def distance_seq(seq1,seq2,attributes,type_dist='edit'):
    if type_dist=='edit':
        r = np.zeros((len(seq1),len(seq2)))

        for i in range(len(seq1)):
            op_cost = 1
            slide_const = 1

            for j in range(len(seq2)):
                if i==0 and j==0:
                    r[i,j] = 0
                elif j==0:
                    r[i,j] = r[i-1,j] + op_cost
                elif i==0:
                    r[i,j] = r[i,j-1] + op_cost
                else:
                    for a_id in range(1,len(seq1[i])):
                        if seq1[i][a_id]!=seq2[j][a_id]:
                            kij = 2*op_cost
                            break
                        time1 = datetime.strptime(seq1[i][0]+" "+seq1[i][1],'%Y-%m-%d %H:%M:%S')
                        time2 = datetime.strptime(seq1[j][0]+' '+seq1[j][1],'%Y-%d-%m %H:%M:%S')
                        kij = slide_const *abs((time1-time2).total_seconds())

                    r[i,j] = min(
                                r[i-1,j]+op_cost,
                                r[i,j-1]+op_cost,
                                r[i-1,j-1]+kij)
        return(r[-1,-1])

    elif type_dist=='M&M':
        pass #TODO

def sequence_predict(sequence,model):
    for event in sequence:
        event = [string_to_number(k) for k in event[:-1]]
        if model.predict([event])[0]>0:
            return 1
    return 0

def find_closest_opposite_sequence(seq_origin,sequences,attributes,model,type_dist='edit'):
    if model==None:
        print("no model has been trained yet")
        return seq_origin,'-1',-1

    model = bytes(model,'ISO-8859-1')
    model = pickle.loads(model)

    min_dist = -1

    pred_seq_origin = sequence_predict(seq_origin,model)
    for seq_id,seq in sequences.items():
        if sequence_predict(seq,model)!=pred_seq_origin:
            dist = distance_seq(seq_origin,seq,attributes,type_dist)
            if dist<=min_dist or min_dist==-1:
                closest_seq = seq
                closest_seq_id = seq_id
                min_dist = dist
    if min_dist ==-1:
        print("Could not find sequence of opposite predict tag")
        return seq_origin,'-1',-1
    return  closest_seq,closest_seq_id,min_dist

def distance_event(event1,event2,attr_types,attr_values):   #TODO add log on amt, divide by ecart-type instead of max, frequence ?
    dist = 0
    for attr_id,attr_type in enumerate(attr_types.values()):
        if attr_type == 'numerical':
            val1 = float(event1[attr_id-1])
            val2 = float(event2[attr_id-1])
            dist += abs((val1-val2)/max(map(float,list(attr_values.values())[attr_id])))
        if attr_type == 'categorical':
            if event1[attr_id-1]!=event2[attr_id-1]:
                dist += 1
            else:
                dist += 0
    return dist/len(attr_types.keys())

def find_closest_opposite_event(event,sequences,attr_types,attr_values,model):
    if model==None:
        print("no model has been trained yet")
        return event,'-1',-1,-1

    model = bytes(model,'ISO-8859-1')
    model = pickle.loads(model)

    min_dist = -1
    for cur_seq_id, seq in sequences.items():
        for cur_event_id,cur_event in enumerate(seq):
            event_pred = [string_to_number(k) for k in event[:-1]]
            cur_event_pred = [string_to_number(k) for k in cur_event[:-1]]
            if model.predict([cur_event_pred])!=model.predict([event_pred]):
                dist = distance_event(cur_event,event,attr_types,attr_values)
                if dist <= min_dist or min_dist==-1:
                    closest_opposite_event = cur_event
                    min_dist = dist
                    min_seq_id = cur_seq_id
                    min_event_id = cur_event_id
    if min_dist == -1:
        print("Could not find an event with opposite target")
        return event,'-1',-1,-1
    return closest_opposite_event, min_seq_id, min_event_id, min_dist

####Run
if __name__=='__main__':
    app.run_server(debug=True)