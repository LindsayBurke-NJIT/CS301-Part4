import pandas as pd
from dash import Dash, dcc, html
from dash.dependencies import Input, Output, State
import plotly.express as px
from dataFuncs import *

app = Dash(__name__)
server = app.server

app.layout = html.Div(children=[
    dcc.Upload(
        id='upload-data',
        children=html.Div([
            html.A('Upload File')
        ]),
        multiple=False,
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'textAlign': 'center',
            'marginBottom': '10px',
            'background-color': 'silver'
        }
    ),
    html.Div(children=[
        html.A('Select Target: ', style={
            'display': 'inline-block',
            'textAlign': 'center',
            'paddingLeft': '500px'
        }),
        dcc.Dropdown(
                id='dropdown-target',
                options=[],
                value=None,
                style={
                        'padding': '10px',
                        'width': '200px',
                        'display': 'inline-block',
                        'marginLeft': '10px'
                }
        )
    ],
    style={
        'display': 'flex',
        'alignItems': 'center',
        'backgroundColor': 'lightGray'
    }),
    html.Div([
        dcc.RadioItems(
            options=[],
            value=None,
            id='radio-choice',
            inline=True
        ,style={
            'width': '50%',
            'display': 'block'
        }
        )
    ]),
    html.Div(children=[
        dcc.Graph(id="bar1", style={"width": "50%", 'display': 'inline-block'}),
        dcc.Graph(id="bar2", style={"width": "50%", 'display': 'inline-block'})
    ]
    ),
    dcc.Checklist(
        options=[],
        value=None,
        id='checklist-regr',
        inline=True
    ),
    html.Button(
        'Train', 
        id='start-train',
        style={
            "width": "20%",
            "margin": "auto",
            "marginTop": "20px",
            "display": "block"
        },
        n_clicks=0
    ),
    html.Div(
        id="r2-output",
        style={
            "display": "block",
            "margin": "auto",
            "textAlign":"center",
            "padding": "20px"
        },
        children=[]
    ),
    html.Div(children=[
        dcc.Input(
            id='inputPrediction',
            placeholder='',
            type='text',
            style={
                "display": "inline-block",
                "marginLeft": "25vw",
                "marginRight": "0px",
                "width": "50%"
            },
            value = ""
        ),
        html.Button(
            "Predict",
            id="start-predict",
            style={
                "display": "inline-block",
                "margin": "0px"
            },
            n_clicks=0
        )
    ], style={
        'display': 'flex',
        "alignItems": "center"
    }),
    html.Div(
        id="predict-output",
        style={
            "display": "block",
            "margin": "auto",
            "textAlign":"center",
            "paddingTop": "20px"
        },
        children=[]
    )
])

@app.callback(
    [Output('dropdown-target', 'options'), Output('dropdown-target', 'value')],
    [Input('upload-data', 'filename'), Input('upload-data', 'contents')]
)
def setDropdownOpts(filename, contents):
    df = parseDf(filename, contents)

    df = df.select_dtypes(include='number')
    options = [{'label': col, 'value': col} for col in df.columns]
    value = df.columns.unique()[0]
    return options, value

@app.callback(
    [Output('radio-choice', 'options'), Output('radio-choice', 'value')],
    [Input('upload-data', 'filename'), Input('upload-data', 'contents'), Input('dropdown-target', 'value')]
)
def setRadioOpts(filename, contents, target):
    df = parseDf(filename, contents)
    df = preprocess(df)

    df = df.drop(columns=target)
    df = df.select_dtypes(exclude='number')
    options = [{'label': col, 'value': col} for col in df.columns]
    value = df.columns.unique()[0]
    return options, value

@app.callback(
    Output('bar1', 'figure'),
    [Input('upload-data', 'contents'), Input('upload-data', 'filename'), Input('dropdown-target', 'value'), Input('radio-choice', 'value')]
)
def updateBar1(contents, filename, target, choice):
    df = parseDf(filename, contents)
    df = preprocess(df)

    res = df.groupby(choice)[target].mean().reset_index()

    fig = px.bar(res, x=choice, y=target, title=f"Average {target} by {choice}", text=target)
    fig.update_layout(yaxis_title=f"{target} (average)")
    return fig

@app.callback(
    Output('bar2', 'figure'),
    [Input('upload-data', 'contents'), Input('upload-data', 'filename'), Input('dropdown-target', 'value')]
)
def updateBar2(contents, filename, target):
    df = parseDf(filename, contents)
    df = preprocess(df)
    numCols = df.select_dtypes(include='number')

    res = numCols.corr()[target].abs()
    res = res.drop(target)

    fig = px.bar(res, x=res.index, y=res.values, title=f"Correlation Strength of Numerical Variables with {target}", text=target)
    fig.update_layout(xaxis_title="Numerical Variables", yaxis_title=f"Correlation Strength (absolute value)")
    return fig

@app.callback(
    [Output('checklist-regr', 'options'), Output('checklist-regr', 'value')],
    [Input('upload-data', 'contents'), Input('upload-data', 'filename'), Input('dropdown-target', 'value')]
)
def setCheckListRegr(contents, filename, targetVar):
    df = parseDf(filename, contents)
    df = preprocess(df)
    df = df.drop(columns=targetVar)

    return df.columns, df.columns

@app.callback(
    [Output('r2-output', 'children'), Output('inputPrediction', 'placeholder')],
    [Input('start-train', 'n_clicks')],
    [State('upload-data', 'contents'), State('upload-data', 'filename'), State('checklist-regr', 'value'), State('dropdown-target', 'value')]
)
def setR2(n_clicks, contents, filename, checklistVal, target):
    if(n_clicks==0):
        return [""]
    df = parseDf(filename, contents)
    r2 = gradBoostRegr(df, target, checklistVal)

    placeholderStr = ""
    for col in checklistVal:
        placeholderStr+= col+","
    placeholderStr = placeholderStr[:-1]
    return [f"The R2 score is: {r2}"], placeholderStr

@app.callback(
    [Output('predict-output', 'children')],
    [Input('start-predict', 'n_clicks')],
    [State('upload-data', 'contents'), State('upload-data', 'filename'), State('inputPrediction', 'value'), State('dropdown-target', 'value'),State('checklist-regr', 'value')]
)
def setPredict(n_clicks, contents, fname, predictVals, target, checklistVals):
    if(n_clicks==0):
        return [""]

    df = parseDf(fname, contents)
    df = preprocess(df)
    if(predictVals==""):
        return ["No values entered."]
    else:
        temp = predictVals.replace(",", "")
        if(not (temp.replace(" ", "")).isalnum()):
            return ["Incorrect character entered in field. Fields must be alphanumeric. Separate fields by \",\""]
        predictVals = predictVals.split(",")
        if not (len(predictVals) == len(checklistVals)):
            return [f"Wrong number of values entered. Must be one parameter for each checked off variable. You provided {len(predictVals)}, when {len(checklistVals)} were needed."]
        for idx, col in enumerate(checklistVals):
            if(pd.api.types.is_numeric_dtype(df[col])):
                if(predictVals[idx].isdigit() == False):
                    return [f"Incorrect value entered for {col}. You entered \"{predictVals[idx]}\". Must be a digit."]
                else:
                    predictVals[idx] = float(predictVals[idx])
            else:
                if(predictVals[idx] not in df[col].unique()):
                    return [f"Incorrect value entered for {col}. You entered \"{predictVals[idx]}\". Must be one of {df[col].unique()}"]
        predArr = []
        predArr.append(predictVals)
        predictData = pd.DataFrame(predArr, columns=checklistVals)
    pipe, _, _ = getPipeline(df[checklistVals], df[target])
    y_pred = pipe.predict(predictData)
    return [f"Predicted {target} is: {y_pred[0]}"]

if __name__ == '__main__':
    app.run_server(debug=True)