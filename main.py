import pandas as pd
from dash import Dash, dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
from dataFuncs import *

url = "https://drive.google.com/file/d/1GiZL0hZa9pVuYSMnwJVWrtas_ESnNXif/view?usp=sharing"
path = 'https://drive.google.com/uc?export=download&id='+url.split('/')[-2]
df = pd.read_csv(path)

app = Dash(__name__)

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
            'backgroundColor': 'silver',
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
        'backgroundColor': 'silver'
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
    res = df.corr()[target]

    fig = px.bar(df, x=df.columns, y=res, title=f"Correlation Strength of Numerical Variables with {target}", text=target)
    fig.update_layout(yaxis_title=f"{target} (average)")
    return fig

# @app.callback(
#     Output('regr', 'figure'),
#     [Input('upload-data', 'contents'), Input('upload-data', 'filename'), Input('dropdown-target', 'value')]
# )
# def regression(filename, contents, targetVar):
#     df = parseDf(filename, contents)
#     gradBoostRegr(df, targetVar)
    

if __name__ == '__main__':
    app.run_server(debug=True)