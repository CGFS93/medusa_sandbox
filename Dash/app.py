# Project is using python 3.7
import os
import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objs as go
import joblib
import pandas as pd
import yfinance as yf
import requests

import warnings
warnings.filterwarnings('ignore')

# ==============================================================================
# Data Preparation
# ==============================================================================

ticker_symbol = 'ETH-USD'
asset = yf.Ticker(ticker_symbol)
df = asset.history(period='max')

# Convert Date column into datetime data type
df['Date'] = df.index.strftime('%m/%d/%Y')
df['Year'] = df.index.year


# Sort columns
df = df[['Open', 'High', 'Low', 'Close', 'Volume', 'Year']].sort_index(ascending=False)

# Group by Year and calculate the high, low, and range for each year
price_df = df.groupby('Year')['Close'].agg(['max', 'min']).reset_index()
price_df['range'] = price_df['max'] - price_df['min']

# Rename columns
price_df.columns = ['Year', 'High', 'Low', 'Range']

# Melt the DataFrame
price_df_melted = price_df.melt(id_vars=['Year'], value_vars=['High', 'Low'],
                                var_name='Price Type', value_name='Price')

# ==============================================================================
# Load the Random Forest Classifier
rf_model = joblib.load('rf_model.pkl')
rfr_model = joblib.load('rfr_price_model.pkl')

def calculate_sma(data, window):
    return data.rolling(window=window).mean()

def calculate_rsi(data, timeperiod):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    avg_gain = gain.rolling(window=timeperiod, min_periods=1).mean()
    avg_loss = loss.rolling(window=timeperiod, min_periods=1).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def count_unique(values):
    unique_counts = {}
    for value in values:
        unique_counts[value] = unique_counts.get(value, 0) + 1
    return unique_counts

def Model(asset, table_data=False):
    periods = ['1mo', '3mo', '6mo', '1y', '2y', '5y', 'ytd', 'max']
    all_prob = []
    all_pred = []

    for i in periods:
        apmax = yf.Ticker(asset).history(period='max')
        apmax = apmax['Close']

        coin = yf.Ticker(asset).history(period=i)
        coin.index = coin.index.strftime('%d/%m/%Y')
        coin.drop(['Dividends', 'Stock Splits'], axis=1, inplace=True)
        coin['SMA'] = calculate_sma(coin['Close'], window=5)
        coin['RSI'] = calculate_rsi(coin['Close'], timeperiod=14)
        coin.dropna(inplace=True)

        priced = yf.Ticker(asset).history(period=i)
        priced.index = priced.index.strftime('%d/%m/%Y')
        priced['Target'] = (priced['Close'] > priced['Open']).astype(int)
        priced['SMA'] = calculate_sma(priced['Close'], window=5)
        priced['RSI'] = calculate_rsi(coin['Close'], timeperiod=14)
        priced = priced[['Open', 'High', 'Low', 'Volume', 'SMA', 'RSI', 'Target']]
        priced.dropna(inplace=True)

        prob = rf_model.predict(coin)
        pred = rfr_model.predict(priced)

        all_prob.append(list(prob))
        all_pred.append(list(pred))

    combined_prob = [item for sublist in all_prob for item in sublist]
    combined_pred = [item for sublist in all_pred for item in sublist]

    unique_counts = count_unique(combined_prob)

    label_mapping = {0: "Sell", 1: "Buy"}
    total_samples = len(combined_prob)
    percentage_counts = {label_mapping[value]: count / total_samples * 100 for value, count in unique_counts.items()}

    df = pd.DataFrame(list(percentage_counts.items()), columns=['Label', 'Percentage'])
    df['Percentage'] = df['Percentage'].round(2).astype(float)

    mean_price_5y = round(sum(apmax) / len(apmax))
    mean_price_252days = round(sum(apmax[-252:]) / len(apmax[-252:]))
    actual_ytd_mean = round(sum(apmax[-35:]) / len(apmax[-35:]))
    current_price = round(apmax[-1])

    mean_price_prediction_5y = round(sum(combined_pred) / len(combined_pred))
    mean_price_prediction_252days = round(sum(combined_pred[-252:]) / len(combined_pred[-252:]))
    predicted_ytd_mean = round(sum(combined_pred[-35:]) / len(combined_pred[-35:]))
    current_price_prediction = round(combined_pred[-1])

    absolute_percentage_error = abs((actual_ytd_mean - predicted_ytd_mean) / actual_ytd_mean)
    mape = round(absolute_percentage_error * 100, 2)
    accuracy = (100 - mape)

    table = pd.DataFrame({
        'Current Price': [current_price],
        'Current Predicted Price': [current_price_prediction],
        'YTD Price Mean': [actual_ytd_mean],
        'YTD Predicted Price Mean': [predicted_ytd_mean],
        'YTD MAPE Actual Vs Predicted Price Mean': [mape],
        'YTD Accuracy Actual Vs Predicted Price Mean': [accuracy],
        '252 Day Mean': [mean_price_252days],
        '252 Day Predicted Mean': [mean_price_prediction_252days],
        '5 Year Price Mean': [mean_price_5y],
        '5 Year Predicted Price Mean': [mean_price_prediction_5y],
    })

    table = table.transpose().reset_index().rename(columns={0: 'Values', 'index': 'Description'})

    return table if table_data else df

probability = Model(ticker_symbol)
prediction = Model(ticker_symbol, table_data=True)


# ==============================================================================
# Function to fetch news articles from NewsData.io
def fetch_articles():
    api_key = 'pub_469391a49069800b98c40bcbac8bca47dec87'  # Replace with your NewsData.io API key
    url = 'https://newsdata.io/api/1/news'
    params = {
        'apikey': api_key,
        'q': 'ethereum',
        'language': 'en',
        'country': 'us',
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        return data.get('results', [])
    else:
        print(f'Error: {response.status_code}')
        return []

# Fetch the articles
articles = fetch_articles()

# Check if articles are fetched successfully
if not articles:
    articles = [{'title': 'No articles found', 'description': '', 'link': '#'}]

# ==============================================================================
# Navbar Configuration
# ==============================================================================
# Navigation bar with Logo
navbar = dbc.Navbar(
    dbc.Container([
        dbc.Row([
            dbc.Col(
                html.A(
                    html.Img(src='assets/logo.png', height="40px"),
                    href="#"
                ),
                width="auto"
            ),
            dbc.Col(
                dbc.NavbarToggler(id="navbar-toggler"),
                width="auto"
            ),
            dbc.Col(
                dbc.Collapse(
                    dbc.Nav(
                        [
                            dbc.NavItem(dbc.NavLink("Home", href="#", style={"color": "#333333"})),
                            dbc.NavItem(dbc.NavLink("About", href="#", style={"color": "#333333"})),
                            dbc.NavItem(dbc.NavLink("Services", href="#", style={"color": "#333333"})),
                            dbc.NavItem(dbc.NavLink("Contact", href="#", style={"color": "#333333"})),
                        ],
                        className="ml-auto",
                        navbar=True
                    ),
                    id="navbar-collapse",
                    navbar=True,
                ),
                width="auto"
            ),
        ], align="center", className="g-0 flex-nowrap"),
    ], fluid=True),
    color='#F1F3FD',
    dark=True,
)

# ==============================================================================
# Prepare figure plots
# ==============================================================================
# Line Chart
line_fig = px.line(
    df,
    x=df.index,
    y='Close',
)

line_fig.update_yaxes(
    tickprefix='$',
    title_text='Price (USD)'
)

line_fig.update_layout(
    title='Ethereum Price',
    xaxis=dict(showgrid=False),  # Remove vertical grid lines
    yaxis=dict(showgrid=True),    # Keep horizontal grid lines
    plot_bgcolor='#333333',
    paper_bgcolor='#333333',
    font=dict(color='white')
)

# Add gradient shading under the line
line_fig.add_traces(
    px.area(
        df,
        x=df.index,
        y='Close'
    ).data
)

# ==============================================================================
# Bar Chart
bar_fig = px.bar(
    price_df_melted,
    x='Year',
    y='Price',
    color='Price Type',
    barmode='group',
    title='Yearly High and Low Prices',
    labels={
        'Year': 'Year',
        'Price': 'Price (USD)',
        'Price Type': 'Price Type'
    },
    text='Price'
)

bar_fig.update_yaxes(
    tickprefix='$',
    title_text='Price (USD)'
)

bar_fig.update_layout(
    plot_bgcolor='#333333',
    paper_bgcolor='#333333',
    font=dict(color='white'),
    xaxis=dict(
        title='Year',
        tickmode='linear'
    ),
    yaxis=dict(
        title='Price (USD)'
    ),
    autosize=True,
    height=None,
    width=None,
    margin=dict(l=20, r=20, t=50, b=20),
)

# ==============================================================================
# Pie Chart
# Update layout to improve the appearance
pie_fig = px.pie(
    probability,
    names="Label",
    values="Percentage",
    title="Medusa's Action Recommendation",
    hole= 0.5
)

pie_fig.update_layout(
    plot_bgcolor='#333333',
    paper_bgcolor='#333333',
    font=dict(color='white'),
    margin=dict(l=20, r=20, t=50, b=20)
)

pie_fig.update_layout(
    plot_bgcolor='#333333',
    paper_bgcolor='#333333',
    font=dict(color='white'),
    margin=dict(l=20, r=20, t=75, b=20)
)

# ==============================================================================
# Indicator Chart

current_price = prediction['Values'][1]
current_price_prediction = prediction['Values'][2]

# Calculate accuracy as absolute percentage difference
accuracy = abs((current_price - current_price_prediction) / current_price_prediction) * 100

# Create the gauge chart figure with futuristic styling
# Create the gauge chart figure
gauge_fig = go.Figure(go.Indicator(
    mode="gauge+number+delta",
    value=current_price,
    title={'text': "Medusa's Predicted Price"},
    delta={'reference': current_price_prediction, 'position': "top"},
    gauge={'axis': {'range': [None, 1000]},
           'bar': {'color': "darkblue"},
           'steps': [
               {'range': [0, 250], 'color': "lightgray"},
               {'range': [250, 500], 'color': "gray"},
               {'range': [500, 750], 'color': "lightblue"},
               {'range': [750, 1000], 'color': "blue"}],
           'threshold': {'line': {'color': "red", 'width': 4},
           'thickness': 0.75, 'value': current_price_prediction}}))


gauge_fig.update_layout(
    plot_bgcolor='#333333',
    paper_bgcolor='#333333',
    font=dict(color='white'),
)

# ==============================================================================
# Initialize app
# ==============================================================================
# Define app with BOOTSTRAP themes
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# ==============================================================================
# Directory path
# ==============================================================================
# Get the directory where the script is running
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ==============================================================================
# Customize configuration to remove plotly branding
# ==============================================================================
# remove toolbar and logo
config = {
    'displayModeBar': False,  # Disable the mode bar
    'displaylogo': False  # Remove the Plotly logo
}

# Customize the HTML template to update tab-name and tab-logo
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        <meta charset="UTF-8">
        <title>Medusa</title>
        <link rel="icon" href="assets/custom_favicon.ico?v=1.0" type="image/x-icon">
        {%metas%}
        {%css%}
    </head>
    <body>
        <div id="app-entry">
            {%app_entry%}
        </div>
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# ==============================================================================
# App Layout
# ==============================================================================
app.layout = html.Div(
    style={'backgroundColor': '#333333', 'minHeight': '100vh', 'display': 'flex', 'flexDirection': 'column'},  # Page background color
    children=[
        navbar,
        dbc.Container([
            dbc.Row([
                dbc.Col([
                    dcc.Graph(
                        figure=line_fig,
                        config=config
                    ),
                    dbc.Row(dbc.Col(html.H2("News Articles", style={'color': 'white'}), className="mt-4")),
                    html.Div([
                        dbc.Card([
                            dbc.CardBody([
                                html.H5(article['title'], className="card-title", style={'color': 'white'}),
                                html.P(article['description'], className="card-text", style={'color': 'white'}),
                                dbc.CardLink("Read more", href=article['link'], target='_blank', style={'color': 'white'})
                            ])
                        ], className='mb-4', style={'backgroundColor': '#444444'}) for article in articles
                    ], style={'height': '500px', 'overflowY': 'scroll', 'backgroundColor': '#333333', 'padding': '10px', 'textAlign': 'left'})
                ], width=6),
                dbc.Col([
                    dcc.Graph(
                        figure=bar_fig,
                        config=config
                    ),
                    html.Div(
                        dcc.Graph(
                            figure=pie_fig,  # Add the pie chart here
                            config=config
                        ),
                        style={'marginTop': '50px'}  # Adjust the marginTop
                    )
                ], width=6)  # Right column width
            ]),
            dbc.Row([
                dbc.Col([
                    dcc.Graph(
                        figure=gauge_fig,
                        config=config
                    )
                ])
            ], className="mt-4", style={'backgroundColor': '#333333', })
        ], fluid=True, style={'padding': '20px'})
    ]
)

# ==============================================================================
# Run the app
# ==============================================================================
if __name__ == '__main__':
    app.run_server(debug=False)
# ==============================================================================
# server = app.server
# if __name__ == '__main__':
#     app.run_server(debug=True,host='0.0.0.0',port='8050')
