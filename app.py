#Import Streamlit and Plotly packages
import streamlit as st
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import plotly.express as px
#Import pandas, np and hvplot
import pandas as pd
import numpy as np
import hvplot
import hvplot.pandas
import holoviews as hv
#Import Alpaca Trading API
import alpaca_trade_api as tradeapi
#Import additional packages
import datetime
import random
# Import MCForecast Simulation
from MCForecastTools import MCSimulation
from warnings import simplefilter
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
# Import environment variables
from dotenv import load_dotenv
import os
load_dotenv('api.env')
alpaca_api_key = os.getenv("ALPACA_API_KEY")
alpaca_secret_key = os.getenv("ALPACA_SECRET_KEY")

#Full list of stocks available for selection. Got from the list of companies in SP500.
full_tickers = ['MSFT','AAPL','NVDA','AMZN','GOOGL','META','BRK.B','LLY','TSM','TSLA','AVGO','V','NVO','JPM','WMT','UNH','MA','XOM','JNJ','ASML','PG','HD','COST','MRK','TM','ABBV','ORCL','CRM','AMD','CVX','BAC','KO','NFLX','ADBE']

#Define App Title
st.title('Portfolio Dashboard')
st.markdown("-----")

#Create Sidebar with sections for user inputs
with st.sidebar:
    #Define user input dialog boxes
    st.markdown('-----')
    st.header('Stock Selection')
    stock_1 = st.selectbox(
       "Pick your first stock",
       options = full_tickers,
       index=full_tickers.index('MSFT'),
       placeholder="Pick your first stock",
    )
    stock_2 = st.selectbox(
       "Pick your second stock",
       options = full_tickers,
       index=full_tickers.index('AAPL'),
       placeholder="Pick your second stock",
    )
    start_date = st.date_input('Start Date', 
                                   value=datetime.date(2020,1,1),
                                   min_value=datetime.date(2018,1,1),
                                   max_value=datetime.date(2023,2,1))
    end_date = st.date_input('End Date', 
                                  value=datetime.date(2020,2,1),
                                  min_value=datetime.date(2018,1,1),
                                  max_value=datetime.date(2023,2,1))
    st.markdown('-----')
    st.sidebar.header('Portfolio Configuration')
    investment_amount = st.number_input("How much would you like to invest", min_value = 100000, max_value = 1000000000,value=100000,step=10000 , placeholder="How much would you like to invest")
    investment_horizon = st.number_input("How long would you like to invest for", min_value = 2, max_value = 100,value=5,step=1 , placeholder="How long would you like to invest")
    #weight_1 = st.sidebar.slider("How much would you like to invest (by %) in stock 1", 0, 100, (0, 100))
    weight_2 = 100#-weight_1
    weight1 = st.number_input("How much would you like to invest in stock 1", min_value = 0, max_value = 100,value=50,step=5 , placeholder="What % would you like to invest in stock 1")
    weight2 = 100 - weight1
    #weight2 = st.number_input("How much would you like to invest in stock 2", min_value = 0, max_value = 100,value=50,step=5 , placeholder="What % would you like to invest in stock 1")
    #st.write(weight_1)
    st.markdown('-----')
    


#Fetch all the stock data upfront
def fetch_stock_data(full_tickers):
    tickers = full_tickers
    start_date = pd.Timestamp("2014-01-01", tz="America/New_York").isoformat()
    end_date = pd.Timestamp("2024-02-29", tz="America/New_York").isoformat()
    timeframe = '1D'
    alpaca_api_key = os.getenv("ALPACA_API_KEY")    
    alpaca_secret_key = os.getenv("ALPACA_SECRET_KEY")    
    alpaca = tradeapi.REST(
        alpaca_api_key,
        alpaca_secret_key,
        api_version="v2")
    # Get current price data for tickers
    alpaca_df = alpaca.get_bars(
        tickers,
        timeframe,
        start = start_date,
        end = end_date
    ).df
    alpaca_df['date'] = alpaca_df.index.date
    return(alpaca_df)

#Format the portfolio dataframe for monte carlo simulation
def fetch_portfolio_data(stock_df, stock1, stock2):
    df1 = alpaca_df[alpaca_df['symbol']==stock1].drop('symbol', axis=1)
    df2 = alpaca_df[alpaca_df['symbol']==stock2].drop('symbol', axis=1)
    portfolio_df = pd.concat([df1, df2], axis=1, keys=[stock1, stock2])
    tickers = [stock1, stock2]
    df_closing_prices = pd.DataFrame()
    for ticker in tickers:
            df_closing_prices[ticker] = portfolio_df[ticker]["close"]
    df_closing_prices.index = df_closing_prices.index.date
    return(portfolio_df)
    
#Create a function to run the monte carlo simulation
def monte_carlo_simulation(portfolio_df, weights, investment_amount, investment_horizon):
    # Configuring a Monte Carlo simulation to forecast five years cumulative returns
    MC_fiveyear = MCSimulation(
        portfolio_data = portfolio_df,
        weights = weights,
        num_simulation = 500,
        num_trading_days = 252*investment_horizon
    )
    MC_fiveyear.calc_cumulative_return() 
    line_plot = MC_fiveyear.plot_simulation()
    dist_plot = MC_fiveyear.plot_distribution()
    tbl = MC_fiveyear.summarize_cumulative_return()
    max = round(tbl[7]*investment_amount,2)
    std_dev = round(tbl[2]*investment_amount,2)
    ci_lower = round(tbl[8]*investment_amount,2)
    ci_upper = round(tbl[9]*investment_amount,2)
    message = (f"There is a 95% chance that an initial investment of ${investment_amount} in the portfolio\nover the next 5 years will end within in the range of\n ${ci_lower} and ${ci_upper}")
    return(std_dev, line_plot, dist_plot, max, ci_lower, ci_upper)


alpaca_df = fetch_stock_data(full_tickers)

#Filter Alpaca DF based on user inputs
selected_stocks = [stock_1,stock_2]
weights = [weight1/100,weight2/100]
portfolio_df = alpaca_df.query('date >= @start_date and date <= @end_date and symbol in @selected_stocks')
mc_df = fetch_portfolio_data(alpaca_df, stock_1, stock_2)
stock1_df = alpaca_df.query('date >= @start_date and date <= @end_date and symbol == @stock_1')
stock2_df = alpaca_df.query('date >= @start_date and date <= @end_date and symbol == @stock_2')

#Calculate individual stock metrics for display:
def calculate_stock_metrics(stock_df):
    low = round(stock_df.loc[stock_df.index.min(), 'close'],2)
    high = round(stock_df.loc[stock_df.index.max(), 'close'],2)
    mean_return = round(stock_df['close'].pct_change().mean()*100,2)
    total_appreciation = round(((high-low)/low)*100,2)
    return_std = round(stock_df['close'].pct_change().std(),2)
    return low, high, mean_return, total_appreciation, return_std

#Create a function to generate a candlestick chart:
def get_candlestick_plot(stock_df):
    fig = make_subplots(
        rows = 2,
        cols = 1,
        shared_xaxes = True,
        vertical_spacing = 0.1,
        subplot_titles = ('Stock Price', 'Volume Chart'),
        row_width = [0.3, 0.7]
    )
    fig.add_trace(
        go.Candlestick(
            x = stock_df.index,
            open = stock_df['open'], 
            high = stock_df['high'],
            low = stock_df['low'],
            close = stock_df['close'],
            name = 'Candlestick chart'
        ),
        row = 1,
        col = 1,
    )
    fig.add_trace(
        go.Bar(x = stock_df.index, y = stock_df['volume'], name = 'volume'),
        row = 2,
        col = 1,
    )
    fig['layout']['xaxis2']['title'] = 'Date'
    fig['layout']['yaxis']['title'] = 'Price'
    fig['layout']['yaxis2']['title'] = 'Volume'
    fig.update_xaxes(
        rangebreaks = [{'bounds': ['sat', 'mon']}],
        rangeslider_visible = False,
    )
    return fig

#Calculate_mc_returns


#Create main panel area where we will house the various tabs:
#user_guide,
stock_performance, portfolio_simulation = st.tabs(['Stock Returns', 'Portfolio Simulation'])

#Create a section for a user guide
#with user_guide:
#    st.header("User Guide")
#    st.markdown("-----")
#    st.subheader("Add Text here and let's play with some shiz")

#Create a section for the user to evaluate individual stock performance
with stock_performance:
    stock1_tab, stock2_tab, dataframe = st.tabs([f"{stock_1 or 'Please pick a stock'}",f"{stock_2 or 'Please pick a stock'}","Data"])
    with stock1_tab:
        low1, high1, total_return1, total_appreciation1, return_std1 = calculate_stock_metrics(stock1_df)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write(f'Low: {low1}')
        with col2:
            st.write(f'High: {high1}')
        with col3:
            st.write(f'Avg Daily Return: {total_return1}%')
        col4, col5, col6 = st.columns(3)
        with col4:
            st.write(f'Total Increase: {total_appreciation1}')
        with col6:
            st.write(f'Returns Volatility: {return_std1}')
        st.markdown('-----')
        st.plotly_chart(get_candlestick_plot(stock1_df),use_container_width = True)
    with stock2_tab:
        low2, high2, total_return2, total_appreciation2, return_std2 = calculate_stock_metrics(stock2_df)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write(f'Low: {low2}')
        with col2:
            st.write(f'High: {high2}')
        with col3:
            st.write(f'Avg Daily Return: {total_return2}%')
        col4, col5, col6 = st.columns(3)
        with col4:
            st.write(f'Total Increase: {total_appreciation2}')
        with col6:
            st.write(f'Returns Volatility: {return_std2}')
        st.plotly_chart(get_candlestick_plot(stock2_df),use_container_width = True)
    with dataframe:
        st.write(portfolio_df)

#Create a section to review portfolio performance
with portfolio_simulation:
    st.markdown("-----")
    std, line_plot, dist_plot, max, ci_lower, ci_upper = monte_carlo_simulation(mc_df, weights, investment_amount, investment_horizon)
    #st.write(f"Of your ${investment_amount:,.0f} investment; {weights[0]*100}% will be invested in {stock_1}, and {weights[1]*100}% will be invested in {stock_2}")
    pie_df = pd.DataFrame()
    pie_df.at[0,'symbol'] = stock_1
    pie_df.at[0,'weight'] = weights[0]*100
    pie_df.at[1,'symbol'] = stock_2
    pie_df.at[1,'weight'] = weights[1]*100
    fig = px.pie(pie_df, values='weight', names='symbol', title='Portfolio Composition')
    #st.write(dist_plot)
    #st.write(message)
    st.write(f"After {investment_horizon} years your investment of ${investment_amount:,.0f}")
    st.write(f"will be worth between \$ {ci_lower:,.0f} and \$ {ci_upper:,.0f}")
    st.write(f"the volatility of your returns is: {std}")
    st.markdown('-----')
    st.plotly_chart(fig, use_container_width = True)
    
        