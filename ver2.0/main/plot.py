from plotly import graph_objects

def show_fig(df):
    fig = graph_objects.Figure()
    fig.add_trace(graph_objects.Candlestick(
        open = df['open'],
        high = df['high'],
        low = df['low'],
        close =  df['close'],
        name = 'graph'
    ))
    fig.add_trace(graph_objects.Scatter(
        y = df['sma5'],
        name = 'SMA5'
    ))
    fig.add_trace(graph_objects.Scatter(
        y = df['sma25'],
        name = 'SMA25'
    ))
    fig.add_trace(graph_objects.Scatter(
        y = df['sma50'],
        name = 'SMA50'
    ))
    fig.add_trace(graph_objects.Scatter(
        y = df['sma100'],
        name = 'SMA100'
    ))
    fig.show()
    
