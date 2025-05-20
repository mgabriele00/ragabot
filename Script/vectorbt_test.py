import vectorbtpro as vbt
import pandas as pd
import matplotlib.pyplot as plt

column_names = ['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume']
csv_path='/Users/raffaele/Documents/GitHub/ragabot/Script/ragasim/dati_forex/EURUSD/DAT_MT_EURUSD_M1_2013.csv'
data = pd.read_csv(csv_path, header=None, names=column_names)
data['DateTime'] = pd.to_datetime(data['Date'] + ' ' + data['Time'])
data.set_index('DateTime', inplace=True)
data.drop(['Date', 'Time'], axis=1, inplace=True)
data_obj = vbt.Data.from_data(data)
open_price = data_obj.get('Open')
close_price = data_obj.get('Close')
rsi = vbt.RSI.run(close_price)
entries = rsi.rsi.vbt.crossed_below(30)
exit = rsi.rsi.vbt.crossed_above(70)

def plot_rsi(rsi, entries, exit):
    fig = rsi.plot()
    entries.vbt.signals.plot_as_entries(rsi.rsi, fig=fig)
    exit.vbt.signals.plot_as_exits(rsi.rsi, fig=fig)
    return fig

plot_rsi(rsi,entries, exit).show()