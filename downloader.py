from polygon import RESTClient
import vectorbt as vbt
import json
import vectorbt as vbt

price = vbt.YFData.download('BTC-USD').get('Close')

def get_currency_price(ticker):
    client = RESTClient(api_key="zTruAu2OsRIS0vvonmfhZfB7DtxIJTIb")
    aggs = []
    for a in client.list_aggs(
        "C:" + ticker,
        1,
        "minute",
        "2024-01-30",
        "2024-02-03",
        limit=50000,
    ):
        aggs.append(a)
    return aggs    

def save_currency_price(ticker):
    data = get_currency_price(ticker)
    print(data)
    with open("data/" + ticker + ".json", "w") as file:
        json.dump(data, file, indent=4)
        
save_currency_price("EURUSD")
    

