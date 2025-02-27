from polygon import RESTClient
import pandas as pd
import json

# Get the price of a currency pair
# ticker: the currency pair to get the price for
# start: the start date
# end: the end date
# timeframe: the timeframe to get the price for
# return: a dictionary with the price of the currency pair
def get_currency_price(ticker, start, end, timeframe) -> dict[str, str]:
    client = RESTClient(api_key="zTruAu2OsRIS0vvonmfhZfB7DtxIJTIb")
    aggs = []
    for a in client.list_aggs(
        "C:" + ticker,
        1,
        timeframe,
        start,
        end,
        limit=50000,
    ):
        aggs.append(a)
        aggs_json = [vars(agg) for agg in aggs]
    return aggs_json    

# Save the price of a currency pair to a file
# ticker: the currency pair to get the price for
# start: the start date
# end: the end date
# timeframe: the timeframe to get the price for
# return: None
def save_currency_price(ticker, start, end, timeframe) -> None:
    aggs = get_currency_price(ticker, start, end, timeframe)
    filename = get_file_name(ticker, start, end, timeframe)
    with open(filename, "w") as file:
        json.dump(aggs, file, indent=4)
        
# Get the file name for a currency pair
# ticker: the currency pair to get the price for
# start: the start date
# end: the end date
# timeframe: the timeframe to get the price for
# return: the file name        
def get_file_name(ticker, start, end, timeframe) -> str:
    return "data/" + ticker + "_" + start + "_" + end + "_" + timeframe + ".json"


# Read the price of a currency pair from a file
# ticker: the currency pair to get the price for
# start: the start date
# end: the end date
# timeframe: the timeframe to get the price for
# return: a dictionary with the price of the currency pair
def read_currency_price(ticker, start, end, timeframe) -> dict[str, str]:
    filename = get_file_name(ticker, start, end, timeframe)
    with open(filename, "r") as file:
        aggs = json.load(file)
        return aggs
    
# Load the price of a currency pair, it will try to read the price from a file, if the file does not exist it will download the price and save it to a file
# ticker: the currency pair to get the price for
# start: the start date
# end: the end date
# timeframe: the timeframe to get the price for
# return: a dictionary with the price of the currency pair
def load_currency_price(ticker, start, end, timeframe) -> dict[str, str]:
    try:
        return read_currency_price(ticker, start, end, timeframe)
    except FileNotFoundError:
        save_currency_price(ticker, start, end, timeframe)
        return read_currency_price(ticker, start, end, timeframe)

# Get the price of a currency pair as a pandas DataFrame
# ticker: the currency pair to get the price for
# start: the start date
# end: the end date
# timeframe: the timeframe to get the price for
# return: a pandas DataFrame with the price of the currency pair
def get_currency_df(ticker, start, end, timeframe) -> pd.DataFrame:
    df = pd.DataFrame(load_currency_price(ticker, start, end, timeframe))
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.columns = [col.capitalize() for col in df.columns]
    df.set_index("Datetime", inplace=True)
    return df    
            

    

