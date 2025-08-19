import yfinance as yf
import pandas as pd
import pandas_datareader as pdr
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
def download_prep_data(start_date='2023-01-01', end_date='2023-12-31'):

    # fetch if file doesnt exist
    try:
        df = pd.read_csv('data.csv')
    except:

        sptl = yf.download('SPTL', start=start_date, end=end_date,auto_adjust=False)
        effr = pdr.get_data_fred('DFF', start_date, end_date)

            # Fetch SPTL ETF adjusted closing price (trading days only)
        sptl = yf.download("SPTL", start_date, end_date)
        print(sptl.head())
        sptl_close = sptl['Close']

        # Fetch Effective Federal Funds Rate (EFFR) from FRED (available daily)
        effr = pdr.get_data_fred('DFF', start_date, end_date)

        sptl_close.index = pd.to_datetime(sptl_close.index)
        effr.index = pd.to_datetime(effr.index)

        # we make empty df with SPTL dates
        df = pd.DataFrame(index=sptl_close.index)
        # we merge EFFR data with SPTL dates
        df['SPTL_Close'] = sptl_close
        df['SPTL_Volume'] = sptl['Volume']
        df['SPTL_Open'] = sptl['Open']
        df['SPTL_High'] = sptl['High']
        df['SPTL_Low'] = sptl['Low']
        df['EFFR'] = effr['DFF']

        dc = 1/252
        df['EFFR_Daily'] = (df['EFFR']/100) * dc
        #he daily excess return per unit of SPTL reads exc_ret = change in sptl/sptl - effr_daily
        df['Excessive_Returns_SPTL'] = df['SPTL_Close'].pct_change() - df['EFFR_Daily']
        # lets add returns of sptl
        df['SPTL_Returns'] = df['SPTL_Close'].pct_change()


        #save to csv
        df.to_csv('data.csv')

    return df





def plot(df):
    df_plot = df[['SPTL_Returns', 'EFFR_Daily', 'Excessive_Returns_SPTL']].dropna().reset_index(drop=True)

    

    #plot with sns
    sns.set_style("whitegrid")
    fig,ax = plt.subplots(figsize=(12,6))
    sns.lineplot(data=df_plot, ax=ax)
    ax.set_title('SPTL Return, EFFR Daily, and Excess Return')
    ax.set_xlabel('t (Days since Jan 1, 2023)')
    ax.set_ylabel('Return')
    #save plot
    plt.savefig('SPTL_EFFR_EXCESSIVE.png', dpi=300)
    plt.show()


def plot_volume_price(df):
    top_plt = plt.subplot2grid((5,4), (0, 0), rowspan=3, colspan=4)
    top_plt.plot(df.index, df["SPTL_Close"])
    plt.title('SPTL Historcial Prices. [01-01-2023 to 31-12-2023]')
    bottom_plt = plt.subplot2grid((5,4), (3,0), rowspan=1, colspan=4)
    bottom_plt.bar(df.index, df['SPTL_Volume'],color='green')
    plt.title('\nSPTL Trading Volume', y=-0.60)
    plt.gcf().set_size_inches(12,8)

def plot_effr(df):
    plt.figure(figsize=(8, 6))
    plt.plot(df['EFFR'])
    plt.title('EFFR Daily')
    plt.xlabel('Days from Jan 1, 2023')
    plt.ylabel('EFFR in %')
    plt.savefig('EFFR_daily.png', dpi=300)
    plt.show()
        
## main
df = download_prep_data()
plot_effr(df)
#print(df.columns)
plot_volume_price(df)

plot(df)
