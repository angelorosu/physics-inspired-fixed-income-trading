import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_regression
import statsmodels.api as sm
from scipy import stats
from scipy.stats import mannwhitneyu

class Trading:
    def __init__(self, data: pd.DataFrame, initial_V=100_000, leverage=10, strategy='mean_reversion'):
        # make a copy and reset index 
        self.data = data.copy().reset_index(drop=True)
        self.initial_V = initial_V
        self.leverage = leverage
        self.strategy = strategy
        self.N = len(self.data)
        
    
        self.theta_t = np.zeros(self.N) # dollar position in SPTL
        self.V_total = np.zeros(self.N) # total portfolio value
        self.delta_V = np.zeros(self.N) # change in portfolio value due to postion of SPTL changes
        self.delta_V_cap = np.zeros(self.N) #the rest of margin getting interest at daily risk free rate
        
        # Set initial total portfolio value
        self.V_total[0] = initial_V # we start with 100k
        
        # For leveraged trading strategies, start with full notional exposure
        # For an interest-only strategy, start with zero exposure so that all capital earns interest.
        if self.strategy == 'interest_market':
            self.theta_t[0] = 0 # incase we want to just see how much interst we can make
        else:
            self.theta_t[0] = initial_V * leverage # this is for trading strategies

        
    def sharpe_confidence_interval(self, alpha=0.05):
        """
        Computes a (1 - alpha)% confidence interval for the annualized Sharpe ratio
        using the normal approximation.
        """
        returns = np.diff(self.V_total) / self.V_total[:-1]
        rf = np.mean(self.data['EFFR_Daily'].iloc[1:])
        excess_returns = returns - rf
        sharpe_daily = np.mean(excess_returns) / np.std(excess_returns)
        sharpe_annual = sharpe_daily * np.sqrt(252)
        
        N = len(returns)
        se_annual = np.sqrt((1 + 0.5 * sharpe_annual**2) / N) * np.sqrt(252)
        z = stats.norm.ppf(1 - alpha/2)
        
        lower = sharpe_annual - z * se_annual
        upper = sharpe_annual + z * se_annual

        print(f"{(1-alpha)*100:.0f}% CI for Sharpe Ratio: ({lower:.3f}, {upper:.3f})")
        return lower, upper
    
    
    def ising_strategy(self, window=20, threshold=0.8, tune=False, train_split=0.6,
                    window_candidates=np.arange(30, 50, 1),     
                    threshold_candidates=np.arange(0.1,0.2,0.001)) -> np.array:
        if tune:
            split_idx = int(train_split * self.N)
            orig_data = self.data.copy()
            train_data = self.data.iloc[:split_idx].reset_index(drop=True)
            train_N = train_data.shape[0]
            prices = train_data['SPTL_Close'].to_numpy()
            max_exposure = self.initial_V * self.leverage
            
            best_sharpe = -np.inf
            best_params = None
            best_candidate_signal = None
            
            # Compute training returns.
            train_returns = train_data['SPTL_Close'].pct_change().fillna(0).to_numpy()
            # Compute spins.
            spins_train = np.where(train_returns > 0, 1, np.where(train_returns < 0, -1, 0))
            
            for w in window_candidates:
                for th in threshold_candidates:
                    candidate_signal = np.zeros(train_N)
                    # For t < w, signal is neutral.
                    for t in range(w, train_N):
                        magnetization = np.mean(spins_train[t-w:t])
                        if magnetization > th:
                            candidate_signal[t] = -1
                        elif magnetization < -th:
                            candidate_signal[t] = 1
                        else:
                            candidate_signal[t] = 0
                    
                    # Simulate trading on training set.
                    V_total_sim = np.zeros(train_N)
                    theta_sim = np.zeros(train_N)
                    V_total_sim[0] = self.initial_V
                    theta_sim[0] = self.initial_V * self.leverage
                    for i in range(1, train_N):
                        incremental_trade = max_exposure * candidate_signal[i]
                        new_position = theta_sim[i-1] + incremental_trade
                        if abs(new_position) > max_exposure:
                            new_position = np.sign(new_position) * max_exposure
                        theta_sim[i] = new_position
                        
                        price_change = (train_data['SPTL_Close'].iloc[i] - train_data['SPTL_Close'].iloc[i-1]) / train_data['SPTL_Close'].iloc[i-1]
                        delta_V = theta_sim[i-1] * (price_change - train_data['EFFR_Daily'].iloc[i])
                        margin_used = abs(theta_sim[i-1]) / self.leverage
                        delta_V_cap = (V_total_sim[i-1] - margin_used) * train_data['EFFR_Daily'].iloc[i]
                        V_total_sim[i] = V_total_sim[i-1] + delta_V + delta_V_cap
                    
                    period_returns = np.diff(V_total_sim) / V_total_sim[:-1]
                    rf_period = np.mean(train_data['EFFR_Daily'].iloc[1:])
                    if np.std(period_returns) > 0:
                        sharpe = (np.mean(period_returns) - rf_period) / np.std(period_returns)
                    else:
                        sharpe = -np.inf
                    
                    if sharpe > best_sharpe:
                        best_sharpe = sharpe
                        best_params = {'window': w, 'threshold': th}
                        best_candidate_signal = candidate_signal.copy()
            
            print("Tuned Ising parameters:", best_params, "with Sharpe:", best_sharpe)
            # Restore full dataset.
            self.data = orig_data
            # Recompute signal on full data using best parameters.
            full_signal = np.zeros(self.N)
            full_prices = self.data['SPTL_Close'].to_numpy()
            # Compute full returns and spins.
            full_returns = self.data['SPTL_Close'].pct_change().fillna(0).to_numpy()
            spins_full = np.where(full_returns > 0, 1, np.where(full_returns < 0, -1, 0))
            w = best_params['window']
            th = best_params['threshold']
            for t in range(w, self.N):
                magnetization = np.mean(spins_full[t-w:t])
                if magnetization > th:
                    full_signal[t] = -1
                elif magnetization < -th:
                    full_signal[t] = 1
                else:
                    full_signal[t] = 0
            return full_signal
        else:
            # Without tuning: compute signal on full data.
            returns = self.data['SPTL_Close'].pct_change().fillna(0).to_numpy()
            spins = np.where(returns > 0, 1, np.where(returns < 0, -1, 0))
            signal = np.zeros(self.N)
            for t in range(window, self.N):
                magnetization = np.mean(spins[t-window:t])
                if magnetization > threshold:
                    signal[t] = -1
                elif magnetization < -threshold:
                    signal[t] = 1
                else:
                    signal[t] = 0
            return signal
    
    def buy_and_hold(self) -> np.array:
        """
        Generates a buy-and-hold signal for the entire dataset.
        """
        signal = np.ones(self.N)
        return signal
        
    def hurst_exponent(self, ts: np.ndarray) -> float:
        # Use a range of lags.
        lags = range(2, 20)
        # Compute tau for each lag.
        tau = [np.std(ts[lag:] - ts[:-lag]) for lag in lags]
        # Fit a line to log(lags) vs. log(tau).
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        H = poly[0]
        return H




    def hurst_strategy(self, window=50, upper_threshold=0.55, lower_threshold=0.45, tune=False, train_split=0.7) -> np.array:

        if tune:
            # hyperparameters.
            candidate_windows = [30,31,32,33,34,35,36,37,]
            candidate_upper = [0.53,0.54,0.55,0.56,0.57,0.58,0.59,0.60]
            candidate_lower = [0.13,0.14,0.15,0.16,0.17,0.18,0.19,0.20,0.21]
            
            # split the data .
            split_idx = int(train_split * self.N)
            orig_data = self.data.copy()
            self.data = self.data.iloc[:split_idx].reset_index(drop=True)
            train_N = self.data.shape[0]
            max_exposure = self.initial_V * self.leverage
            prices = self.data['SPTL_Close'].to_numpy()
            
            best_sharpe = -np.inf
            best_params = {}
            best_signal = None
            
            # Loop over params
            for w in candidate_windows:
                for up in candidate_upper:
                    for low in candidate_lower:
                        if low >= up:
                            continue  
                        
                        candidate_signal = np.zeros(train_N)
                        for t in range(w, train_N):
                            window_prices = prices[t-w:t]
                            H = self.hurst_exponent(window_prices)
                            if H > up:
                                candidate_signal[t] = 1
                            elif H < low:
                                candidate_signal[t] = -1
                            else:
                                candidate_signal[t] = 0
                                
                        # simulate trading performance on training set.
                        V_total_sim = np.zeros(train_N)
                        theta_sim = np.zeros(train_N)
                        V_total_sim[0] = self.initial_V
                        theta_sim[0] = self.initial_V * self.leverage
                        for i in range(1, train_N):
                            incremental_trade = max_exposure * candidate_signal[i]
                            new_position = theta_sim[i-1] + incremental_trade
                            if abs(new_position) > max_exposure:
                                new_position = np.sign(new_position) * max_exposure
                            theta_sim[i] = new_position

                            price_change = (self.data['SPTL_Close'].iloc[i] - self.data['SPTL_Close'].iloc[i-1]) / self.data['SPTL_Close'].iloc[i-1]
                            delta_V = theta_sim[i-1] * (price_change - self.data['EFFR_Daily'].iloc[i])
                            margin_used = abs(theta_sim[i-1]) / self.leverage
                            delta_V_cap = (V_total_sim[i-1] - margin_used) * self.data['EFFR_Daily'].iloc[i]
                            V_total_sim[i] = V_total_sim[i-1] + delta_V + delta_V_cap

                        period_returns = np.diff(V_total_sim) / V_total_sim[:-1]
                        rf_period = np.mean(self.data['EFFR_Daily'].iloc[1:])
                        if np.std(period_returns) > 0:
                            sharpe = (np.mean(period_returns) - rf_period) / np.std(period_returns)
                        else:
                            sharpe = -np.inf

                        if sharpe > best_sharpe:
                            best_sharpe = sharpe
                            best_params = {'window': w, 'upper_threshold': up, 'lower_threshold': low}
                            best_signal = candidate_signal.copy()

            print("Tuned Hurst parameters:", best_params, "with Sharpe:", best_sharpe)
            # return the full dataset
            self.data = orig_data
            # compute the signal on the full data using the best parameter
            full_signal = np.zeros(self.N)
            full_prices = self.data['SPTL_Close'].to_numpy()
            w = best_params['window']
            up = best_params['upper_threshold']
            low = best_params['lower_threshold']
            for t in range(w, self.N):
                window_prices = full_prices[t-w:t]
                H = self.hurst_exponent(window_prices)
                if H > up:
                    full_signal[t] = 1
                elif H < low:
                    full_signal[t] = -1
                else:
                    full_signal[t] = 0
            return full_signal
        else:
            # no tuning = provided parames
            signal = np.zeros(self.N)
            prices = self.data['SPTL_Close'].to_numpy()
            for t in range(window, self.N):
                window_prices = prices[t-window:t]
                H = self.hurst_exponent(window_prices)
                if H > upper_threshold:
                    signal[t] = 1
                elif H < lower_threshold:
                    signal[t] = -1
                else:
                    signal[t] = 0
            return signal


    
        
    def run_strategy(self):
        # pcik chosen strat => function defined
        if self.strategy == 'hurst':
            signal = self.hurst_strategy(tune=True)
        elif self.strategy == 'ising':
            signal = self.ising_strategy(tune=True)
        elif self.strategy == 'buy_and_hold':
            signal = self.buy_and_hold()

        else:
            raise ValueError('Invalid strategy')
        
        # max dollar exposed (max leverage)
        max_exposure = self.initial_V * self.leverage
        
        # loop over time steps
        for i in range(1, self.N):
            # Calculate the incremental trade order based on the signal
            # eg => if signal[i] == 0.5 then trade 50% of max_exposure
            incremental_trade = max_exposure * signal[i]
            
            # Update the position at each step by adding the incremental trade
            new_position = self.theta_t[i-1] + incremental_trade
            
            #  |theta| must not exceed max_exposure
            if abs(new_position) > max_exposure:
                new_position = np.sign(new_position) * max_exposure
            
            self.theta_t[i] = new_position

            #  the price change between day i-1 and i
            price_change = (self.data['SPTL_Close'][i] - self.data['SPTL_Close'][i-1]) / self.data['SPTL_Close'][i-1]
            
            #  trading PnL (ΔV) based on the previous position
            #  incorporates the net return (price change minus the risk-free rate)
            self.delta_V[i] = self.theta_t[i-1] * (price_change - self.data['EFFR_Daily'][i])
            
            #  margin used at time i-1 (capital tied up by the position)
            margin_used = np.abs(self.theta_t[i-1]) / self.leverage
            
            #  money-market gain (ΔV_cap) earned on unused margin capital
            self.delta_V_cap[i] = (self.V_total[i-1] - margin_used) * self.data['EFFR_Daily'][i]
            
            #  total portfolio value using both components.
            self.V_total[i] = self.V_total[i-1] + self.delta_V[i] + self.delta_V_cap[i]

        print("Final Portfolio Value:", self.V_total[-1])
        #  period returns: r_t = (V_total[t] - V_total[t-1]) / V_total[t-1]
        period_returns, sharpe_ratio = self.sharpe_ratio()
        
        #  maximum drawdown:
        cum_max = np.maximum.accumulate(self.V_total)
        drawdowns = (self.V_total - cum_max) / cum_max
        max_drawdown = abs(np.min(drawdowns))

        # annualise the mean daily return (compound growth)
        annualized_return = (1 + np.mean(period_returns))**252 - 1

        # Calculate Calmar Ratio: annualized return divided by maximum drawdown
        calmar_ratio = annualized_return / max_drawdown if max_drawdown != 0 else np.nan
        
        print("Strategy:", self.strategy)
        print("Sharpe Ratio:", sharpe_ratio)
        print("Calmar Ratio:", calmar_ratio)

        return signal

    def sharpe_ratio(self):
        period_returns = np.diff(self.V_total) / self.V_total[:-1]
        
        # Use the average risk-free rate over the periods without first day 
        rf_period = np.mean(self.data['EFFR_Daily'][1:])
        
        # Calculate Sharpe Ratio 
        sharpe_ratio = ((np.mean(period_returns) - rf_period) / np.std(period_returns)) * np.sqrt(252)
        return period_returns,sharpe_ratio
    

     
    def plot_results(self):
        #  x-axis as simple integer days.
        x = np.arange(self.N)
        
        #  change in position (delta) to determine buy/sell signals.
        #  positive change indicates buying <=> a negative change indicates selling.
        dtheta = np.diff(self.theta_t)
        # We add 1 to the indices because np.diff returns an array of length N-1.
        buy_indices = np.where(dtheta > 0)[0] + 1
        sell_indices = np.where(dtheta < 0)[0] + 1

        plt.figure(figsize=(12, 10))
        
        # Price chart with buy / sell arrows
        ax1 = plt.subplot(4, 1, 1)
        ax1.plot(x, self.data['SPTL_Close'], label='SPTL_Close', color='blue')
        ax1.scatter(buy_indices, self.data['SPTL_Close'].iloc[buy_indices],
                    marker='^', color='green', s=100, label='Buy')
        ax1.scatter(sell_indices, self.data['SPTL_Close'].iloc[sell_indices],
                    marker='v', color='red', s=100, label='Sell')
        ax1.set_title('SPTL Price with Buy/Sell Signals')
        ax1.legend()
        
        # leveraged Position - theta)
        ax2 = plt.subplot(4, 1, 2)
        ax2.plot(x, self.theta_t, label='Position (theta)', color='purple')
        ax2.set_title('Leveraged Position')
        ax2.legend()
        
        # total Portfolio Value
        ax3 = plt.subplot(4, 1, 3)
        ax3.plot(x, self.V_total, label='Total Portfolio Value', color='orange')
        ax3.set_title('Portfolio Value')
        ax3.legend()

        #lets save all the plots
        plt.savefig('plots.png')

        plt.show()


    def plot_pnl_components(self):

        
        x = np.arange(self.N)
        
        # Calculate daily total PnL and cumulative sums
        delta_trading = self.delta_V
        delta_interest = self.delta_V_cap
        delta_total = delta_trading + delta_interest  # ΔV_total
        
        cum_trading = np.cumsum(delta_trading)
        cum_interest = np.cumsum(delta_interest)
        cum_total = np.cumsum(delta_total)
        
        # Create figure and axes
        fig, axs = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
        
        # 1) Plot daily trading PnL (ΔV) & cumulative
        axs[0].plot(x, delta_trading, label=r'Daily Trading $ \Delta V_t $', color='blue')
        axs[0].plot(x, cum_trading, label=r'Cumulative Trading $ \sum \Delta V_t $', 
                    color='navy', linestyle='--')
        axs[0].set_title(rf'Trading PnL ($\Delta V_t, \sum \Delta V_t $) — Strategy: {self.strategy}')
        axs[0].set_ylabel('PnL ($)')
        axs[0].set_xlabel('Time Steps')
        axs[0].legend(loc='best')
        
        # 2) Plot daily money-market gain (ΔV_cap) & cumulative
        axs[1].plot(x, delta_interest, label=r'Daily Interest $ \Delta V_{cap,t} $', color='green')
        axs[1].plot(x, cum_interest, label=r'Cumulative Interest', color='darkgreen', linestyle='--')
        axs[1].set_title(rf'Interest PnL ($\Delta V_{{cap}}, \sum \Delta V_{{cap}}$) — Strategy: {self.strategy}')
        axs[1].set_ylabel('PnL ($)')
        axs[1].set_xlabel('Time Steps')
        axs[1].legend(loc='best')
        #add more spacing between the plots

        
        # 3) Plot daily total PnL (ΔV_total) & cumulative
        axs[2].plot(x, delta_total, label=r'Daily Total $ \Delta V_{t}^{total} $', color='red')
        axs[2].plot(x, cum_total, label=r'Cumulative Total PnL', color='darkred', linestyle='--')
        axs[2].set_title(rf'Total PnL ($\Delta V_{{t}}^{{total}}, \sum \Delta V_{{t}}^{{total}}$) — Strategy: {self.strategy}')
        axs[2].set_ylabel('PnL ($)')
        axs[2].set_xlabel('Time Steps')
        axs[2].legend(loc='best')
        
        
        # Adjust layout to prevent overlaps
        plt.tight_layout(pad=1.0)
        plt.savefig(f'{self.strategy}_pnl.png', dpi=300)
        plt.show()



    def plot_additional_metrics(self, rolling_window=60):
        import matplotlib.pyplot as plt
        import numpy as np

        # Create x-axis
        x = np.arange(self.N)

        # drawdonw
        cum_max = np.maximum.accumulate(self.V_total)
        drawdown = (self.V_total - cum_max) / cum_max

        # rollign sharpe 
        period_returns = np.diff(self.V_total) / self.V_total[:-1]
        rolling_sharpe = np.full_like(period_returns, np.nan)

        rf_daily = np.mean(self.data['EFFR_Daily'].iloc[1:])
        for i in range(rolling_window, len(period_returns)):
            window = period_returns[i - rolling_window:i]
            std = np.std(window)
            if std > 0:
                rolling_sharpe[i] = (np.mean(window) - rf_daily) / std * np.sqrt(252)

        #  x-axis for rolling_sharpe -> one less because of np.diff
        x_sharpe = x[1:]

        # turn over 
        turnover = np.zeros(self.N)
        for i in range(1, self.N):
            turnover[i] = abs(self.theta_t[i] - self.theta_t[i-1]) / self.data['SPTL_Close'].iloc[i]
        cum_turnover = np.cumsum(turnover)

        fig, axs = plt.subplots(3, 1, figsize=(10, 10))

        #  Drawdown
        axs[0].plot(x, drawdown, color='red', label='Drawdown')
        axs[0].set_title('Drawdown Profile')
        axs[0].set_ylabel('Drawdown')
        axs[1].set_xlabel('Time Steps')
        axs[0].legend()

        #     Rolling Sharpe Ratio
        axs[1].plot(x_sharpe[rolling_window:], rolling_sharpe[rolling_window:], color='purple', label=f'{rolling_window}-Day Rolling Sharpe')
        axs[1].set_title(f'Rolling {rolling_window}-Day Sharpe Ratio')
        axs[1].set_ylabel('Sharpe Ratio')
        axs[1].set_xlabel('Time Steps')
        axs[1].legend()

        # Turnover - jsut for visualisation
        axs[2].plot(x, turnover, label='Daily Turnover', color='green')
        axs[2].plot(x, cum_turnover, label='Cumulative Turnover', linestyle='--', color='blue')
        axs[2].set_title('Turnover')
        axs[2].set_xlabel('Time Steps')
        axs[2].set_ylabel('Turnover')
        axs[2].legend()

        plt.tight_layout()
        plt.show()

def mann_whitney_test(returns_strategy_1, returns_strategy_2):
 
    u_stat, p_value = mannwhitneyu(returns_strategy_1, returns_strategy_2, alternative='less')
    print(f"Mann-Whitney U Test:")
    print(f"U statistic: {u_stat}")
    print(f"P-value (Strategy 2 > Strategy 1): {p_value:.4f}")
    return u_stat, p_value



import data_prep as dp
df = dp.download_prep_data()  
dp.plot(df)  


def compute_metrics(ts: Trading, train_frac=0.3):
    
    split_idx = int(train_frac * ts.N)
    returns = np.diff(ts.V_total) / ts.V_total[:-1]
    
    # Split train and test returns
    train_returns = returns[:split_idx-1]  # returns length is N-1
    test_returns  = returns[split_idx-1:]
    
    # Risk-free rate 
    rf = np.mean(ts.data['EFFR_Daily'].iloc[1:])
    
    #  annualised Sharpe ratios
    sharpe_train = (np.mean(train_returns) - rf) / np.std(train_returns) * np.sqrt(252) if np.std(train_returns) > 0 else np.nan
    sharpe_test  = (np.mean(test_returns) - rf) / np.std(test_returns) * np.sqrt(252) if np.std(test_returns) > 0 else np.nan

    #  drawdowns for Calmar ratio
    cum_max = np.maximum.accumulate(ts.V_total)
    drawdowns = (ts.V_total - cum_max) / cum_max
    max_drawdown_train = abs(np.min(drawdowns[:split_idx]))
    max_drawdown_test  = abs(np.min(drawdowns[split_idx:]))
    
    # Annualised returns 
    annual_return_train = (1 + np.mean(train_returns))**252 - 1
    annual_return_test  = (1 + np.mean(test_returns))**252 - 1
    
    calmar_train = annual_return_train / max_drawdown_train if max_drawdown_train > 0 else np.nan
    calmar_test  = annual_return_test / max_drawdown_test if max_drawdown_test > 0 else np.nan

    final_value = ts.V_total[-1]
    return sharpe_train, sharpe_test, calmar_train, calmar_test, final_value

# trading instances for Hurst, Ising, and Buy & Hold strategies
hurst_ts = Trading(df, strategy='hurst')
hurst_ts.run_strategy()

ising_ts = Trading(df, strategy='ising')
ising_ts.run_strategy()

bh_ts = Trading(df, strategy='buy_and_hold')
bh_ts.run_strategy()

# Compute metrics 
hurst_sharp_train, hurst_sharp_test, hurst_calmar_train, hurst_calmar_test, hurst_final = compute_metrics(hurst_ts, train_frac=0.3)
ising_sharp_train, ising_sharp_test, ising_calmar_train, ising_calmar_test, ising_final = compute_metrics(ising_ts, train_frac=0.3)
bh_sharp_train, bh_sharp_test, bh_calmar_train, bh_calmar_test, bh_final = compute_metrics(bh_ts, train_frac=0.3)

# summary table using pandas
import pandas as pd

data = {
    'Metric': ['Sharpe Ratio', 'Calmar Ratio', 'Final Value ($)'],
    'Hurst (Train)': [round(hurst_sharp_train, 2), round(hurst_calmar_train, 2), round(hurst_final, 2)],
    'Hurst (Test)':  [round(hurst_sharp_test, 2), round(hurst_calmar_test, 2), '---'],
    'Ising (Train)': [round(ising_sharp_train, 2), round(ising_calmar_train, 2), round(ising_final, 2)],
    'Ising (Test)':  [round(ising_sharp_test, 2), round(ising_calmar_test, 2), '---'],
    'Buy & Hold':    [round(bh_sharp_test, 2), round(bh_calmar_test, 2), round(bh_final, 2)]
}

results_table = pd.DataFrame(data)
print(results_table)
