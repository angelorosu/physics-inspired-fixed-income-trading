# Physics-Inspired Fixed-Income Trading (SPTL, 2023)

Two physics-inspired strategies—**Hurst mean-reversion** and an **Ising-style state model**—applied to the SPDR Portfolio Long Term Treasury ETF (**SPTL**) with leverage and proper capital accounting. The **Effective Federal Funds Rate (EFFR)** is used as the risk-free benchmark to compute excess returns and interest on unallocated capital.

---

## What’s inside
- **Data prep:** SPTL prices (Yahoo), daily **EFFR** (NY Fed), excess returns = Δp/p − r_f (with dailyised EFFR). 
- **Strategies:**
  - **Hurst-based** mean reversion/trend filter with rolling H estimate and grid-searched thresholds. 
  - **Ising-based** signal using exponentially-weighted magnetisation + spin–spin correlation, combined via tanh. 
- **Leverage & capital:** position θ_t constrained by L×equity, margin M_t=|θ_t| / L, interest on idle cash via EFFR. 
- **Evaluation:** Sharpe, Calmar, PnL decomposition (trading vs interest), drawdowns, rolling Sharpe; plus non-parametric tests.

---

## Results (Jan–Dec 2023, daily)
| Strategy   | Final Value ($) | Sharpe | Calmar |
|------------|-----------------:|------:|------:|
| Hurst      | 244,349.61       | 1.385 | 5.074 |
| Ising      | 363,120.80       | 1.796 | 8.855 |

Ising outperforms economically (higher terminal value and risk-adjusted metrics), though statistical tests on daily returns do **not** reject equality given the short sample (250 obs). 

---

## Key Figures
- `SPTL_EFFR_EXCESSIVE.png` – SPTL returns, EFFR (dailyised), and excess returns  
- `EFFR_daily.png` – step profile of EFFR in 2023

## Plots
><img width="500" height="500" alt="EFFR_daily" src="https://github.com/user-attachments/assets/ecd6bde9-6b76-4934-8ec0-57be74419165" />
<img width="500" height="500" alt="plots" src="https://github.com/user-attachments/assets/a6c4b9c1-d070-4167-b47e-7250fc79b48d" />
<img width="500" height="500" alt="SPTL_EFFR_EXCESSIVE" src="https://github.com/user-attachments/assets/330bb0e6-b2d6-41cb-bf3e-ad677bc4cf25" />


---

## Files
- `data_prep.py` – loading, cleaning, excess return calc  
- `trading_framework.py` – strategies, leverage/capital engine, metrics  

---

## Limitations (transparent)
- Single asset (SPTL), single year (2023) → limited statistical power.  
- No transaction costs, slippage, or execution latency (would compress returns).  
- Hyperparameters tuned on a small split; results may not generalise.

---

## Roadmap
- Multi-year, multi-asset backtests (Treasuries curve, TLT/IEF).  
- Costs/Slippage model; borrow/financing costs for leverage.  
- Cross-validation and blocked bootstrap for inference.  
- Intraday extension (minute bars) and microstructure diagnostics.

