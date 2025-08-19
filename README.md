# Physics-Inspired Fixed-Income Trading (SPTL, 2023)

Two physics-inspired strategies—**Hurst mean-reversion** and an **Ising-style state model**—applied to the SPDR Portfolio Long Term Treasury ETF (**SPTL**) with leverage and proper capital accounting. The **Effective Federal Funds Rate (EFFR)** is used as the risk-free benchmark to compute excess returns and interest on unallocated capital. :contentReference[oaicite:0]{index=0}

---

## What’s inside
- **Data prep:** SPTL prices (Yahoo), daily **EFFR** (NY Fed), excess returns = Δp/p − r_f (with dailyised EFFR). :contentReference[oaicite:1]{index=1}
- **Strategies:**
  - **Hurst-based** mean reversion/trend filter with rolling H estimate and grid-searched thresholds. :contentReference[oaicite:2]{index=2}
  - **Ising-based** signal using exponentially-weighted magnetisation + spin–spin correlation, combined via tanh. :contentReference[oaicite:3]{index=3}
- **Leverage & capital:** position θ_t constrained by L×equity, margin M_t=|θ_t|/L, interest on idle cash via EFFR. :contentReference[oaicite:4]{index=4}
- **Evaluation:** Sharpe, Calmar, PnL decomposition (trading vs interest), drawdowns, rolling Sharpe; plus non-parametric tests. :contentReference[oaicite:5]{index=5}

---

## Results (Jan–Dec 2023, daily)
| Strategy   | Final Value ($) | Sharpe | Calmar |
|------------|-----------------:|------:|------:|
| Hurst      | 244,349.61       | 1.385 | 5.074 |
| Ising      | 363,120.80       | 1.796 | 8.855 |

Ising outperforms economically (higher terminal value and risk-adjusted metrics), though statistical tests on daily returns do **not** reject equality given the short sample (250 obs). :contentReference[oaicite:6]{index=6}

---

## Key Figures
- `SPTL_EFFR_EXCESSIVE.png` – SPTL returns, EFFR (dailyised), and excess returns  
- `EFFR_daily.png` – step profile of EFFR in 2023  
- `plots.png` – price with signals, position, and portfolio value

> Add a `/figures` folder and reference them like:
>
> ![EFFR Daily](figures/EFFR_daily.png)  
> ![Signals & PnL](figures/plots.png)  
> ![SPTL vs EFFR](figures/SPTL_EFFR_EXCESSIVE.png)

---

## Files
- `data_prep.py` – loading, cleaning, excess return calc  
- `trading_framework.py` – strategies, leverage/capital engine, metrics  
- `project_report.pdf` – full write-up with methods, formulas, and experiments (renamed from coursework) :contentReference[oaicite:7]{index=7}

---

## Limitations (transparent)
- Single asset (SPTL), single year (2023) → limited statistical power.  
- No transaction costs, slippage, or execution latency (would compress returns).  
- Hyperparameters tuned on a small split; results may not generalise. :contentReference[oaicite:8]{index=8}

---

## Roadmap
- Multi-year, multi-asset backtests (Treasuries curve, TLT/IEF).  
- Costs/Slippage model; borrow/financing costs for leverage.  
- Cross-validation and blocked bootstrap for inference.  
- Intraday extension (minute bars) and microstructure diagnostics.

---

**Credit:** Developed as part of MSc work at UCL (report included for methodology depth). :contentReference[oaicite:9]{index=9}
