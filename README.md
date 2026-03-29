# Statistical Arbitrage: PCA Factor Modeling & Mean Reversion

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📌 Overview
This repository implements a systematic statistical arbitrage strategy based on the seminal paper *"Statistical Arbitrage in the U.S. Equities Market"* by Marco Avellaneda and Jeong-Hyun Lee (2008). 

Unlike traditional pairs trading (which relies on cointegration between two stocks), this project builds a "generalized pairs trading" framework. It uses Principal Component Analysis (PCA) to extract systematic market factors, isolates the idiosyncratic residuals of each equity, and models those residuals as an Ornstein-Uhlenbeck (OU) mean-reverting process to generate trading signals.

## 🏗️ System Architecture
The pipeline is fully object-oriented and designed for highly vectorized, rolling-window backtesting using `pandas` and `numpy` to ensure computational efficiency and prevent look-ahead bias.

*   **`data/`**: Ingestion and cleaning of historical daily price/volume data. Handles survivorship bias, corporate actions, and NaN forward-filling.
*   **`factors/`**: The PCA pipeline. Computes the empirical correlation matrix on a rolling window and extracts the top $K$ eigenportfolios (market factors).
*   **`models/`**: Estimates the Ornstein-Uhlenbeck (OU) process parameters. Uses an AR(1) linear regression to calculate the mean reversion speed ($\kappa$), equilibrium mean ($m$), and volatility ($\sigma$).
*   **`signals/`**: Vectorized logic to generate the dimensionless $s$-score (z-score) and map it to Long/Short/Flat target positions.
*   **`backtest/`**: The vectorized execution engine. Calculates portfolio PnL, transaction costs, and performance metrics.

## 🧮 Mathematical Framework

### 1. Factor Extraction
Returns are decomposed into a systematic component and an idiosyncratic component:
$$R_i = \beta_{i,1}F_1 + \beta_{i,2}F_2 + ... + \beta_{i,K}F_K + \tilde{R}_i$$
Where $F$ are the PCA factors and $\tilde{R}_i$ is the residual return.

### 2. Ornstein-Uhlenbeck Process
The cumulative residual price path ($x_t$) is modeled as a mean-reverting stochastic process:
$$dx_t = \kappa(m - x_t)dt + \sigma dW_t$$

### 3. Signal Generation
Trading signals are generated based on the $s$-score (distance from the equilibrium mean adjusted for volatility):
$$s_t = \frac{x_t - m}{\sigma_{eq}}$$
*   **Buy:** $s < -1.25$ (Asset is heavily oversold relative to factors)
*   **Short:** $s > 1.25$ (Asset is heavily overbought relative to factors)
*   **Close:** Reversion to $s \approx 0$

## 🚀 Quickstart & Reproduction

### 1. Installation
Clone the repository and install the required quantitative libraries:
```bash
git clone [https://github.com/yourusername/stat-arb-pca.git](https://github.com/yourusername/stat-arb-pca.git)
cd stat-arb-pca
pip install -r requirements.txt
