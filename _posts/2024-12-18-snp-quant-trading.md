---
layout: post
title:  "Financial Trade Backtesting with the S&P 500"
excerpt_separator: <!--more-->
---

Quantitative backtesting of a model used to initiate financial trading for all stocks contained within the S&P 500 for the time period between January 2020 - December 2024

<!--more-->

```python
#!pip install pandas-datareader yfinance
```


```python
# Import libraries
import pandas as pd
import time
import numpy as np
import warnings
warnings.filterwarnings("ignore")
```


```python
# Fetch the data on tickers
table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
tickers = table[0]
df = pd.read_csv('/Users/human/Documents/quant/snp500prices.csv')
symbols = [i for i in df['ticker'].unique()]
```


```python
df.shape
```




    (617702, 4)




```python
df.columns
```




    Index(['Date', 'price', 'ticker', 'daily_pct_change'], dtype='object')




```python
df['ticker'].nunique()
```




    501




```python
%%time

# Init df to store trade signals
signals = pd.DataFrame()

# Iterate the stock symbols to calculate trade signals
for nm, grp in df.groupby('ticker'):

    # Calculate
    grp['signal'] = 0.0

    # Calculate sma
    grp['short'] = grp['price'].rolling(window = 30, min_periods = 1, center = False).mean()

    # Calculate lma
    grp['long'] = grp['price'].rolling(window = 90, min_periods = 1, center = False).mean()

    # Create signals
    grp['signal'][30:] = np.where(grp['short'][30:] > grp['long'][30:], 1.0, 0.0)

    # Generate trading orders
    grp['positions'] = grp['signal'].diff()

    signals = pd.concat([signals, grp])
    
    print(f"Signals calculated for {nm}..")
```

    Signals calculated for A..
    Signals calculated for AAPL..
    Signals calculated for ABBV..
    Signals calculated for ABNB..
    Signals calculated for ABT..
    Signals calculated for ACGL..
    Signals calculated for ACN..
    Signals calculated for ADBE..
    Signals calculated for ADI..
    Signals calculated for ADM..
    Signals calculated for ADP..
    Signals calculated for ADSK..
    Signals calculated for AEE..
    Signals calculated for AEP..
    Signals calculated for AES..
    Signals calculated for AFL..
    Signals calculated for AIG..
    Signals calculated for AIZ..
    Signals calculated for AJG..
    Signals calculated for AKAM..
    Signals calculated for ALB..
    Signals calculated for ALGN..
    Signals calculated for ALL..
    Signals calculated for ALLE..
    Signals calculated for AMAT..
    Signals calculated for AMCR..
    Signals calculated for AMD..
    Signals calculated for AME..
    Signals calculated for AMGN..
    Signals calculated for AMP..
    Signals calculated for AMT..
    Signals calculated for AMTM..
    Signals calculated for AMZN..
    Signals calculated for ANET..
    Signals calculated for ANSS..
    Signals calculated for AON..
    Signals calculated for AOS..
    Signals calculated for APA..
    Signals calculated for APD..
    Signals calculated for APH..
    Signals calculated for APTV..
    Signals calculated for ARE..
    Signals calculated for ATO..
    Signals calculated for AVB..
    Signals calculated for AVGO..
    Signals calculated for AVY..
    Signals calculated for AWK..
    Signals calculated for AXON..
    Signals calculated for AXP..
    Signals calculated for AZO..
    Signals calculated for BA..
    Signals calculated for BAC..
    Signals calculated for BALL..
    Signals calculated for BAX..
    Signals calculated for BBY..
    Signals calculated for BDX..
    Signals calculated for BEN..
    Signals calculated for BG..
    Signals calculated for BIIB..
    Signals calculated for BK..
    Signals calculated for BKNG..
    Signals calculated for BKR..
    Signals calculated for BLDR..
    Signals calculated for BLK..
    Signals calculated for BMY..
    Signals calculated for BR..
    Signals calculated for BRO..
    Signals calculated for BSX..
    Signals calculated for BWA..
    Signals calculated for BX..
    Signals calculated for BXP..
    Signals calculated for C..
    Signals calculated for CAG..
    Signals calculated for CAH..
    Signals calculated for CARR..
    Signals calculated for CAT..
    Signals calculated for CB..
    Signals calculated for CBOE..
    Signals calculated for CBRE..
    Signals calculated for CCI..
    Signals calculated for CCL..
    Signals calculated for CDNS..
    Signals calculated for CDW..
    Signals calculated for CE..
    Signals calculated for CEG..
    Signals calculated for CF..
    Signals calculated for CFG..
    Signals calculated for CHD..
    Signals calculated for CHRW..
    Signals calculated for CHTR..
    Signals calculated for CI..
    Signals calculated for CINF..
    Signals calculated for CL..
    Signals calculated for CLX..
    Signals calculated for CMCSA..
    Signals calculated for CME..
    Signals calculated for CMG..
    Signals calculated for CMI..
    Signals calculated for CMS..
    Signals calculated for CNC..
    Signals calculated for CNP..
    Signals calculated for COF..
    Signals calculated for COO..
    Signals calculated for COP..
    Signals calculated for COR..
    Signals calculated for COST..
    Signals calculated for CPAY..
    Signals calculated for CPB..
    Signals calculated for CPRT..
    Signals calculated for CPT..
    Signals calculated for CRL..
    Signals calculated for CRM..
    Signals calculated for CRWD..
    Signals calculated for CSCO..
    Signals calculated for CSGP..
    Signals calculated for CSX..
    Signals calculated for CTAS..
    Signals calculated for CTLT..
    Signals calculated for CTRA..
    Signals calculated for CTSH..
    Signals calculated for CTVA..
    Signals calculated for CVS..
    Signals calculated for CVX..
    Signals calculated for CZR..
    Signals calculated for D..
    Signals calculated for DAL..
    Signals calculated for DAY..
    Signals calculated for DD..
    Signals calculated for DE..
    Signals calculated for DECK..
    Signals calculated for DELL..
    Signals calculated for DFS..
    Signals calculated for DG..
    Signals calculated for DGX..
    Signals calculated for DHI..
    Signals calculated for DHR..
    Signals calculated for DIS..
    Signals calculated for DLR..
    Signals calculated for DLTR..
    Signals calculated for DOC..
    Signals calculated for DOV..
    Signals calculated for DOW..
    Signals calculated for DPZ..
    Signals calculated for DRI..
    Signals calculated for DTE..
    Signals calculated for DUK..
    Signals calculated for DVA..
    Signals calculated for DVN..
    Signals calculated for DXCM..
    Signals calculated for EA..
    Signals calculated for EBAY..
    Signals calculated for ECL..
    Signals calculated for ED..
    Signals calculated for EFX..
    Signals calculated for EG..
    Signals calculated for EIX..
    Signals calculated for EL..
    Signals calculated for ELV..
    Signals calculated for EMN..
    Signals calculated for EMR..
    Signals calculated for ENPH..
    Signals calculated for EOG..
    Signals calculated for EPAM..
    Signals calculated for EQIX..
    Signals calculated for EQR..
    Signals calculated for EQT..
    Signals calculated for ERIE..
    Signals calculated for ES..
    Signals calculated for ESS..
    Signals calculated for ETN..
    Signals calculated for ETR..
    Signals calculated for EVRG..
    Signals calculated for EW..
    Signals calculated for EXC..
    Signals calculated for EXPD..
    Signals calculated for EXPE..
    Signals calculated for EXR..
    Signals calculated for F..
    Signals calculated for FANG..
    Signals calculated for FAST..
    Signals calculated for FCX..
    Signals calculated for FDS..
    Signals calculated for FDX..
    Signals calculated for FE..
    Signals calculated for FFIV..
    Signals calculated for FI..
    Signals calculated for FICO..
    Signals calculated for FIS..
    Signals calculated for FITB..
    Signals calculated for FMC..
    Signals calculated for FOX..
    Signals calculated for FOXA..
    Signals calculated for FRT..
    Signals calculated for FSLR..
    Signals calculated for FTNT..
    Signals calculated for FTV..
    Signals calculated for GD..
    Signals calculated for GDDY..
    Signals calculated for GE..
    Signals calculated for GEHC..
    Signals calculated for GEN..
    Signals calculated for GEV..
    Signals calculated for GILD..
    Signals calculated for GIS..
    Signals calculated for GL..
    Signals calculated for GLW..
    Signals calculated for GM..
    Signals calculated for GNRC..
    Signals calculated for GOOG..
    Signals calculated for GOOGL..
    Signals calculated for GPC..
    Signals calculated for GPN..
    Signals calculated for GRMN..
    Signals calculated for GS..
    Signals calculated for GWW..
    Signals calculated for HAL..
    Signals calculated for HAS..
    Signals calculated for HBAN..
    Signals calculated for HCA..
    Signals calculated for HD..
    Signals calculated for HES..
    Signals calculated for HIG..
    Signals calculated for HII..
    Signals calculated for HLT..
    Signals calculated for HOLX..
    Signals calculated for HON..
    Signals calculated for HPE..
    Signals calculated for HPQ..
    Signals calculated for HRL..
    Signals calculated for HSIC..
    Signals calculated for HST..
    Signals calculated for HSY..
    Signals calculated for HUBB..
    Signals calculated for HUM..
    Signals calculated for HWM..
    Signals calculated for IBM..
    Signals calculated for ICE..
    Signals calculated for IDXX..
    Signals calculated for IEX..
    Signals calculated for IFF..
    Signals calculated for INCY..
    Signals calculated for INTC..
    Signals calculated for INTU..
    Signals calculated for INVH..
    Signals calculated for IP..
    Signals calculated for IPG..
    Signals calculated for IQV..
    Signals calculated for IR..
    Signals calculated for IRM..
    Signals calculated for ISRG..
    Signals calculated for IT..
    Signals calculated for ITW..
    Signals calculated for IVZ..
    Signals calculated for J..
    Signals calculated for JBHT..
    Signals calculated for JBL..
    Signals calculated for JCI..
    Signals calculated for JKHY..
    Signals calculated for JNJ..
    Signals calculated for JNPR..
    Signals calculated for JPM..
    Signals calculated for K..
    Signals calculated for KDP..
    Signals calculated for KEY..
    Signals calculated for KEYS..
    Signals calculated for KHC..
    Signals calculated for KIM..
    Signals calculated for KKR..
    Signals calculated for KLAC..
    Signals calculated for KMB..
    Signals calculated for KMI..
    Signals calculated for KMX..
    Signals calculated for KO..
    Signals calculated for KR..
    Signals calculated for KVUE..
    Signals calculated for L..
    Signals calculated for LDOS..
    Signals calculated for LEN..
    Signals calculated for LH..
    Signals calculated for LHX..
    Signals calculated for LIN..
    Signals calculated for LKQ..
    Signals calculated for LLY..
    Signals calculated for LMT..
    Signals calculated for LNT..
    Signals calculated for LOW..
    Signals calculated for LRCX..
    Signals calculated for LULU..
    Signals calculated for LUV..
    Signals calculated for LVS..
    Signals calculated for LW..
    Signals calculated for LYB..
    Signals calculated for LYV..
    Signals calculated for MA..
    Signals calculated for MAA..
    Signals calculated for MAR..
    Signals calculated for MAS..
    Signals calculated for MCD..
    Signals calculated for MCHP..
    Signals calculated for MCK..
    Signals calculated for MCO..
    Signals calculated for MDLZ..
    Signals calculated for MDT..
    Signals calculated for MET..
    Signals calculated for META..
    Signals calculated for MGM..
    Signals calculated for MHK..
    Signals calculated for MKC..
    Signals calculated for MKTX..
    Signals calculated for MLM..
    Signals calculated for MMC..
    Signals calculated for MMM..
    Signals calculated for MNST..
    Signals calculated for MO..
    Signals calculated for MOH..
    Signals calculated for MOS..
    Signals calculated for MPC..
    Signals calculated for MPWR..
    Signals calculated for MRK..
    Signals calculated for MRNA..
    Signals calculated for MS..
    Signals calculated for MSCI..
    Signals calculated for MSFT..
    Signals calculated for MSI..
    Signals calculated for MTB..
    Signals calculated for MTCH..
    Signals calculated for MTD..
    Signals calculated for MU..
    Signals calculated for NCLH..
    Signals calculated for NDAQ..
    Signals calculated for NDSN..
    Signals calculated for NEE..
    Signals calculated for NEM..
    Signals calculated for NFLX..
    Signals calculated for NI..
    Signals calculated for NKE..
    Signals calculated for NOC..
    Signals calculated for NOW..
    Signals calculated for NRG..
    Signals calculated for NSC..
    Signals calculated for NTAP..
    Signals calculated for NTRS..
    Signals calculated for NUE..
    Signals calculated for NVDA..
    Signals calculated for NVR..
    Signals calculated for NWS..
    Signals calculated for NWSA..
    Signals calculated for NXPI..
    Signals calculated for O..
    Signals calculated for ODFL..
    Signals calculated for OKE..
    Signals calculated for OMC..
    Signals calculated for ON..
    Signals calculated for ORCL..
    Signals calculated for ORLY..
    Signals calculated for OTIS..
    Signals calculated for OXY..
    Signals calculated for PANW..
    Signals calculated for PARA..
    Signals calculated for PAYC..
    Signals calculated for PAYX..
    Signals calculated for PCAR..
    Signals calculated for PCG..
    Signals calculated for PEG..
    Signals calculated for PEP..
    Signals calculated for PFE..
    Signals calculated for PFG..
    Signals calculated for PG..
    Signals calculated for PGR..
    Signals calculated for PH..
    Signals calculated for PHM..
    Signals calculated for PKG..
    Signals calculated for PLD..
    Signals calculated for PLTR..
    Signals calculated for PM..
    Signals calculated for PNC..
    Signals calculated for PNR..
    Signals calculated for PNW..
    Signals calculated for PODD..
    Signals calculated for POOL..
    Signals calculated for PPG..
    Signals calculated for PPL..
    Signals calculated for PRU..
    Signals calculated for PSA..
    Signals calculated for PSX..
    Signals calculated for PTC..
    Signals calculated for PWR..
    Signals calculated for PYPL..
    Signals calculated for QCOM..
    Signals calculated for QRVO..
    Signals calculated for RCL..
    Signals calculated for REG..
    Signals calculated for REGN..
    Signals calculated for RF..
    Signals calculated for RJF..
    Signals calculated for RL..
    Signals calculated for RMD..
    Signals calculated for ROK..
    Signals calculated for ROL..
    Signals calculated for ROP..
    Signals calculated for ROST..
    Signals calculated for RSG..
    Signals calculated for RTX..
    Signals calculated for RVTY..
    Signals calculated for SBAC..
    Signals calculated for SBUX..
    Signals calculated for SCHW..
    Signals calculated for SHW..
    Signals calculated for SJM..
    Signals calculated for SLB..
    Signals calculated for SMCI..
    Signals calculated for SNA..
    Signals calculated for SNPS..
    Signals calculated for SO..
    Signals calculated for SOLV..
    Signals calculated for SPG..
    Signals calculated for SPGI..
    Signals calculated for SRE..
    Signals calculated for STE..
    Signals calculated for STLD..
    Signals calculated for STT..
    Signals calculated for STX..
    Signals calculated for STZ..
    Signals calculated for SW..
    Signals calculated for SWK..
    Signals calculated for SWKS..
    Signals calculated for SYF..
    Signals calculated for SYK..
    Signals calculated for SYY..
    Signals calculated for T..
    Signals calculated for TAP..
    Signals calculated for TDG..
    Signals calculated for TDY..
    Signals calculated for TECH..
    Signals calculated for TEL..
    Signals calculated for TER..
    Signals calculated for TFC..
    Signals calculated for TFX..
    Signals calculated for TGT..
    Signals calculated for TJX..
    Signals calculated for TMO..
    Signals calculated for TMUS..
    Signals calculated for TPL..
    Signals calculated for TPR..
    Signals calculated for TRGP..
    Signals calculated for TRMB..
    Signals calculated for TROW..
    Signals calculated for TRV..
    Signals calculated for TSCO..
    Signals calculated for TSLA..
    Signals calculated for TSN..
    Signals calculated for TT..
    Signals calculated for TTWO..
    Signals calculated for TXN..
    Signals calculated for TXT..
    Signals calculated for TYL..
    Signals calculated for UAL..
    Signals calculated for UBER..
    Signals calculated for UDR..
    Signals calculated for UHS..
    Signals calculated for ULTA..
    Signals calculated for UNH..
    Signals calculated for UNP..
    Signals calculated for UPS..
    Signals calculated for URI..
    Signals calculated for USB..
    Signals calculated for V..
    Signals calculated for VICI..
    Signals calculated for VLO..
    Signals calculated for VLTO..
    Signals calculated for VMC..
    Signals calculated for VRSK..
    Signals calculated for VRSN..
    Signals calculated for VRTX..
    Signals calculated for VST..
    Signals calculated for VTR..
    Signals calculated for VTRS..
    Signals calculated for VZ..
    Signals calculated for WAB..
    Signals calculated for WAT..
    Signals calculated for WBA..
    Signals calculated for WBD..
    Signals calculated for WDC..
    Signals calculated for WEC..
    Signals calculated for WELL..
    Signals calculated for WFC..
    Signals calculated for WM..
    Signals calculated for WMB..
    Signals calculated for WMT..
    Signals calculated for WRB..
    Signals calculated for WST..
    Signals calculated for WTW..
    Signals calculated for WY..
    Signals calculated for WYNN..
    Signals calculated for XEL..
    Signals calculated for XOM..
    Signals calculated for XYL..
    Signals calculated for YUM..
    Signals calculated for ZBH..
    Signals calculated for ZBRA..
    Signals calculated for ZTS..
    CPU times: user 3.91 s, sys: 1.54 s, total: 5.45 s
    Wall time: 5.47 s



```python
signals.shape
```




    (617702, 8)




```python
signals.tail(n = 20)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>price</th>
      <th>ticker</th>
      <th>daily_pct_change</th>
      <th>signal</th>
      <th>short</th>
      <th>long</th>
      <th>positions</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>617682</th>
      <td>2024-11-19</td>
      <td>175.559998</td>
      <td>ZTS</td>
      <td>-0.004875</td>
      <td>0.0</td>
      <td>182.164099</td>
      <td>184.629349</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>617683</th>
      <td>2024-11-20</td>
      <td>175.669998</td>
      <td>ZTS</td>
      <td>0.000627</td>
      <td>0.0</td>
      <td>181.695751</td>
      <td>184.558520</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>617684</th>
      <td>2024-11-21</td>
      <td>176.710007</td>
      <td>ZTS</td>
      <td>0.005920</td>
      <td>0.0</td>
      <td>181.258413</td>
      <td>184.508757</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>617685</th>
      <td>2024-11-22</td>
      <td>176.960007</td>
      <td>ZTS</td>
      <td>0.001415</td>
      <td>0.0</td>
      <td>180.836058</td>
      <td>184.488243</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>617686</th>
      <td>2024-11-25</td>
      <td>178.710007</td>
      <td>ZTS</td>
      <td>0.009889</td>
      <td>0.0</td>
      <td>180.386573</td>
      <td>184.494378</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>617687</th>
      <td>2024-11-26</td>
      <td>175.699997</td>
      <td>ZTS</td>
      <td>-0.016843</td>
      <td>0.0</td>
      <td>179.845400</td>
      <td>184.442904</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>617688</th>
      <td>2024-11-27</td>
      <td>176.740005</td>
      <td>ZTS</td>
      <td>0.005919</td>
      <td>0.0</td>
      <td>179.270722</td>
      <td>184.418282</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>617689</th>
      <td>2024-11-29</td>
      <td>175.250000</td>
      <td>ZTS</td>
      <td>-0.008430</td>
      <td>0.0</td>
      <td>178.760774</td>
      <td>184.374002</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>617690</th>
      <td>2024-12-02</td>
      <td>176.809998</td>
      <td>ZTS</td>
      <td>0.008902</td>
      <td>0.0</td>
      <td>178.227004</td>
      <td>184.334861</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>617691</th>
      <td>2024-12-03</td>
      <td>176.940002</td>
      <td>ZTS</td>
      <td>0.000735</td>
      <td>0.0</td>
      <td>177.824933</td>
      <td>184.307363</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>617692</th>
      <td>2024-12-04</td>
      <td>175.320007</td>
      <td>ZTS</td>
      <td>-0.009156</td>
      <td>0.0</td>
      <td>177.366868</td>
      <td>184.239805</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>617693</th>
      <td>2024-12-05</td>
      <td>174.770004</td>
      <td>ZTS</td>
      <td>-0.003137</td>
      <td>0.0</td>
      <td>176.907760</td>
      <td>184.166137</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>617694</th>
      <td>2024-12-06</td>
      <td>176.460007</td>
      <td>ZTS</td>
      <td>0.009670</td>
      <td>0.0</td>
      <td>176.754063</td>
      <td>184.131089</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>617695</th>
      <td>2024-12-09</td>
      <td>178.149994</td>
      <td>ZTS</td>
      <td>0.009577</td>
      <td>0.0</td>
      <td>176.706248</td>
      <td>184.088436</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>617696</th>
      <td>2024-12-10</td>
      <td>176.710007</td>
      <td>ZTS</td>
      <td>-0.008083</td>
      <td>0.0</td>
      <td>176.518983</td>
      <td>184.050068</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>617697</th>
      <td>2024-12-11</td>
      <td>177.169998</td>
      <td>ZTS</td>
      <td>0.002603</td>
      <td>0.0</td>
      <td>176.396601</td>
      <td>184.080772</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>617698</th>
      <td>2024-12-12</td>
      <td>178.839996</td>
      <td>ZTS</td>
      <td>0.009426</td>
      <td>0.0</td>
      <td>176.281001</td>
      <td>184.013972</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>617699</th>
      <td>2024-12-13</td>
      <td>178.179993</td>
      <td>ZTS</td>
      <td>-0.003690</td>
      <td>0.0</td>
      <td>176.261001</td>
      <td>183.945603</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>617700</th>
      <td>2024-12-16</td>
      <td>175.809998</td>
      <td>ZTS</td>
      <td>-0.013301</td>
      <td>0.0</td>
      <td>176.056334</td>
      <td>183.811771</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>617701</th>
      <td>2024-12-17</td>
      <td>170.539993</td>
      <td>ZTS</td>
      <td>-0.029976</td>
      <td>0.0</td>
      <td>175.901667</td>
      <td>183.653193</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
signals.set_index('Date', inplace = True)
```


```python
%%time

# Init df to store aggregate
backtest = pd.DataFrame()

# Iterate the trades to calculate earnings
for nm, grp in signals.groupby('ticker'):

    pos = pd.DataFrame(index = grp.index).fillna(0)

    # Trigger to purchase specificed shares of each stock
    pos['shares'] = 100 * grp['signal']

    portfolio = pos.multiply(grp['price'], axis = 0)
    pos_diff = pos.diff()

    # Add `holdings` to portfolio
    portfolio['holdings'] = (pos.multiply(grp['price'], axis = 0)).sum(axis = 1)

    # Add `cash` to portfolio
    portfolio['cash'] = 10000 - (pos_diff.multiply(grp['price'], axis = 0)).sum(axis = 1).cumsum()   

    # Add `total` to portfolio
    portfolio['total'] = portfolio['cash'] + portfolio['holdings']

    # Add `returns` to portfolio
    portfolio['returns'] = portfolio['total'].pct_change() 
    
    portfolio['ticker'] = nm
    
    backtest = pd.concat([backtest, portfolio])
```

    CPU times: user 4.56 s, sys: 2.35 s, total: 6.91 s
    Wall time: 6.99 s



```python
backtest.tail(n = 25)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>shares</th>
      <th>holdings</th>
      <th>cash</th>
      <th>total</th>
      <th>returns</th>
      <th>ticker</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2024-11-12</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>5868.318176</td>
      <td>5868.318176</td>
      <td>-0.035976</td>
      <td>ZTS</td>
    </tr>
    <tr>
      <th>2024-11-13</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>5868.318176</td>
      <td>5868.318176</td>
      <td>0.000000</td>
      <td>ZTS</td>
    </tr>
    <tr>
      <th>2024-11-14</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>5868.318176</td>
      <td>5868.318176</td>
      <td>0.000000</td>
      <td>ZTS</td>
    </tr>
    <tr>
      <th>2024-11-15</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>5868.318176</td>
      <td>5868.318176</td>
      <td>0.000000</td>
      <td>ZTS</td>
    </tr>
    <tr>
      <th>2024-11-18</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>5868.318176</td>
      <td>5868.318176</td>
      <td>0.000000</td>
      <td>ZTS</td>
    </tr>
    <tr>
      <th>2024-11-19</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>5868.318176</td>
      <td>5868.318176</td>
      <td>0.000000</td>
      <td>ZTS</td>
    </tr>
    <tr>
      <th>2024-11-20</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>5868.318176</td>
      <td>5868.318176</td>
      <td>0.000000</td>
      <td>ZTS</td>
    </tr>
    <tr>
      <th>2024-11-21</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>5868.318176</td>
      <td>5868.318176</td>
      <td>0.000000</td>
      <td>ZTS</td>
    </tr>
    <tr>
      <th>2024-11-22</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>5868.318176</td>
      <td>5868.318176</td>
      <td>0.000000</td>
      <td>ZTS</td>
    </tr>
    <tr>
      <th>2024-11-25</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>5868.318176</td>
      <td>5868.318176</td>
      <td>0.000000</td>
      <td>ZTS</td>
    </tr>
    <tr>
      <th>2024-11-26</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>5868.318176</td>
      <td>5868.318176</td>
      <td>0.000000</td>
      <td>ZTS</td>
    </tr>
    <tr>
      <th>2024-11-27</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>5868.318176</td>
      <td>5868.318176</td>
      <td>0.000000</td>
      <td>ZTS</td>
    </tr>
    <tr>
      <th>2024-11-29</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>5868.318176</td>
      <td>5868.318176</td>
      <td>0.000000</td>
      <td>ZTS</td>
    </tr>
    <tr>
      <th>2024-12-02</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>5868.318176</td>
      <td>5868.318176</td>
      <td>0.000000</td>
      <td>ZTS</td>
    </tr>
    <tr>
      <th>2024-12-03</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>5868.318176</td>
      <td>5868.318176</td>
      <td>0.000000</td>
      <td>ZTS</td>
    </tr>
    <tr>
      <th>2024-12-04</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>5868.318176</td>
      <td>5868.318176</td>
      <td>0.000000</td>
      <td>ZTS</td>
    </tr>
    <tr>
      <th>2024-12-05</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>5868.318176</td>
      <td>5868.318176</td>
      <td>0.000000</td>
      <td>ZTS</td>
    </tr>
    <tr>
      <th>2024-12-06</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>5868.318176</td>
      <td>5868.318176</td>
      <td>0.000000</td>
      <td>ZTS</td>
    </tr>
    <tr>
      <th>2024-12-09</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>5868.318176</td>
      <td>5868.318176</td>
      <td>0.000000</td>
      <td>ZTS</td>
    </tr>
    <tr>
      <th>2024-12-10</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>5868.318176</td>
      <td>5868.318176</td>
      <td>0.000000</td>
      <td>ZTS</td>
    </tr>
    <tr>
      <th>2024-12-11</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>5868.318176</td>
      <td>5868.318176</td>
      <td>0.000000</td>
      <td>ZTS</td>
    </tr>
    <tr>
      <th>2024-12-12</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>5868.318176</td>
      <td>5868.318176</td>
      <td>0.000000</td>
      <td>ZTS</td>
    </tr>
    <tr>
      <th>2024-12-13</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>5868.318176</td>
      <td>5868.318176</td>
      <td>0.000000</td>
      <td>ZTS</td>
    </tr>
    <tr>
      <th>2024-12-16</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>5868.318176</td>
      <td>5868.318176</td>
      <td>0.000000</td>
      <td>ZTS</td>
    </tr>
    <tr>
      <th>2024-12-17</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>5868.318176</td>
      <td>5868.318176</td>
      <td>0.000000</td>
      <td>ZTS</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Compile the performance from all stocks
performance = backtest.groupby(backtest.index).agg({'holdings': 'sum',
                                                    'cash': 'sum',
                                                    'total': 'sum'})
```


```python
performance.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>holdings</th>
      <th>cash</th>
      <th>total</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2020-01-02</th>
      <td>0.0</td>
      <td>4890000.0</td>
      <td>4890000.0</td>
    </tr>
    <tr>
      <th>2020-01-03</th>
      <td>0.0</td>
      <td>4890000.0</td>
      <td>4890000.0</td>
    </tr>
    <tr>
      <th>2020-01-06</th>
      <td>0.0</td>
      <td>4890000.0</td>
      <td>4890000.0</td>
    </tr>
    <tr>
      <th>2020-01-07</th>
      <td>0.0</td>
      <td>4890000.0</td>
      <td>4890000.0</td>
    </tr>
    <tr>
      <th>2020-01-08</th>
      <td>0.0</td>
      <td>4890000.0</td>
      <td>4890000.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
performance.tail()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>holdings</th>
      <th>cash</th>
      <th>total</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2024-12-11</th>
      <td>7.710354e+06</td>
      <td>-445815.955639</td>
      <td>7.264538e+06</td>
    </tr>
    <tr>
      <th>2024-12-12</th>
      <td>7.674541e+06</td>
      <td>-438761.955166</td>
      <td>7.235779e+06</td>
    </tr>
    <tr>
      <th>2024-12-13</th>
      <td>7.679022e+06</td>
      <td>-474822.955990</td>
      <td>7.204199e+06</td>
    </tr>
    <tr>
      <th>2024-12-16</th>
      <td>7.641005e+06</td>
      <td>-441268.955135</td>
      <td>7.199736e+06</td>
    </tr>
    <tr>
      <th>2024-12-17</th>
      <td>7.272233e+06</td>
      <td>-410232.415295</td>
      <td>6.862000e+06</td>
    </tr>
  </tbody>
</table>
</div>




```python
start = performance['total'][0]
end = performance['total'][-1]

print(f'Estimated Returns: {round(end / start, 2)}X')
```

    Estimated Returns: 1.4X

