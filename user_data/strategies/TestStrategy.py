# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
# --- Do not remove these libs ---
import numpy as np
import pandas as pd
from pandas import DataFrame
from datetime import datetime
from typing import Optional, Union

from freqtrade.strategy import (
    BooleanParameter,
    CategoricalParameter,
    DecimalParameter,
    IntParameter,
    IStrategy,
    merge_informative_pair,
)

# --------------------------------
# Add your lib to import here
import talib.abstract as ta
import pandas_ta as pta
from technical import qtpylib


class TestStrategy(IStrategy):
    """
    This is a strategy template to get you started.
    More information in https://www.freqtrade.io/en/latest/strategy-customization/

    You can:
        :return: a Dataframe with all mandatory indicators for the strategies
    - Rename the class name (Do not forget to update class_name)
    - Add any methods you want to build your strategy
    - Add any lib you need to build your strategy

    You must keep:
    - the lib in the section "Do not remove these libs"
    - the methods: populate_indicators, populate_entry_trend, populate_exit_trend
    You should keep:
    - timeframe, minimal_roi, stoploss, trailing_*
    """

    # Strategy interface version - allow new iterations of the strategy interface.
    # Check the documentation or the Sample strategy to get the latest version.
    INTERFACE_VERSION = 3

    # Optimal timeframe for the strategy.
    timeframe = "1h"

    # Can this strategy go short?
    can_short: bool = False

    # Minimal ROI designed for the strategy.
    # This attribute will be overridden if the config file contains "minimal_roi".
    minimal_roi = {"60": 0.04, "30": 0.02, "0": 0.1}

    # Optimal stoploss designed for the strategy.
    # This attribute will be overridden if the config file contains "stoploss".
    stoploss = -0.02

    # Trailing stoploss
    trailing_stop = False
    # trailing_only_offset_is_reached = False
    # trailing_stop_positive = 0.01
    # trailing_stop_positive_offset = 0.0  # Disabled / not configured

    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = True

    # These values can be overridden in the config.
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 30

    # Strategy parameters
    buy_rsi = IntParameter(10, 40, default=30, space="buy")
    sell_rsi = IntParameter(70, 90, default=80, space="sell")

    # Optional order type mapping.
    order_types = {
        "entry": "limit",
        "exit": "limit",
        "stoploss": "market",
        "stoploss_on_exchange": False,
    }

    # Optional order time in force.
    order_time_in_force = {"entry": "GTC", "exit": "GTC"}

    @property
    def plot_config(self):
        return {
            # Main plot indicators (Moving averages, ...)
            "main_plot": {
                "tema": {},
                "sar": {"color": "white"},
            },
            "subplots": {
                # Subplots - each dict defines one additional plot
                "MACD": {
                    "macd": {"color": "blue"},
                    "macdsignal": {"color": "orange"},
                },
                "RSI": {
                    "rsi": {"color": "red"},
                },
            },
        }

    def informative_pairs(self):
        """
        Define additional, informative pair/interval combinations to be cached from the exchange.
        These pair/interval combinations are non-tradeable, unless they are part
        of the whitelist as well.
        For more information, please consult the documentation
        :return: List of tuples in the format (pair, interval)
            Sample: return [("ETH/USDT", "5m"),
                            ("BTC/USDT", "15m"),
                            ]
        """
        return []

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        rsiLen = 14
        smoothLen = 2


        inputMfiLen = 30
        smoothHist = 2
        fastMfiLen = round(inputMfiLen / 1.33)
        slowMfiLen = round(inputMfiLen * 1.33)


        MFI_LEN     = 7
        STOCH_K     = 2
        STOCH_D     = 5
        SMOOTH_LEN  = 1.75
        STOCH_WEIGHT = 0.4
        OVERBOUGHT   = 60.0
        EXTEND_MULT  = 1
        mfiWeight    = 0.4


        rsiLen = 14

        UP_BORDER = 50
        DN_BORDER = -50
        lastSigBar = 0

        def transform(src, mult=1):
            tmp = (src / 100 - 0.5) * 2
            sign = np.where(tmp > 0, 1, -1)
            return mult * 100 * sign * np.power(np.abs(tmp), 0.75)

        def pivot_high(series, left_bars, right_bars):
            """检查给定点是否是一个 pivot high，即它比左边和右边的 left_bars 和 right_bars 数量的点都高。"""
            pivots = [np.nan] * len(series)
            for i in range(left_bars, len(series) - right_bars):
                is_pivot = True
                for j in range(1, left_bars + 1):
                    if series[i] <= series[i - j]:
                        is_pivot = False
                        break
                for j in range(1, right_bars + 1):
                    if series[i] <= series[i + j]:
                        is_pivot = False
                        break
                if is_pivot:
                    pivots[i+right_bars] = series[i]
            return pivots

        def pivot_low(series, left_bars, right_bars):
            """检查给定点是否是一个 pivot low，即它比左边和右边的 left_bars 和 right_bars 数量的点都低。"""
            pivots = [np.nan] * len(series)
            for i in range(left_bars, len(series) - right_bars):
                is_pivot = True
                for j in range(1, left_bars + 1):
                    if series[i] >= series[i - j]:
                        is_pivot = False
                        break
                for j in range(1, right_bars + 1):
                    if series[i] >= series[i + j]:
                        is_pivot = False
                        break
                if is_pivot:
                    pivots[i+right_bars] = series[i]
            return pivots


        def calc(df):
            df = df.copy()
            fastMfi = ta.MFI(df['high'], df['low'], df['close'], df['volume'], timeperiod=fastMfiLen)
            slowMfi = ta.MFI(df['high'], df['low'], df['close'], df['volume'], timeperiod=slowMfiLen)
            resMfi = transform(ta.SMA((fastMfi * 0.5 + slowMfi * 0.5), timeperiod=smoothHist), 0.7)

            mfi = ta.MFI(df['high'], df['low'], df['close'], df['volume'], timeperiod=MFI_LEN)
            rsi = ta.RSI(df['hlc3'], timeperiod=rsiLen)
            # 计算Stochastic RSI
            stoch_rsi_k, stoch_rsi_d = ta.STOCH(rsi, rsi, rsi, fastk_period=rsiLen, slowk_period=1, slowk_matype=0, slowd_period=1, slowd_matype=0)
            # 计算STOCH_K的SMA
            stoch = ta.SMA(stoch_rsi_k, timeperiod=STOCH_K)
            sigStoch = ta.SMA(stoch, timeperiod=STOCH_D)
            df["mfi"] = mfi
            df["rsi"] = rsi
            df["stoch"] = stoch
            df["sigStoch"] = sigStoch
            
            # ---
            signal = (rsi + mfiWeight * mfi + STOCH_WEIGHT * stoch) / (1 + mfiWeight + STOCH_WEIGHT)
            avg = transform(ta.EMA(signal, smoothLen), EXTEND_MULT)
            avg2 = transform(ta.EMA(signal, round(smoothLen * SMOOTH_LEN)), EXTEND_MULT)

            df['resMfi'] = resMfi
            df['signal'] = signal
            df['avg'] = avg
            df['avg2'] = avg2
            
            # ----
            df["indPh"] = pivot_high(avg, 5, 5)
            df["indPl"] = pivot_low(avg, 5, 5)
            
            # ---
            # Initialize last pivot high and low
            lastIndPh_price = np.nan
            lastIndPh_ndx = np.nan
            lastIndPl_price = np.nan
            lastIndPl_ndx = np.nan

            # Calculate speedH and speedL
            speedH = np.full(avg.shape, np.nan)
            speedL = np.full(avg.shape, np.nan)

            for i in range(len(avg)):
                if not np.isnan(df["indPh"][i]):
                    lastIndPh_price = df["indPh"][i]
                    lastIndPh_ndx = i - 5
                if not np.isnan(df["indPl"][i]):
                    lastIndPl_price = df["indPl"][i]
                    lastIndPl_ndx = i - 5

                if not np.isnan(lastIndPh_price) and not np.isnan(lastIndPh_ndx):
                    speedH[i] = (avg[i] - lastIndPh_price) / (i - lastIndPh_ndx)
                if not np.isnan(lastIndPl_price) and not np.isnan(lastIndPl_ndx):
                    speedL[i] = (avg[i] - lastIndPl_price) / (i - lastIndPl_ndx)

            # 将速度结果添加到 DataFrame
            df['speedH'] = speedH
            df['speedL'] = speedL

            # ---
            sellSigRule = [False] * len(df)
            buySigRule = [False] * len(df)
            lastSigBar = 0

            # 遍历 DataFrame 计算信号规则
            for i in range(2, len(df)):
                if df['avg'][i] > UP_BORDER and df['avg'][i] > df['avg'][i-2] and df['speedH'][i] < df['speedH'][i-1] and (i - lastSigBar >= 10):
                    sellSigRule[i] = True
                    lastSigBar = i

                if df['avg'][i] < DN_BORDER and df['avg'][i] < df['avg'][i-2] and df['speedL'][i] > df['speedL'][i-1] and (i - lastSigBar >= 10):
                    buySigRule[i] = True
                    lastSigBar = i

            # 将信号规则添加到 DataFrame
            df['sellSigRule'] = sellSigRule
            df['buySigRule'] = buySigRule
            return df

        dataframe["hlc3"] = (dataframe["high"] + dataframe["low"] + dataframe["close"]) / 3
        dataframe = calc(dataframe)
        return dataframe


    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the entry signal for the given dataframe
        :param dataframe: DataFrame
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with entry columns populated
        """
        dataframe.loc[(dataframe["buySigRule"] == True), "enter_long"] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the exit signal for the given dataframe
        :param dataframe: DataFrame
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with exit columns populated
        """
        dataframe.loc[(dataframe["sellSigRule"] == True), "exit_long"] = 1
        return dataframe
