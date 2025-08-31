"""수학/통계 유틸리티"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Union, Optional, Tuple
import numba

class MathUtils:
    """수학 계산 유틸리티"""
    
    @staticmethod
    @numba.jit(nopython=True, cache=True)
    def fast_ema(values: np.ndarray, span: int) -> np.ndarray:
        """빠른 지수이동평균 계산 (Numba 가속)"""
        alpha = 2.0 / (span + 1.0)
        ema = np.empty_like(values)
        ema[0] = values[0]
        
        for i in range(1, len(values)):
            ema[i] = alpha * values[i] + (1 - alpha) * ema[i-1]
        
        return ema
    
    @staticmethod
    @numba.jit(nopython=True, cache=True)
    def fast_std(values: np.ndarray, window: int) -> np.ndarray:
        """빠른 롤링 표준편차 계산"""
        n = len(values)
        stds = np.empty(n)
        stds[:window] = np.nan
        
        for i in range(window, n):
            window_vals = values[i-window+1:i+1]
            stds[i] = np.std(window_vals)
        
        return stds
    
    @staticmethod
    def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """RSI 계산"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """ATR (Average True Range) 계산"""
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        return atr
    
    @staticmethod
    def calculate_bollinger_bands(prices: pd.Series, period: int = 20, std_dev: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """볼린저 밴드 계산"""
        middle = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        
        upper = middle + (std_dev * std)
        lower = middle - (std_dev * std)
        
        return upper, middle, lower
    
    @staticmethod
    def calculate_vwap(prices: pd.Series, volumes: pd.Series) -> pd.Series:
        """VWAP (Volume Weighted Average Price) 계산"""
        return (prices * volumes).cumsum() / volumes.cumsum()
    
    @staticmethod
    def calculate_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """ADX (Average Directional Index) 계산"""
        plus_dm = high.diff()
        minus_dm = -low.diff()
        
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        
        tr = MathUtils.calculate_atr(high, low, close, 1)
        
        plus_di = 100 * (plus_dm.rolling(period).mean() / tr.rolling(period).mean())
        minus_di = 100 * (minus_dm.rolling(period).mean() / tr.rolling(period).mean())
        
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(period).mean()
        
        return adx
    
    @staticmethod
    def calculate_hurst_exponent(prices: pd.Series, max_lag: int = 100) -> float:
        """허스트 지수 계산 (트렌드 강도 측정)"""
        lags = range(2, max_lag)
        tau = []
        
        for lag in lags:
            returns = prices.pct_change(lag).dropna()
            tau.append(returns.std())
        
        # 로그-로그 회귀
        reg = np.polyfit(np.log(lags), np.log(tau), 1)
        return reg[0]
    
    @staticmethod
    def calculate_parkinson_volatility(high: pd.Series, low: pd.Series, period: int = 20) -> pd.Series:
        """파킨슨 변동성 계산"""
        hl_ratio = np.log(high / low)
        return np.sqrt(hl_ratio.rolling(period).apply(lambda x: np.sum(x**2) / (4 * period * np.log(2))))
    
    @staticmethod
    def z_score(value: float, mean: float, std: float) -> float:
        """Z-score 계산"""
        if std == 0:
            return 0
        return (value - mean) / std
    
    @staticmethod
    def kelly_criterion(win_prob: float, avg_win: float, avg_loss: float, fraction: float = 0.25) -> float:
        """켈리 기준 계산 (보수적 1/4 켈리)"""
        if avg_loss == 0:
            return 0
        
        b = avg_win / avg_loss
        p = win_prob
        q = 1 - p
        
        kelly = (p * b - q) / b
        return max(0, min(kelly * fraction, 0.3))  # 최대 30% 제한