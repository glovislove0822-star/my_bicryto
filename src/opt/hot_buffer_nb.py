"""Numba 가속 핫 버퍼"""

import numpy as np
import numba as nb
from numba import jit, njit, prange
from typing import Optional

@njit(parallel=True, cache=True)
def rolling_mean_nb(arr: np.ndarray, window: int) -> np.ndarray:
    """Numba 가속 롤링 평균"""
    
    n = len(arr)
    result = np.empty(n)
    result[:window-1] = np.nan
    
    for i in prange(window-1, n):
        result[i] = np.mean(arr[i-window+1:i+1])
    
    return result

@njit(parallel=True, cache=True)
def rolling_std_nb(arr: np.ndarray, window: int) -> np.ndarray:
    """Numba 가속 롤링 표준편차"""
    
    n = len(arr)
    result = np.empty(n)
    result[:window-1] = np.nan
    
    for i in prange(window-1, n):
        result[i] = np.std(arr[i-window+1:i+1])
    
    return result

@njit(cache=True)
def ema_nb(arr: np.ndarray, span: int) -> np.ndarray:
    """Numba 가속 EMA"""
    
    n = len(arr)
    result = np.empty(n)
    alpha = 2.0 / (span + 1)
    
    result[0] = arr[0]
    
    for i in range(1, n):
        result[i] = alpha * arr[i] + (1 - alpha) * result[i-1]
    
    return result

@njit(cache=True)
def rsi_nb(arr: np.ndarray, period: int) -> np.ndarray:
    """Numba 가속 RSI"""
    
    n = len(arr)
    result = np.empty(n)
    result[:period] = np.nan
    
    deltas = np.diff(arr)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    
    for i in range(period, n-1):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        
        if avg_loss == 0:
            result[i+1] = 100
        else:
            rs = avg_gain / avg_loss
            result[i+1] = 100 - (100 / (1 + rs))
    
    return result

class HotBuffer:
    """고성능 링 버퍼"""
    
    def __init__(self, capacity: int):
        """
        Args:
            capacity: 버퍼 용량
        """
        self.capacity = capacity
        self.buffer = np.empty(capacity)
        self.position = 0
        self.size = 0
    
    def append(self, value: float):
        """값 추가"""
        
        self.buffer[self.position] = value
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def get_array(self) -> np.ndarray:
        """배열 조회"""
        
        if self.size < self.capacity:
            return self.buffer[:self.size]
        else:
            # 순환 버퍼 정렬
            return np.concatenate([
                self.buffer[self.position:],
                self.buffer[:self.position]
            ])
    
    def rolling_mean(self, window: int) -> Optional[float]:
        """롤링 평균"""
        
        if self.size < window:
            return None
        
        arr = self.get_array()
        return rolling_mean_nb(arr, window)[-1]
    
    def rolling_std(self, window: int) -> Optional[float]:
        """롤링 표준편차"""
        
        if self.size < window:
            return None
        
        arr = self.get_array()
        return rolling_std_nb(arr, window)[-1]