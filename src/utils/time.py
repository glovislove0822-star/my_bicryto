"""시간 관련 유틸리티"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import Union, Optional

class TimeUtils:
    """시간 처리 유틸리티 클래스"""
    
    @staticmethod
    def to_unix_ms(dt: Union[datetime, pd.Timestamp, str]) -> int:
        """datetime을 Unix 밀리초로 변환"""
        if isinstance(dt, str):
            dt = pd.to_datetime(dt)
        elif isinstance(dt, datetime):
            dt = pd.Timestamp(dt)
        return int(dt.timestamp() * 1000)
    
    @staticmethod
    def from_unix_ms(unix_ms: int) -> pd.Timestamp:
        """Unix 밀리초를 pandas Timestamp로 변환"""
        return pd.Timestamp(unix_ms, unit='ms', tz='UTC')
    
    @staticmethod
    def get_bar_timestamps(start: datetime, end: datetime, bar_size: str) -> pd.DatetimeIndex:
        """지정된 기간의 바 타임스탬프 생성"""
        freq_map = {
            '1m': '1T',
            '3m': '3T', 
            '5m': '5T',
            '15m': '15T',
            '1h': '1H',
            '4h': '4H',
            '1d': '1D'
        }
        return pd.date_range(start=start, end=end, freq=freq_map.get(bar_size, '1T'))
    
    @staticmethod
    def get_session_hour(ts: pd.Timestamp) -> int:
        """UTC 시간대 세션 시간 반환 (0-23)"""
        return ts.hour
    
    @staticmethod
    def get_day_of_week(ts: pd.Timestamp) -> int:
        """요일 반환 (0=월요일, 6=일요일)"""
        return ts.dayofweek
    
    @staticmethod
    def is_asian_session(ts: pd.Timestamp) -> bool:
        """아시아 세션 여부 (UTC 0-8시)"""
        hour = ts.hour
        return 0 <= hour < 8
    
    @staticmethod
    def is_european_session(ts: pd.Timestamp) -> bool:
        """유럽 세션 여부 (UTC 7-16시)"""
        hour = ts.hour
        return 7 <= hour < 16
    
    @staticmethod
    def is_american_session(ts: pd.Timestamp) -> bool:
        """미국 세션 여부 (UTC 13-22시)"""
        hour = ts.hour
        return 13 <= hour < 22
    
    @staticmethod
    def get_rolling_window_start(end: datetime, window_days: int) -> datetime:
        """롤링 윈도우 시작 시간 계산"""
        return end - timedelta(days=window_days)