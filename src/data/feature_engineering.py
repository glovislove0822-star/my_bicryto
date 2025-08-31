"""피처 엔지니어링 모듈"""

import pandas as pd
import numpy as np
import duckdb
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
import logging
from scipy import stats
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from ..utils.logging import Logger
from ..utils.math import MathUtils
from ..utils.time import TimeUtils
from .ofi import OFICalculator

logger = Logger.get_logger(__name__)

class FeatureEngineer:
    """피처 엔지니어링 클래스"""
    
    def __init__(self, db_path: str = "data/trading.db"):
        """
        Args:
            db_path: DuckDB 데이터베이스 경로
        """
        self.db_path = Path(db_path)
        self.conn = duckdb.connect(str(self.db_path))
        self.math_utils = MathUtils()
        self.time_utils = TimeUtils()
        self.ofi_calc = OFICalculator(db_path)
        
        # 피처 설정
        self.feature_config = {
            'direction': {
                'ret_periods': [60, 120, 180],  # 분
                'ema_fast': 10,
                'ema_slow': 30,
                'donchian_period': 50,
                'adx_period': 14
            },
            'entry': {
                'rsi_periods': [2, 3, 6],
                'bb_period': 20,
                'bb_std': 2.0
            },
            'risk': {
                'atr_period': 14,
                'vol_periods': [20, 50],
                'parkinson_period': 20
            },
            'market_micro': {
                'ofi_period': 20,
                'depth_levels': 5
            }
        }
    
    def generate_features(self, 
                         timeframe: str = '3m',
                         symbols: Optional[List[str]] = None,
                         start_date: Optional[str] = None,
                         end_date: Optional[str] = None):
        """전체 피처 생성 파이프라인
        
        Args:
            timeframe: 시간 프레임
            symbols: 심볼 리스트
            start_date: 시작 날짜
            end_date: 종료 날짜
        """
        
        # 심볼 목록
        if symbols is None:
            result = self.conn.execute(f"""
                SELECT DISTINCT symbol 
                FROM trading.bars_{timeframe}
            """).fetchall()
            symbols = [r[0] for r in result]
        
        logger.info(f"피처 생성 시작: {symbols}")
        logger.info(f"Timeframe: {timeframe}")
        
        for symbol in tqdm(symbols, desc="피처 생성"):
            try:
                # 1. 기본 데이터 로드
                df = self._load_base_data(symbol, timeframe, start_date, end_date)
                
                if df.empty or len(df) < 200:
                    logger.warning(f"{symbol} 데이터 부족, 스킵")
                    continue
                
                # 2. 방향성 피처
                df = self._add_directional_features(df)
                
                # 3. 엔트리 피처
                df = self._add_entry_features(df)
                
                # 4. 마켓 마이크로구조 피처
                df = self._add_market_microstructure_features(df, symbol)
                
                # 5. 리스크/변동성 피처
                df = self._add_risk_features(df)
                
                # 6. 펀딩/캐리 피처
                df = self._add_funding_features(df, symbol)
                
                # 7. 시간 피처
                df = self._add_time_features(df)
                
                # 8. 레짐 관련 피처 (v2.0)
                df = self._add_regime_features(df)
                
                # 9. 피처 정규화 및 클리닝
                df = self._clean_features(df)
                
                # 10. DB 저장
                self._save_features(df, symbol, timeframe)
                
            except Exception as e:
                logger.error(f"{symbol} 피처 생성 실패: {e}")
                continue
        
        logger.info("피처 생성 완료!")
        self._print_feature_stats(timeframe)
    
    def _load_base_data(self, 
                       symbol: str, 
                       timeframe: str,
                       start_date: Optional[str] = None,
                       end_date: Optional[str] = None) -> pd.DataFrame:
        """기본 데이터 로드"""
        
        where_clause = f"WHERE symbol = '{symbol}'"
        if start_date:
            where_clause += f" AND open_time >= '{start_date}'"
        if end_date:
            where_clause += f" AND open_time <= '{end_date}'"
        
        query = f"""
            SELECT 
                symbol,
                open_time as ts,
                open,
                high,
                low,
                close,
                volume,
                quote_volume,
                trade_count,
                vwap
            FROM trading.bars_{timeframe}
            {where_clause}
            ORDER BY open_time
        """
        
        df = self.conn.execute(query).df()
        df['ts'] = pd.to_datetime(df['ts'])
        df.set_index('ts', inplace=True)
        
        logger.debug(f"{symbol} 데이터 로드: {len(df)} rows")
        return df
    
    def _add_directional_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """방향성 피처 추가"""
        
        config = self.feature_config['direction']
        
        # 수익률 피처
        for period in config['ret_periods']:
            bars_back = period // 3  # 3분봉 기준
            df[f'ret_{period}'] = df['close'].pct_change(bars_back)
        
        # EMA 피처
        df['ema_fast'] = df['close'].ewm(span=config['ema_fast'], adjust=False).mean()
        df['ema_slow'] = df['close'].ewm(span=config['ema_slow'], adjust=False).mean()
        
        # EMA 기울기
        df['ema_slope_fast'] = df['ema_fast'].diff() / df['ema_fast'].shift(1)
        df['ema_slope_slow'] = df['ema_slow'].diff() / df['ema_slow'].shift(1)
        
        # EMA 크로스
        df['ema_cross'] = np.where(df['ema_fast'] > df['ema_slow'], 1, -1)
        
        # Donchian 채널
        period = config['donchian_period']
        df['donchian_upper'] = df['high'].rolling(period).max()
        df['donchian_lower'] = df['low'].rolling(period).min()
        df['donchian_middle'] = (df['donchian_upper'] + df['donchian_lower']) / 2
        df['donchian_width'] = df['donchian_upper'] - df['donchian_lower']
        
        # Donchian 돌파 신호
        df['donchian_signal'] = 0
        df.loc[df['close'] > df['donchian_upper'].shift(1), 'donchian_signal'] = 1
        df.loc[df['close'] < df['donchian_lower'].shift(1), 'donchian_signal'] = -1
        
        # ADX (트렌드 강도)
        df['adx'] = self.math_utils.calculate_adx(
            df['high'], 
            df['low'], 
            df['close'],
            config['adx_period']
        )
        
        # Hurst Exponent (트렌드 지속성)
        if len(df) > 100:
            hurst_values = []
            for i in range(100, len(df)):
                window = df['close'].iloc[i-100:i]
                hurst = self.math_utils.calculate_hurst_exponent(window, max_lag=50)
                hurst_values.append(hurst)
            
            df['hurst_exp'] = np.nan
            df.iloc[100:, df.columns.get_loc('hurst_exp')] = hurst_values
        else:
            df['hurst_exp'] = np.nan
        
        # TSMOM (Time Series Momentum)
        df['tsmom_60'] = np.sign(df['ret_60'])
        df['tsmom_120'] = np.sign(df['ret_120'])
        
        return df
    
    def _add_entry_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """엔트리 타이밍 피처 추가"""
        
        config = self.feature_config['entry']
        
        # RSI 피처
        for period in config['rsi_periods']:
            df[f'rsi_{period}'] = self.math_utils.calculate_rsi(df['close'], period)
        
        # Bollinger Bands
        upper, middle, lower = self.math_utils.calculate_bollinger_bands(
            df['close'],
            config['bb_period'],
            config['bb_std']
        )
        
        df['bb_upper'] = upper
        df['bb_middle'] = middle
        df['bb_lower'] = lower
        df['bb_width'] = upper - lower
        df['bb_width_ratio'] = df['bb_width'] / df['bb_middle']
        
        # BB 포지션
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        df['bb_position'] = df['bb_position'].clip(0, 1)
        
        # VWAP
        df['vwap'] = self.math_utils.calculate_vwap(df['close'], df['volume'])
        
        # VWAP Z-score
        vwap_std = df['vwap'].rolling(20).std()
        df['vwap_z'] = (df['close'] - df['vwap']) / vwap_std
        df['vwap_z'] = df['vwap_z'].fillna(0)
        
        # 되돌림 신호
        df['pullback_long'] = ((df['rsi_2'] < 30) & 
                               (df['close'] > df['ema_slow']) & 
                               (df['vwap_z'] < -0.5)).astype(int)
        
        df['pullback_short'] = ((df['rsi_2'] > 70) & 
                                (df['close'] < df['ema_slow']) & 
                                (df['vwap_z'] > 0.5)).astype(int)
        
        return df
    
    def _add_market_microstructure_features(self, 
                                           df: pd.DataFrame,
                                           symbol: str) -> pd.DataFrame:
        """마켓 마이크로구조 피처 추가"""
        
        # OFI 계산 (별도 모듈 사용)
        ofi_data = self.ofi_calc.calculate_ofi(symbol, df.index[0], df.index[-1])
        
        if not ofi_data.empty:
            # OFI 병합
            ofi_data.set_index('ts', inplace=True)
            df = df.join(ofi_data[['ofi', 'queue_imbalance', 'spread_bps', 'depth_total']], how='left')
            
            # OFI Z-score
            df['ofi_z'] = (df['ofi'] - df['ofi'].rolling(20).mean()) / df['ofi'].rolling(20).std()
        else:
            # 기본값
            df['ofi'] = 0
            df['ofi_z'] = 0
            df['queue_imbalance'] = 0.5
            df['spread_bps'] = 1.0
            df['depth_total'] = 10000
        
        # Trade Intensity (체결 강도)
        df['trade_intensity'] = df['trade_count'] / df['trade_count'].rolling(20).mean()
        df['trade_intensity'] = df['trade_intensity'].fillna(1)
        
        # Volume Imbalance
        # taker_buy_volume 데이터가 있다면 사용
        if 'taker_buy_volume' in df.columns:
            df['vol_imbalance'] = df['taker_buy_volume'] / df['volume']
        else:
            # 가격 방향으로 추정
            df['vol_imbalance'] = np.where(df['close'] > df['open'], 0.55, 0.45)
        
        # Liquidity Score (종합 유동성 점수)
        df['liquidity_score'] = (
            (1 / (df['spread_bps'] / 100 + 0.001)) * 0.3 +  # 스프레드 (낮을수록 좋음)
            (df['depth_total'] / df['depth_total'].rolling(50).mean()) * 0.3 +  # 심도
            (df['volume'] / df['volume'].rolling(50).mean()) * 0.2 +  # 볼륨
            (df['trade_count'] / df['trade_count'].rolling(50).mean()) * 0.2  # 체결 빈도
        )
        
        # 유동성 포켓 (가격 레벨별 밀집도)
        df['price_level'] = (df['close'] / 10).round() * 10  # 10달러 단위로 반올림
        price_counts = df.groupby('price_level').size()
        df['liquidity_pocket'] = df['price_level'].map(price_counts)
        df['liquidity_pocket_ratio'] = df['liquidity_pocket'] / df['liquidity_pocket'].rolling(100).mean()
        
        return df
    
    def _add_risk_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """리스크/변동성 피처 추가"""
        
        config = self.feature_config['risk']
        
        # ATR
        df['atr'] = self.math_utils.calculate_atr(
            df['high'],
            df['low'], 
            df['close'],
            config['atr_period']
        )
        
        # 파킨슨 변동성
        df['parkinson_vol'] = self.math_utils.calculate_parkinson_volatility(
            df['high'],
            df['low'],
            config['parkinson_period']
        )
        
        # 실현 변동성
        for period in config['vol_periods']:
            returns = df['close'].pct_change()
            df[f'realized_vol_{period}'] = returns.rolling(period).std() * np.sqrt(252 * 24 * 60 / 3)  # 연율화
        
        df['realized_vol'] = df['realized_vol_20']  # 기본값
        
        # 변동성 클러스터링
        df['vol_ratio'] = df['realized_vol_20'] / df['realized_vol_50']
        df['vol_cluster'] = np.where(df['vol_ratio'] > 1.5, 1, 
                                     np.where(df['vol_ratio'] < 0.7, -1, 0))
        
        # 가격 범위
        df['hl_ratio'] = df['high'] / df['low'] - 1
        df['oc_ratio'] = abs(df['close'] - df['open']) / df['open']
        
        # Tail Risk (극단 움직임)
        returns = df['close'].pct_change()
        df['left_tail'] = returns.rolling(100).quantile(0.05)
        df['right_tail'] = returns.rolling(100).quantile(0.95)
        df['tail_ratio'] = abs(df['left_tail'] / df['right_tail'])
        
        # Maximum Adverse Excursion (MAE) 근사
        df['mae_long'] = (df['low'] - df['open']) / df['open']
        df['mae_short'] = (df['high'] - df['open']) / df['open']
        
        return df
    
    def _add_funding_features(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """펀딩 레이트 피처 추가"""
        
        # 펀딩 데이터 로드
        funding_query = f"""
            SELECT 
                ts,
                funding_rate,
                mark_price
            FROM trading.funding
            WHERE symbol = '{symbol}'
                AND ts >= '{df.index[0]}'
                AND ts <= '{df.index[-1]}'
            ORDER BY ts
        """
        
        funding_df = self.conn.execute(funding_query).df()
        
        if not funding_df.empty:
            funding_df['ts'] = pd.to_datetime(funding_df['ts'])
            funding_df.set_index('ts', inplace=True)
            
            # 가장 가까운 시간으로 병합
            df = pd.merge_asof(
                df.sort_index(),
                funding_df.sort_index(),
                left_index=True,
                right_index=True,
                direction='backward'
            )
            
            # 펀딩 통계
            df['funding_ma_8h'] = df['funding_rate'].rolling(1, min_periods=1).mean()  # 8시간마다 펀딩
            df['funding_ma_24h'] = df['funding_rate'].rolling(3, min_periods=1).mean()  # 24시간 = 3 펀딩
            df['funding_std'] = df['funding_rate'].rolling(10, min_periods=1).std()
            
            # 펀딩 Z-score
            df['funding_z'] = (df['funding_rate'] - df['funding_ma_24h']) / df['funding_std']
            df['funding_z'] = df['funding_z'].fillna(0)
            
            # 펀딩 편향
            df['funding_bias'] = np.where(df['funding_rate'] > 0.001, 1,
                                          np.where(df['funding_rate'] < -0.001, -1, 0))
            
            # 누적 펀딩
            df['cumulative_funding'] = df['funding_rate'].cumsum()
            
        else:
            # 기본값
            df['funding_rate'] = 0
            df['funding_ma_8h'] = 0
            df['funding_ma_24h'] = 0
            df['funding_std'] = 0.0001
            df['funding_z'] = 0
            df['funding_bias'] = 0
            df['cumulative_funding'] = 0
        
        return df
    
    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """시간 관련 피처 추가"""
        
        # 기본 시간 피처
        df['hour_utc'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        
        # 세션 피처
        df['is_asian_session'] = df.index.map(self.time_utils.is_asian_session)
        df['is_european_session'] = df.index.map(self.time_utils.is_european_session)
        df['is_american_session'] = df.index.map(self.time_utils.is_american_session)
        
        # 시간대별 변동성 (역사적)
        hourly_vol = df.groupby('hour_utc')['realized_vol'].mean()
        df['hour_vol_ratio'] = df['hour_utc'].map(hourly_vol) / hourly_vol.mean()
        
        # 요일별 패턴
        dow_returns = df.groupby('day_of_week')['close'].pct_change().mean()
        df['dow_bias'] = df['day_of_week'].map(dow_returns)
        
        # 월말/월초 효과
        df['is_month_start'] = df.index.day <= 3
        df['is_month_end'] = df.index.day >= 27
        
        # 주말 효과 (금요일 오후 ~ 월요일 오전)
        df['is_weekend_risk'] = ((df['day_of_week'] == 4) & (df['hour_utc'] >= 20)) | \
                                ((df['day_of_week'] == 0) & (df['hour_utc'] <= 4))
        
        return df
    
    def _add_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """마켓 레짐 관련 피처 추가 (v2.0)"""
        
        # 변동성 레짐
        vol_percentiles = df['realized_vol'].rolling(500).quantile([0.25, 0.5, 0.75])
        
        df['vol_regime'] = pd.cut(
            df['realized_vol'],
            bins=[-np.inf, 
                  df['realized_vol'].quantile(0.25),
                  df['realized_vol'].quantile(0.5),
                  df['realized_vol'].quantile(0.75),
                  np.inf],
            labels=['low', 'normal', 'high', 'extreme']
        )
        
        # 트렌드 레짐
        df['trend_strength'] = abs(df['ema_slope_slow']) * df['adx']
        df['trend_regime'] = pd.cut(
            df['trend_strength'],
            bins=5,
            labels=['strong_down', 'weak_down', 'neutral', 'weak_up', 'strong_up']
        )
        
        # 유동성 레짐
        df['liquidity_regime'] = pd.cut(
            df['liquidity_score'],
            bins=3,
            labels=['thin', 'normal', 'deep']
        )
        
        # 레짐 전환 감지
        df['vol_regime_change'] = (df['vol_regime'] != df['vol_regime'].shift(1)).astype(int)
        df['trend_regime_change'] = (df['trend_regime'] != df['trend_regime'].shift(1)).astype(int)
        
        # 레짐 지속 기간
        df['vol_regime_duration'] = df.groupby((df['vol_regime'] != df['vol_regime'].shift()).cumsum()).cumcount()
        df['trend_regime_duration'] = df.groupby((df['trend_regime'] != df['trend_regime'].shift()).cumsum()).cumcount()
        
        return df
    
    def _clean_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """피처 정리 및 정규화"""
        
        # NaN 처리
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        # Inf 처리
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(0)
        
        # 이상치 클리핑 (IQR 방법)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col not in ['symbol', 'open', 'high', 'low', 'close', 'volume']:
                Q1 = df[col].quantile(0.01)
                Q3 = df[col].quantile(0.99)
                df[col] = df[col].clip(Q1, Q3)
        
        # 범주형 변수 인코딩
        if 'vol_regime' in df.columns:
            df['vol_regime_encoded'] = pd.Categorical(df['vol_regime']).codes
        
        if 'trend_regime' in df.columns:
            df['trend_regime_encoded'] = pd.Categorical(df['trend_regime']).codes
            
        if 'liquidity_regime' in df.columns:
            df['liquidity_regime_encoded'] = pd.Categorical(df['liquidity_regime']).codes
        
        return df
    
    def _save_features(self, df: pd.DataFrame, symbol: str, timeframe: str):
        """피처 데이터 저장"""
        
        # symbol 컬럼 추가
        df['symbol'] = symbol
        
        # 인덱스를 ts 컬럼으로
        df.reset_index(inplace=True)
        df.rename(columns={'index': 'ts'}, inplace=True)
        
        # 저장할 컬럼 선택
        feature_columns = [
            'symbol', 'ts',
            # 방향성
            'ret_60', 'ret_120', 'ema_fast', 'ema_slow',
            'ema_slope_fast', 'ema_slope_slow',
            'donchian_upper', 'donchian_lower', 'donchian_signal',
            'adx', 'hurst_exp',
            # 엔트리
            'rsi_2', 'rsi_3', 'vwap', 'vwap_z',
            'bb_upper', 'bb_lower', 'bb_width',
            # 마켓 마이크로
            'ofi', 'ofi_z', 'queue_imbalance', 'spread_bps',
            'depth_total', 'trade_intensity', 'liquidity_score',
            # 리스크
            'atr', 'parkinson_vol', 'realized_vol', 'vol_cluster',
            # 펀딩
            'funding_rate', 'funding_ma_8h', 'funding_ma_24h', 'funding_std',
            # 시간
            'hour_utc', 'day_of_week', 'is_asian_session',
            'is_european_session', 'is_american_session'
        ]
        
        # 존재하는 컬럼만 선택
        save_cols = [col for col in feature_columns if col in df.columns]
        df_save = df[save_cols].copy()
        
        # DuckDB에 저장 (UPSERT)
        table_name = f"features_{timeframe}"
        
        # 임시 테이블 생성
        temp_table = f"temp_features_{int(datetime.now().timestamp())}"
        self.conn.execute(f"CREATE TEMP TABLE {temp_table} AS SELECT * FROM df_save")
        
        # UPSERT
        self.conn.execute(f"""
            INSERT OR REPLACE INTO trading.{table_name}
            SELECT * FROM {temp_table}
        """)
        
        # 임시 테이블 삭제
        self.conn.execute(f"DROP TABLE {temp_table}")
        
        logger.info(f"{symbol} 피처 저장 완료: {len(df_save)} rows")
    
    def _print_feature_stats(self, timeframe: str):
        """피처 통계 출력"""
        
        stats = self.conn.execute(f"""
            SELECT 
                symbol,
                COUNT(*) as row_count,
                MIN(ts) as start_time,
                MAX(ts) as end_time,
                COUNT(DISTINCT DATE(ts)) as days
            FROM trading.features_{timeframe}
            GROUP BY symbol
        """).fetchall()
        
        logger.info(f"\n=== {timeframe} 피처 통계 ===")
        for symbol, count, start, end, days in stats:
            logger.info(f"{symbol}:")
            logger.info(f"  레코드: {count:,}")
            logger.info(f"  기간: {start} ~ {end}")
            logger.info(f"  일수: {days}")

# CLI 실행용
def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='피처 엔지니어링')
    parser.add_argument('--bars', default='3m', help='시간프레임')
    parser.add_argument('--symbols', nargs='+', help='심볼 리스트')
    parser.add_argument('--start', help='시작 날짜')
    parser.add_argument('--end', help='종료 날짜')
    parser.add_argument('--db', default='data/trading.db', help='DB 경로')
    
    args = parser.parse_args()
    
    engineer = FeatureEngineer(db_path=args.db)
    
    engineer.generate_features(
        timeframe=args.bars,
        symbols=args.symbols,
        start_date=args.start,
        end_date=args.end
    )

if __name__ == "__main__":
    main()