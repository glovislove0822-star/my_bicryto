"""데이터셋 준비 모듈"""

import pandas as pd
import numpy as np
import duckdb
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from datetime import datetime, timedelta
import logging
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

from ..utils.logging import Logger
from ..utils.io import IOUtils

logger = Logger.get_logger(__name__)

class DatasetBuilder:
    """ML 데이터셋 빌더"""
    
    def __init__(self, db_path: str = "data/trading.db"):
        """
        Args:
            db_path: DuckDB 데이터베이스 경로
        """
        self.db_path = Path(db_path)
        self.conn = duckdb.connect(str(self.db_path))
        
        # 피처 그룹
        self.feature_groups = {
            'direction': [
                'ret_60', 'ret_120', 'ema_slope_fast', 'ema_slope_slow',
                'donchian_signal', 'adx', 'hurst_exp'
            ],
            'entry': [
                'rsi_2', 'rsi_3', 'vwap_z',
                'bb_width'
            ],
            'microstructure': [
                'ofi', 'ofi_z', 'queue_imbalance', 'spread_bps',
                'depth_total', 'trade_intensity', 'liquidity_score'
            ],
            'risk': [
                'atr', 'parkinson_vol', 'realized_vol', 'vol_cluster'
            ],
            'funding': [
                'funding_rate', 'funding_ma_8h', 'funding_ma_24h', 'funding_std'
            ],
            'time': [
                'hour_utc', 'day_of_week', 'is_asian_session',
                'is_european_session', 'is_american_session'
            ]
        }
        
        # 제외할 피처
        self.exclude_features = [
            'symbol', 'ts', 'open', 'high', 'low', 'close', 'volume',
            'label', 'meta_label', 'pnl', 'pnl_pct'
        ]
    
    def build_dataset(self,
                     timeframe: str = '3m',
                     symbols: Optional[List[str]] = None,
                     feature_groups: Optional[List[str]] = None,
                     target: str = 'meta_label',
                     start_date: Optional[str] = None,
                     end_date: Optional[str] = None,
                     test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """데이터셋 구축
        
        Args:
            timeframe: 시간 프레임
            symbols: 심볼 리스트
            feature_groups: 사용할 피처 그룹
            target: 타겟 변수
            start_date: 시작 날짜
            end_date: 종료 날짜
            test_size: 테스트 셋 비율
            
        Returns:
            (X_train, X_test, y_train, y_test)
        """
        
        # 데이터 로드
        df = self._load_full_dataset(timeframe, symbols, start_date, end_date)
        
        if df.empty:
            raise ValueError("데이터셋이 비어있습니다")
        
        # 피처 선택
        features = self._select_features(df, feature_groups)
        
        # 피처 엔지니어링
        features = self._engineer_features(features)
        
        # 타겟 준비
        if target not in df.columns:
            raise ValueError(f"타겟 변수 '{target}'를 찾을 수 없습니다")
        
        y = df[target]
        
        # NaN 제거
        mask = ~(features.isna().any(axis=1) | y.isna())
        features = features[mask]
        y = y[mask]
        
        # Train/Test 분할 (시계열 보존)
        split_idx = int(len(features) * (1 - test_size))
        
        X_train = features.iloc[:split_idx]
        X_test = features.iloc[split_idx:]
        y_train = y.iloc[:split_idx]
        y_test = y.iloc[split_idx:]
        
        logger.info(f"데이터셋 구축 완료:")
        logger.info(f"  Train: {len(X_train):,} samples")
        logger.info(f"  Test: {len(X_test):,} samples")
        logger.info(f"  Features: {X_train.shape[1]}")
        logger.info(f"  Target distribution:")
        logger.info(f"    Train: {y_train.value_counts().to_dict()}")
        logger.info(f"    Test: {y_test.value_counts().to_dict()}")
        
        return X_train, X_test, y_train, y_test
    
    def _load_full_dataset(self,
                          timeframe: str,
                          symbols: Optional[List[str]],
                          start_date: Optional[str],
                          end_date: Optional[str]) -> pd.DataFrame:
        """전체 데이터셋 로드"""
        
        where_clause = "WHERE 1=1"
        
        if symbols:
            symbols_str = "','".join(symbols)
            where_clause += f" AND f.symbol IN ('{symbols_str}')"
        
        if start_date:
            where_clause += f" AND f.ts >= '{start_date}'"
        
        if end_date:
            where_clause += f" AND f.ts <= '{end_date}'"
        
        query = f"""
            SELECT 
                f.*,
                l.label,
                l.meta_label,
                l.pnl_pct,
                l.tp_hit,
                l.sl_hit
            FROM trading.features_{timeframe} f
            LEFT JOIN trading.labels_{timeframe} l
                ON f.symbol = l.symbol AND f.ts = l.ts
            {where_clause}
            ORDER BY f.ts
        """
        
        df = self.conn.execute(query).df()
        
        if not df.empty:
            df['ts'] = pd.to_datetime(df['ts'])
            df.set_index('ts', inplace=True)
        
        logger.info(f"데이터 로드: {len(df):,} rows")
        
        return df
    
    def _select_features(self,
                        df: pd.DataFrame,
                        feature_groups: Optional[List[str]]) -> pd.DataFrame:
        """피처 선택"""
        
        if feature_groups is None:
            feature_groups = list(self.feature_groups.keys())
        
        selected_features = []
        
        for group in feature_groups:
            if group in self.feature_groups:
                group_features = self.feature_groups[group]
                # 실제 존재하는 피처만 선택
                available_features = [f for f in group_features if f in df.columns]
                selected_features.extend(available_features)
        
        # 중복 제거
        selected_features = list(set(selected_features))
        
        # 제외 피처 제거
        selected_features = [f for f in selected_features 
                           if f not in self.exclude_features]
        
        logger.info(f"선택된 피처: {len(selected_features)}개")
        
        return df[selected_features]
    
    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """추가 피처 엔지니어링"""
        
        # 상호작용 피처
        if 'ofi_z' in df.columns and 'vwap_z' in df.columns:
            df['ofi_vwap_interaction'] = df['ofi_z'] * df['vwap_z']
        
        if 'realized_vol' in df.columns and 'spread_bps' in df.columns:
            df['vol_spread_ratio'] = df['realized_vol'] / (df['spread_bps'] + 0.1)
        
        if 'rsi_2' in df.columns and 'rsi_3' in df.columns:
            df['rsi_divergence'] = df['rsi_2'] - df['rsi_3']
        
        # 랙 피처
        lag_features = ['ofi_z', 'vwap_z', 'realized_vol']
        for feature in lag_features:
            if feature in df.columns:
                df[f'{feature}_lag1'] = df[feature].shift(1)
                df[f'{feature}_lag2'] = df[feature].shift(2)
        
        # 롤링 통계
        if 'ofi' in df.columns:
            df['ofi_rolling_mean'] = df['ofi'].rolling(10).mean()
            df['ofi_rolling_std'] = df['ofi'].rolling(10).std()
        
        # 변화율
        if 'spread_bps' in df.columns:
            df['spread_change'] = df['spread_bps'].pct_change()
        
        if 'depth_total' in df.columns:
            df['depth_change'] = df['depth_total'].pct_change()
        
        return df
    
    def create_purged_kfold(self,
                          X: pd.DataFrame,
                          y: pd.Series,
                          n_splits: int = 5,
                          embargo_bars: int = 2) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Purged K-Fold Cross Validation
        
        시계열 데이터 누수를 방지하는 CV 분할
        
        Args:
            X: 피처 데이터
            y: 타겟 데이터
            n_splits: 분할 수
            embargo_bars: 엠바고 기간
            
        Returns:
            [(train_idx, test_idx), ...]
        """
        
        indices = np.arange(len(X))
        test_starts = [(i * len(X)) // n_splits for i in range(n_splits)]
        test_ends = [(i * len(X)) // n_splits for i in range(1, n_splits + 1)]
        
        splits = []
        
        for test_start, test_end in zip(test_starts, test_ends):
            test_indices = indices[test_start:test_end]
            
            # Purging: 테스트 셋 이후 데이터 제거
            train_indices = indices[indices < test_start - embargo_bars]
            
            # Embargo: 테스트 셋 직전 데이터 제거
            if test_end < len(indices):
                train_indices_after = indices[test_end + embargo_bars:]
                train_indices = np.concatenate([train_indices, train_indices_after])
            
            splits.append((train_indices, test_indices))
        
        logger.info(f"Purged K-Fold 생성: {n_splits} splits, embargo={embargo_bars}")
        
        return splits
    
    def get_sample_weights(self,
                          y: pd.Series,
                          method: str = 'balanced') -> np.ndarray:
        """샘플 가중치 계산
        
        Args:
            y: 타겟 데이터
            method: 가중치 방법 ('balanced', 'time_decay', 'return_based')
            
        Returns:
            샘플 가중치 배열
        """
        
        if method == 'balanced':
            # 클래스 불균형 보정
            class_weights = len(y) / (len(np.unique(y)) * np.bincount(y))
            weights = np.array([class_weights[int(label)] for label in y])
            
        elif method == 'time_decay':
            # 최신 데이터에 더 높은 가중치
            decay_factor = 0.99
            time_weights = decay_factor ** np.arange(len(y) - 1, -1, -1)
            weights = time_weights / time_weights.sum() * len(y)
            
        elif method == 'return_based':
            # 수익률 기반 가중치 (메타 라벨링용)
            # 여기서는 간단히 균등 가중치
            weights = np.ones(len(y))
            
        else:
            weights = np.ones(len(y))
        
        return weights
    
    def prepare_sequence_data(self,
                            X: pd.DataFrame,
                            y: pd.Series,
                            sequence_length: int = 20) -> Tuple[np.ndarray, np.ndarray]:
        """시퀀스 데이터 준비 (LSTM 등을 위한)
        
        Args:
            X: 피처 데이터
            y: 타겟 데이터
            sequence_length: 시퀀스 길이
            
        Returns:
            (X_seq, y_seq)
        """
        
        X_values = X.values
        y_values = y.values
        
        X_seq = []
        y_seq = []
        
        for i in range(sequence_length, len(X_values)):
            X_seq.append(X_values[i-sequence_length:i])
            y_seq.append(y_values[i])
        
        X_seq = np.array(X_seq)
        y_seq = np.array(y_seq)
        
        logger.info(f"시퀀스 데이터 생성: {X_seq.shape}")
        
        return X_seq, y_seq

# CLI 실행용
def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='데이터셋 준비')
    parser.add_argument('--bars', default='3m', help='시간프레임')
    parser.add_argument('--symbols', nargs='+', help='심볼 리스트')
    parser.add_argument('--target', default='meta_label', help='타겟 변수')
    parser.add_argument('--start', help='시작 날짜')
    parser.add_argument('--end', help='종료 날짜')
    parser.add_argument('--test-size', type=float, default=0.2, help='테스트 셋 비율')
    parser.add_argument('--db', default='data/trading.db', help='DB 경로')
    parser.add_argument('--output', help='출력 경로')
    
    args = parser.parse_args()
    
    builder = DatasetBuilder(db_path=args.db)
    
    X_train, X_test, y_train, y_test = builder.build_dataset(
        timeframe=args.bars,
        symbols=args.symbols,
        target=args.target,
        start_date=args.start,
        end_date=args.end,
        test_size=args.test_size
    )
    
    if args.output:
        # 데이터셋 저장
        output_path = Path(args.output)
        output_path.mkdir(parents=True, exist_ok=True)
        
        X_train.to_parquet(output_path / 'X_train.parquet')
        X_test.to_parquet(output_path / 'X_test.parquet')
        y_train.to_frame('target').to_parquet(output_path / 'y_train.parquet')
        y_test.to_frame('target').to_parquet(output_path / 'y_test.parquet')
        
        logger.info(f"데이터셋 저장: {output_path}")

if __name__ == "__main__":
    main()