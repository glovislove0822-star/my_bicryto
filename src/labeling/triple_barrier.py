"""트리플 배리어 라벨링 모듈"""

import pandas as pd
import numpy as np
import duckdb
from pathlib import Path
from typing import Optional, Tuple, Dict, List
from datetime import datetime, timedelta
import logging
from tqdm import tqdm
from numba import jit
import warnings
warnings.filterwarnings('ignore')

from ..utils.logging import Logger
from ..utils.math import MathUtils
from ..utils.io import IOUtils

logger = Logger.get_logger(__name__)

class TripleBarrierLabeler:
    """트리플 배리어 방식 라벨링 클래스"""
    
    def __init__(self, db_path: str = "data/trading.db"):
        """
        Args:
            db_path: DuckDB 데이터베이스 경로
        """
        self.db_path = Path(db_path)
        self.conn = duckdb.connect(str(self.db_path))
        self.math_utils = MathUtils()
        
        # 기본 파라미터
        self.default_params = {
            'tp_atr_multiplier': 1.2,  # Take Profit ATR 배수
            'sl_atr_multiplier': 0.6,  # Stop Loss ATR 배수
            'max_holding_bars': 40,     # 최대 홀딩 기간
            'min_ret_threshold': 0.0001, # 최소 수익률 임계값
            'use_dynamic_barriers': True, # 동적 배리어 사용
            'regime_adaptive': True      # 레짐 적응형 (v2.0)
        }
    
    def generate_labels(self,
                        timeframe: str = '3m',
                        symbols: Optional[List[str]] = None,
                        params: Optional[Dict] = None,
                        start_date: Optional[str] = None,
                        end_date: Optional[str] = None):
        """트리플 배리어 라벨 생성
        
        Args:
            timeframe: 시간 프레임
            symbols: 심볼 리스트
            params: 라벨링 파라미터
            start_date: 시작 날짜
            end_date: 종료 날짜
        """
        
        # 파라미터 병합
        if params:
            self.params = {**self.default_params, **params}
        else:
            self.params = self.default_params
        
        # 심볼 목록
        if symbols is None:
            result = self.conn.execute(f"""
                SELECT DISTINCT symbol
                FROM trading.features_{timeframe}
            """).fetchall()
            symbols = [r[0] for r in result]
        
        logger.info(f"트리플 배리어 라벨링 시작: {symbols}")
        logger.info(f"파라미터: {self.params}")
        
        for symbol in tqdm(symbols, desc="라벨링"):
            try:
                # 데이터 로드
                df = self._load_data(symbol, timeframe, start_date, end_date)
                
                if df.empty or len(df) < 100:
                    logger.warning(f"{symbol} 데이터 부족, 스킵")
                    continue
                
                # 트리플 배리어 라벨 생성
                labels = self._apply_triple_barrier(df)
                
                # 추가 메트릭 계산
                labels = self._calculate_label_metrics(labels, df)
                
                # DB 저장
                self._save_labels(labels, symbol, timeframe)
                
            except Exception as e:
                logger.error(f"{symbol} 라벨링 실패: {e}")
                continue
        
        logger.info("라벨링 완료!")
        self._print_label_statistics(timeframe)
    
    def _load_data(self,
                  symbol: str,
                  timeframe: str,
                  start_date: Optional[str],
                  end_date: Optional[str]) -> pd.DataFrame:
        """데이터 로드"""
        
        where_clause = f"WHERE symbol = '{symbol}'"
        if start_date:
            where_clause += f" AND ts >= '{start_date}'"
        if end_date:
            where_clause += f" AND ts <= '{end_date}'"
        
        query = f"""
            SELECT 
                f.*,
                b.open,
                b.high,
                b.low,
                b.close,
                b.volume
            FROM trading.features_{timeframe} f
            JOIN trading.bars_{timeframe} b
                ON f.symbol = b.symbol
                AND f.ts = b.open_time
            {where_clause}
            ORDER BY f.ts
        """
        
        df = self.conn.execute(query).df()
        
        if not df.empty:
            df['ts'] = pd.to_datetime(df['ts'])
            df.set_index('ts', inplace=True)
        
        return df
    
    def _apply_triple_barrier(self, df: pd.DataFrame) -> pd.DataFrame:
        """트리플 배리어 적용
        
        Args:
            df: 피처 데이터프레임
        
        Returns:
            라벨 데이터프레임
        """
        
        labels = []
        
        for i in tqdm(range(len(df) - self.params['max_holding_bars']), 
                     desc="배리어 적용", leave=False):
            
            entry_idx = df.index[i]
            entry_price = df.loc[entry_idx, 'close']
            
            # ATR 기반 배리어 설정
            atr = df.loc[entry_idx, 'atr']
            
            # 레짐 적응형 배리어 (v2.0)
            if self.params['regime_adaptive'] and 'vol_regime' in df.columns:
                vol_regime = df.loc[entry_idx, 'vol_regime_encoded'] if 'vol_regime_encoded' in df.columns else 1
                
                # 변동성 레짐에 따라 배리어 조정
                if vol_regime == 3:  # extreme
                    tp_multiplier = self.params['tp_atr_multiplier'] * 1.5
                    sl_multiplier = self.params['sl_atr_multiplier'] * 1.2
                elif vol_regime == 2:  # high
                    tp_multiplier = self.params['tp_atr_multiplier'] * 1.2
                    sl_multiplier = self.params['sl_atr_multiplier'] * 1.1
                elif vol_regime == 0:  # low
                    tp_multiplier = self.params['tp_atr_multiplier'] * 0.8
                    sl_multiplier = self.params['sl_atr_multiplier'] * 0.9
                else:  # normal
                    tp_multiplier = self.params['tp_atr_multiplier']
                    sl_multiplier = self.params['sl_atr_multiplier']
            else:
                tp_multiplier = self.params['tp_atr_multiplier']
                sl_multiplier = self.params['sl_atr_multiplier']
            
            # 배리어 가격
            tp_price = entry_price * (1 + tp_multiplier * atr / entry_price)
            sl_price = entry_price * (1 - sl_multiplier * atr / entry_price)
            
            # 동적 배리어 (선택적)
            if self.params['use_dynamic_barriers']:
                # 시장 상황에 따라 배리어 조정
                if 'trend_regime' in df.columns:
                    trend = df.loc[entry_idx, 'trend_regime_encoded'] if 'trend_regime_encoded' in df.columns else 2
                    
                    if trend >= 3:  # 상승 트렌드
                        tp_price *= 1.1
                        sl_price *= 1.05
                    elif trend <= 1:  # 하락 트렌드
                        tp_price *= 0.95
                        sl_price *= 0.9
            
            # 배리어 체크
            label_info = self._check_barriers(
                df=df,
                entry_idx=i,
                entry_price=entry_price,
                tp_price=tp_price,
                sl_price=sl_price,
                max_bars=self.params['max_holding_bars']
            )
            
            labels.append(label_info)
        
        return pd.DataFrame(labels)
    
    @staticmethod
    @jit(nopython=True)
    def _check_barriers_numba(prices: np.ndarray,
                              entry_price: float,
                              tp_price: float,
                              sl_price: float,
                              max_bars: int) -> Tuple[int, int, float, float]:
        """Numba 가속 배리어 체크
        
        Returns:
            (label, exit_bar, exit_price, pnl)
        """
        
        for j in range(1, min(max_bars, len(prices))):
            current_price = prices[j]
            
            # TP 도달
            if current_price >= tp_price:
                pnl = (tp_price - entry_price) / entry_price
                return 1, j, tp_price, pnl
            
            # SL 도달
            if current_price <= sl_price:
                pnl = (sl_price - entry_price) / entry_price
                return -1, j, sl_price, pnl
        
        # 시간 종료
        final_price = prices[min(max_bars-1, len(prices)-1)]
        pnl = (final_price - entry_price) / entry_price
        
        if pnl > 0:
            return 1, max_bars, final_price, pnl
        elif pnl < 0:
            return -1, max_bars, final_price, pnl
        else:
            return 0, max_bars, final_price, pnl
    
    def _check_barriers(self,
                       df: pd.DataFrame,
                       entry_idx: int,
                       entry_price: float,
                       tp_price: float,
                       sl_price: float,
                       max_bars: int) -> Dict:
        """배리어 체크 (Python 버전)"""
        
        entry_ts = df.index[entry_idx]
        
        # 미래 가격
        future_prices = df['close'].iloc[entry_idx:entry_idx+max_bars+1].values
        
        # Numba 가속 사용 가능 시
        try:
            label, exit_bar, exit_price, pnl = self._check_barriers_numba(
                future_prices,
                entry_price,
                tp_price,
                sl_price,
                max_bars
            )
        except:
            # Fallback to Python
            label = 0
            exit_bar = max_bars
            exit_price = entry_price
            pnl = 0
            
            for j in range(1, min(max_bars, len(future_prices))):
                current_price = future_prices[j]
                
                # TP 도달
                if current_price >= tp_price:
                    label = 1
                    exit_bar = j
                    exit_price = tp_price
                    pnl = (tp_price - entry_price) / entry_price
                    break
                
                # SL 도달
                if current_price <= sl_price:
                    label = -1
                    exit_bar = j
                    exit_price = sl_price
                    pnl = (sl_price - entry_price) / entry_price
                    break
            
            # 시간 종료
            if label == 0:
                final_price = future_prices[min(max_bars-1, len(future_prices)-1)]
                pnl = (final_price - entry_price) / entry_price
                
                if pnl > self.params['min_ret_threshold']:
                    label = 1
                elif pnl < -self.params['min_ret_threshold']:
                    label = -1
                else:
                    label = 0
                
                exit_bar = max_bars
                exit_price = final_price
        
        # 결과 반환
        return {
            'ts': entry_ts,
            'label': label,
            'tp_hit': label == 1 and exit_bar < max_bars,
            'sl_hit': label == -1 and exit_bar < max_bars,
            'time_exit': exit_bar >= max_bars,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'exit_bars': exit_bar,
            'exit_time': df.index[min(entry_idx + exit_bar, len(df)-1)],
            'pnl': pnl * entry_price,  # 실제 PnL
            'pnl_pct': pnl,  # 퍼센트 PnL
            'tp_price': tp_price,
            'sl_price': sl_price
        }
    
    def _calculate_label_metrics(self, 
                                labels: pd.DataFrame,
                                df: pd.DataFrame) -> pd.DataFrame:
        """라벨 메트릭 계산"""
        
        if labels.empty:
            return labels
        
        labels = labels.set_index('ts')
        
        # 추가 메트릭
        labels['holding_time'] = labels['exit_bars'] * 3  # 3분봉 기준 (분)
        labels['mae'] = 0  # Maximum Adverse Excursion
        labels['mfe'] = 0  # Maximum Favorable Excursion
        
        # MAE/MFE 계산
        for idx in labels.index:
            if idx not in df.index:
                continue
                
            entry_idx = df.index.get_loc(idx)
            exit_bars = int(labels.loc[idx, 'exit_bars'])
            entry_price = labels.loc[idx, 'entry_price']
            
            if entry_idx + exit_bars <= len(df):
                price_path = df['close'].iloc[entry_idx:entry_idx+exit_bars]
                
                # MAE: 최대 역행
                labels.loc[idx, 'mae'] = (price_path.min() - entry_price) / entry_price
                
                # MFE: 최대 유리
                labels.loc[idx, 'mfe'] = (price_path.max() - entry_price) / entry_price
        
        # Risk-Reward Ratio
        labels['risk_reward'] = abs(labels['mfe'] / labels['mae'])
        labels['risk_reward'] = labels['risk_reward'].replace([np.inf, -np.inf], 0)
        
        # 효율성 (실제 PnL / MFE)
        labels['efficiency'] = labels['pnl_pct'] / labels['mfe']
        labels['efficiency'] = labels['efficiency'].replace([np.inf, -np.inf], 0)
        
        # 메타 라벨 준비 (기본값)
        labels['meta_label'] = 1  # 모든 신호 실행 (메타 라벨러에서 수정)
        
        return labels
    
    def _save_labels(self, 
                    labels: pd.DataFrame,
                    symbol: str,
                    timeframe: str):
        """라벨 데이터 저장"""
        
        if labels.empty:
            return
        
        # symbol 추가
        labels['symbol'] = symbol
        
        # 인덱스 리셋
        labels = labels.reset_index()
        
        # 저장할 컬럼 선택
        save_columns = [
            'symbol', 'ts', 'label',
            'tp_hit', 'sl_hit', 'time_exit',
            'exit_price', 'exit_time',
            'pnl', 'pnl_pct', 'meta_label'
        ]
        
        # 존재하는 컬럼만 선택
        save_cols = [col for col in save_columns if col in labels.columns]
        labels_save = labels[save_cols].copy()
        
        # DuckDB에 저장
        table_name = f"labels_{timeframe}"
        
        # 임시 테이블
        temp_table = f"temp_labels_{int(datetime.now().timestamp())}"
        self.conn.execute(f"CREATE TEMP TABLE {temp_table} AS SELECT * FROM labels_save")
        
        # UPSERT
        self.conn.execute(f"""
            INSERT OR REPLACE INTO trading.{table_name}
            SELECT * FROM {temp_table}
        """)
        
        self.conn.execute(f"DROP TABLE {temp_table}")
        
        logger.info(f"{symbol} 라벨 저장 완료: {len(labels_save)} rows")
    
    def _print_label_statistics(self, timeframe: str):
        """라벨 통계 출력"""
        
        stats = self.conn.execute(f"""
            SELECT 
                symbol,
                COUNT(*) as total_labels,
                SUM(CASE WHEN label = 1 THEN 1 ELSE 0 END) as positive_labels,
                SUM(CASE WHEN label = -1 THEN 1 ELSE 0 END) as negative_labels,
                SUM(CASE WHEN label = 0 THEN 1 ELSE 0 END) as neutral_labels,
                AVG(pnl_pct) as avg_pnl_pct,
                SUM(CASE WHEN tp_hit THEN 1 ELSE 0 END) as tp_hits,
                SUM(CASE WHEN sl_hit THEN 1 ELSE 0 END) as sl_hits,
                SUM(CASE WHEN time_exit THEN 1 ELSE 0 END) as time_exits
            FROM trading.labels_{timeframe}
            GROUP BY symbol
        """).fetchall()
        
        logger.info(f"\n=== {timeframe} 라벨 통계 ===")
        
        for row in stats:
            symbol = row[0]
            total = row[1]
            pos = row[2]
            neg = row[3]
            neutral = row[4]
            avg_pnl = row[5]
            tp_hits = row[6]
            sl_hits = row[7]
            time_exits = row[8]
            
            logger.info(f"\n{symbol}:")
            logger.info(f"  총 라벨: {total:,}")
            logger.info(f"  분포: +{pos} ({pos/total*100:.1f}%), "
                       f"-{neg} ({neg/total*100:.1f}%), "
                       f"0:{neutral} ({neutral/total*100:.1f}%)")
            logger.info(f"  평균 PnL: {avg_pnl:.4%}")
            logger.info(f"  종료 타입: TP={tp_hits}, SL={sl_hits}, Time={time_exits}")

# CLI 실행용
def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='트리플 배리어 라벨링')
    parser.add_argument('--bars', default='3m', help='시간프레임')
    parser.add_argument('--symbols', nargs='+', help='심볼 리스트')
    parser.add_argument('--tp', type=float, default=1.2, help='TP ATR 배수')
    parser.add_argument('--sl', type=float, default=0.6, help='SL ATR 배수')
    parser.add_argument('--max-bars', type=int, default=40, help='최대 홀딩 바')
    parser.add_argument('--start', help='시작 날짜')
    parser.add_argument('--end', help='종료 날짜')
    parser.add_argument('--db', default='data/trading.db', help='DB 경로')
    
    args = parser.parse_args()
    
    params = {
        'tp_atr_multiplier': args.tp,
        'sl_atr_multiplier': args.sl,
        'max_holding_bars': args.max_bars
    }
    
    labeler = TripleBarrierLabeler(db_path=args.db)
    
    labeler.generate_labels(
        timeframe=args.bars,
        symbols=args.symbols,
        params=params,
        start_date=args.start,
        end_date=args.end
    )

if __name__ == "__main__":
    main()