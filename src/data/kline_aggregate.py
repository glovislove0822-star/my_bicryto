"""Kline 데이터 집계 모듈"""

import pandas as pd
import numpy as np
import duckdb
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import logging

from ..utils.logging import Logger
from ..utils.time import TimeUtils
from ..utils.io import IOUtils

logger = Logger.get_logger(__name__)

class KlineAggregator:
    """Kline 데이터를 다양한 시간 프레임으로 집계"""
    
    def __init__(self, db_path: str = "data/trading.db"):
        """
        Args:
            db_path: DuckDB 데이터베이스 경로
        """
        self.db_path = Path(db_path)
        self.conn = duckdb.connect(str(self.db_path))
        
        # 시간 프레임 매핑
        self.timeframe_map = {
            '3m': '3 minutes',
            '5m': '5 minutes',
            '15m': '15 minutes',
            '30m': '30 minutes',
            '1h': '1 hour',
            '4h': '4 hours',
            '1d': '1 day'
        }
    
    def aggregate_klines(self, 
                        source_timeframe: str = '1m',
                        target_timeframes: List[str] = ['3m', '5m'],
                        symbols: Optional[List[str]] = None):
        """1분봉을 다른 시간 프레임으로 집계
        
        Args:
            source_timeframe: 소스 시간 프레임
            target_timeframes: 타겟 시간 프레임 리스트
            symbols: 처리할 심볼 리스트 (None이면 전체)
        """
        
        # 심볼 목록 가져오기
        if symbols is None:
            result = self.conn.execute(f"""
                SELECT DISTINCT symbol 
                FROM trading.klines_{source_timeframe}
            """).fetchall()
            symbols = [r[0] for r in result]
        
        logger.info(f"집계 시작: {symbols}")
        
        for target_tf in target_timeframes:
            logger.info(f"\n{target_tf} 집계 중...")
            
            for symbol in symbols:
                try:
                    self._aggregate_single_symbol(
                        symbol=symbol,
                        source_tf=source_timeframe,
                        target_tf=target_tf
                    )
                except Exception as e:
                    logger.error(f"{symbol} {target_tf} 집계 실패: {e}")
        
        logger.info("\n집계 완료!")
        self._print_aggregation_stats(target_timeframes)
    
    def _aggregate_single_symbol(self,
                                 symbol: str,
                                 source_tf: str,
                                 target_tf: str):
        """단일 심볼 집계"""
        
        interval = self.timeframe_map.get(target_tf)
        if not interval:
            raise ValueError(f"지원하지 않는 timeframe: {target_tf}")
        
        # 집계 쿼리
        query = f"""
            WITH aggregated AS (
                SELECT
                    symbol,
                    time_bucket(INTERVAL '{interval}', open_time) as open_time,
                    FIRST(open ORDER BY open_time) as open,
                    MAX(high) as high,
                    MIN(low) as low,
                    LAST(close ORDER BY open_time) as close,
                    SUM(volume) as volume,
                    SUM(quote_volume) as quote_volume,
                    SUM(trade_count) as trade_count,
                    SUM(taker_buy_volume) as taker_buy_volume,
                    SUM(taker_buy_quote_volume) as taker_buy_quote_volume,
                    -- VWAP 계산
                    SUM(close * volume) / NULLIF(SUM(volume), 0) as vwap
                FROM trading.klines_{source_tf}
                WHERE symbol = '{symbol}'
                GROUP BY symbol, time_bucket(INTERVAL '{interval}', open_time)
            )
            INSERT OR REPLACE INTO trading.bars_{target_tf}
            SELECT 
                symbol,
                open_time,
                open,
                high,
                low,
                close,
                volume,
                quote_volume,
                trade_count,
                vwap
            FROM aggregated
            WHERE volume > 0  -- 볼륨이 있는 바만 저장
            ORDER BY open_time
        """
        
        self.conn.execute(query)
        
        # 집계 결과 확인
        count = self.conn.execute(f"""
            SELECT COUNT(*) 
            FROM trading.bars_{target_tf}
            WHERE symbol = '{symbol}'
        """).fetchone()[0]
        
        logger.debug(f"{symbol} {target_tf}: {count} bars")
    
    def calculate_session_stats(self, timeframe: str = '3m'):
        """세션별 통계 계산"""
        
        query = f"""
            CREATE OR REPLACE VIEW trading.session_stats_{timeframe} AS
            WITH session_data AS (
                SELECT 
                    symbol,
                    open_time,
                    close,
                    volume,
                    EXTRACT(hour FROM open_time) as hour_utc,
                    CASE 
                        WHEN EXTRACT(hour FROM open_time) BETWEEN 0 AND 7 THEN 'asian'
                        WHEN EXTRACT(hour FROM open_time) BETWEEN 7 AND 15 THEN 'european'
                        WHEN EXTRACT(hour FROM open_time) BETWEEN 13 AND 21 THEN 'american'
                        ELSE 'overlap'
                    END as session
                FROM trading.bars_{timeframe}
            )
            SELECT 
                symbol,
                session,
                AVG(volume) as avg_volume,
                STDDEV(volume) as std_volume,
                AVG(ABS(close - LAG(close) OVER (PARTITION BY symbol ORDER BY open_time))) as avg_move,
                COUNT(*) as bar_count
            FROM session_data
            GROUP BY symbol, session
        """
        
        self.conn.execute(query)
        logger.info(f"세션 통계 계산 완료: session_stats_{timeframe}")
    
    def create_rolling_windows(self, timeframe: str = '3m'):
        """롤링 윈도우 뷰 생성"""
        
        windows = [20, 50, 100, 200]  # 바 개수
        
        for window in windows:
            query = f"""
                CREATE OR REPLACE VIEW trading.rolling_{window}_{timeframe} AS
                SELECT 
                    symbol,
                    open_time,
                    close,
                    -- 이동평균
                    AVG(close) OVER w as ma_{window},
                    -- 표준편차
                    STDDEV(close) OVER w as std_{window},
                    -- 최고/최저
                    MAX(high) OVER w as high_{window},
                    MIN(low) OVER w as low_{window},
                    -- 볼륨
                    AVG(volume) OVER w as avg_volume_{window},
                    -- RSI 준비 (변화량)
                    close - LAG(close) OVER (PARTITION BY symbol ORDER BY open_time) as price_change
                FROM trading.bars_{timeframe}
                WINDOW w AS (
                    PARTITION BY symbol 
                    ORDER BY open_time 
                    ROWS BETWEEN {window-1} PRECEDING AND CURRENT ROW
                )
            """
            
            self.conn.execute(query)
            logger.debug(f"롤링 윈도우 생성: rolling_{window}_{timeframe}")
    
    def validate_data_quality(self, timeframe: str = '3m') -> Dict:
        """데이터 품질 검증"""
        
        quality_report = {}
        
        # 1. 결측 데이터 체크
        missing_query = f"""
            WITH expected_bars AS (
                SELECT 
                    symbol,
                    generate_series(
                        MIN(open_time),
                        MAX(open_time),
                        INTERVAL '{self.timeframe_map[timeframe]}'
                    ) as expected_time
                FROM trading.bars_{timeframe}
                GROUP BY symbol
            ),
            actual_bars AS (
                SELECT symbol, open_time
                FROM trading.bars_{timeframe}
            )
            SELECT 
                e.symbol,
                COUNT(*) as missing_bars
            FROM expected_bars e
            LEFT JOIN actual_bars a 
                ON e.symbol = a.symbol 
                AND e.expected_time = a.open_time
            WHERE a.open_time IS NULL
            GROUP BY e.symbol
        """
        
        missing_result = self.conn.execute(missing_query).fetchall()
        quality_report['missing_bars'] = {r[0]: r[1] for r in missing_result}
        
        # 2. 이상치 체크 (가격 점프)
        outlier_query = f"""
            WITH price_changes AS (
                SELECT 
                    symbol,
                    open_time,
                    close,
                    LAG(close) OVER (PARTITION BY symbol ORDER BY open_time) as prev_close,
                    ABS(close - LAG(close) OVER (PARTITION BY symbol ORDER BY open_time)) / 
                        LAG(close) OVER (PARTITION BY symbol ORDER BY open_time) as pct_change
                FROM trading.bars_{timeframe}
            )
            SELECT 
                symbol,
                COUNT(*) as outlier_count
            FROM price_changes
            WHERE pct_change > 0.1  -- 10% 이상 변동
            GROUP BY symbol
        """
        
        outlier_result = self.conn.execute(outlier_query).fetchall()
        quality_report['outliers'] = {r[0]: r[1] for r in outlier_result}
        
        # 3. 볼륨 이상치
        volume_query = f"""
            WITH volume_stats AS (
                SELECT 
                    symbol,
                    AVG(volume) as avg_volume,
                    STDDEV(volume) as std_volume
                FROM trading.bars_{timeframe}
                GROUP BY symbol
            )
            SELECT 
                b.symbol,
                COUNT(*) as abnormal_volume_count
            FROM trading.bars_{timeframe} b
            JOIN volume_stats v ON b.symbol = v.symbol
            WHERE b.volume > v.avg_volume + 5 * v.std_volume  -- 5 시그마 이상
                OR b.volume < 0.01 * v.avg_volume  -- 평균의 1% 미만
            GROUP BY b.symbol
        """
        
        volume_result = self.conn.execute(volume_query).fetchall()
        quality_report['abnormal_volume'] = {r[0]: r[1] for r in volume_result}
        
        # 품질 점수 계산
        for symbol in quality_report.get('missing_bars', {}).keys():
            total_bars = self.conn.execute(f"""
                SELECT COUNT(*) 
                FROM trading.bars_{timeframe}
                WHERE symbol = '{symbol}'
            """).fetchone()[0]
            
            missing = quality_report['missing_bars'].get(symbol, 0)
            outliers = quality_report['outliers'].get(symbol, 0)
            abnormal_vol = quality_report['abnormal_volume'].get(symbol, 0)
            
            # 품질 점수 (100점 만점)
            score = 100
            score -= (missing / max(total_bars, 1)) * 30  # 결측: 최대 -30점
            score -= (outliers / max(total_bars, 1)) * 20  # 이상치: 최대 -20점
            score -= (abnormal_vol / max(total_bars, 1)) * 10  # 볼륨이상: 최대 -10점
            
            quality_report.setdefault('quality_scores', {})[symbol] = max(0, score)
        
        return quality_report
    
    def _print_aggregation_stats(self, timeframes: List[str]):
        """집계 통계 출력"""
        
        logger.info("\n=== 집계 통계 ===")
        
        for tf in timeframes:
            stats = self.conn.execute(f"""
                SELECT 
                    symbol,
                    COUNT(*) as bar_count,
                    MIN(open_time) as start_time,
                    MAX(open_time) as end_time
                FROM trading.bars_{tf}
                GROUP BY symbol
            """).fetchall()
            
            logger.info(f"\n{tf.upper()} Bars:")
            for symbol, count, start, end in stats:
                logger.info(f"  {symbol}: {count:,} bars ({start} ~ {end})")

# CLI 실행용
def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Kline 데이터 집계')
    parser.add_argument('--bars', nargs='+', default=['3m', '5m'], help='타겟 시간프레임')
    parser.add_argument('--symbols', nargs='+', help='심볼 리스트')
    parser.add_argument('--db', default='data/trading.db', help='DB 경로')
    parser.add_argument('--validate', action='store_true', help='데이터 품질 검증')
    
    args = parser.parse_args()
    
    aggregator = KlineAggregator(db_path=args.db)
    
    # 집계 수행
    aggregator.aggregate_klines(
        target_timeframes=args.bars,
        symbols=args.symbols
    )
    
    # 세션 통계
    for tf in args.bars:
        aggregator.calculate_session_stats(tf)
        aggregator.create_rolling_windows(tf)
    
    # 품질 검증
    if args.validate:
        for tf in args.bars:
            report = aggregator.validate_data_quality(tf)
            
            logger.info(f"\n=== {tf} 데이터 품질 리포트 ===")
            logger.info("품질 점수:")
            for symbol, score in report.get('quality_scores', {}).items():
                logger.info(f"  {symbol}: {score:.1f}/100")

if __name__ == "__main__":
    main()