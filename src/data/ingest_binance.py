"""바이낸스 데이터 수집 모듈"""

import asyncio
import aiohttp
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import duckdb
from pathlib import Path
import logging
from tqdm.asyncio import tqdm
import time
from binance.um_futures import UMFutures
from binance.error import ClientError

from ..utils.logging import Logger
from ..utils.time import TimeUtils
from ..utils.io import IOUtils

logger = Logger.get_logger(__name__)

class BinanceDataIngester:
    """바이낸스 과거 데이터 수집기"""
    
    def __init__(self, 
                 db_path: str = "data/trading.db",
                 api_key: Optional[str] = None,
                 api_secret: Optional[str] = None):
        """
        Args:
            db_path: DuckDB 데이터베이스 경로
            api_key: 바이낸스 API 키 (선택)
            api_secret: 바이낸스 API 시크릿 (선택)
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = duckdb.connect(str(self.db_path))
        
        # 바이낸스 클라이언트
        self.client = UMFutures(key=api_key, secret=api_secret)
        
        # Rate limit 설정
        self.rate_limit = 1200  # 분당 요청 수
        self.request_weight = 0
        self.last_reset = time.time()
        
        # 심볼 정보 캐시
        self.symbol_info = {}
        
    async def fetch_klines(self, 
                          symbol: str,
                          interval: str = "1m",
                          start_time: datetime = None,
                          end_time: datetime = None,
                          limit: int = 1500) -> pd.DataFrame:
        """Kline 데이터 수집
        
        Args:
            symbol: 심볼 (예: BTCUSDT)
            interval: 시간 간격 (1m, 3m, 5m, 15m, 1h, 4h, 1d)
            start_time: 시작 시간
            end_time: 종료 시간
            limit: 한 번에 가져올 최대 개수
        """
        all_klines = []
        current_start = TimeUtils.to_unix_ms(start_time)
        end_ms = TimeUtils.to_unix_ms(end_time)
        
        with tqdm(desc=f"Fetching {symbol} klines", unit="batch") as pbar:
            while current_start < end_ms:
                try:
                    # Rate limit 체크
                    await self._check_rate_limit()
                    
                    # API 호출
                    klines = self.client.klines(
                        symbol=symbol,
                        interval=interval,
                        startTime=current_start,
                        endTime=min(current_start + limit * 60000, end_ms),
                        limit=limit
                    )
                    
                    if not klines:
                        break
                    
                    all_klines.extend(klines)
                    
                    # 다음 시작점
                    current_start = klines[-1][0] + 60000  # 1분 추가
                    pbar.update(1)
                    
                    # 짧은 대기
                    await asyncio.sleep(0.1)
                    
                except ClientError as e:
                    logger.error(f"API 에러: {e}")
                    if e.error_code == -1003:  # Too many requests
                        await asyncio.sleep(60)
                    else:
                        raise
                        
        # DataFrame 변환
        df = pd.DataFrame(all_klines, columns=[
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trade_count',
            'taker_buy_volume', 'taker_buy_quote_volume', 'ignore'
        ])
        
        # 타입 변환
        df['symbol'] = symbol
        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
        df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
        
        numeric_cols = ['open', 'high', 'low', 'close', 'volume',
                        'quote_volume', 'taker_buy_volume', 'taker_buy_quote_volume']
        df[numeric_cols] = df[numeric_cols].astype(float)
        df['trade_count'] = df['trade_count'].astype(int)
        
        # 불필요한 컬럼 제거
        df = df.drop(['close_time', 'ignore'], axis=1)
        
        logger.info(f"{symbol} klines 수집 완료: {len(df)} rows")
        return df
    
    async def fetch_funding_rate(self,
                                symbol: str,
                                start_time: datetime,
                                end_time: datetime) -> pd.DataFrame:
        """펀딩 레이트 히스토리 수집"""
        
        all_funding = []
        current_start = TimeUtils.to_unix_ms(start_time)
        end_ms = TimeUtils.to_unix_ms(end_time)
        
        with tqdm(desc=f"Fetching {symbol} funding rates", unit="batch") as pbar:
            while current_start < end_ms:
                try:
                    await self._check_rate_limit()
                    
                    # 펀딩 레이트 히스토리 조회
                    funding_data = self.client.funding_rate(
                        symbol=symbol,
                        startTime=current_start,
                        endTime=min(current_start + 1000 * 8 * 3600 * 1000, end_ms),  # 1000개 * 8시간
                        limit=1000
                    )
                    
                    if not funding_data:
                        break
                    
                    all_funding.extend(funding_data)
                    
                    # 다음 시작점
                    current_start = funding_data[-1]['fundingTime'] + 1
                    pbar.update(1)
                    
                    await asyncio.sleep(0.1)
                    
                except ClientError as e:
                    logger.error(f"Funding rate API 에러: {e}")
                    if e.error_code == -1003:
                        await asyncio.sleep(60)
                    else:
                        raise
        
        # DataFrame 변환
        df = pd.DataFrame(all_funding)
        
        if not df.empty:
            df['symbol'] = symbol
            df['ts'] = pd.to_datetime(df['fundingTime'], unit='ms')
            df['funding_rate'] = df['fundingRate'].astype(float)
            df['mark_price'] = df.get('markPrice', 0).astype(float)
            
            # 필요한 컬럼만 선택
            df = df[['symbol', 'ts', 'funding_rate', 'mark_price']]
        
        logger.info(f"{symbol} funding rates 수집 완료: {len(df)} rows")
        return df
    
    async def fetch_ticker_24h(self, symbols: List[str]) -> pd.DataFrame:
        """24시간 티커 정보 수집"""
        
        tickers = []
        
        for symbol in symbols:
            try:
                await self._check_rate_limit()
                
                ticker = self.client.ticker_24hr_price_change(symbol=symbol)
                tickers.append(ticker)
                
                await asyncio.sleep(0.1)
                
            except ClientError as e:
                logger.error(f"Ticker API 에러 {symbol}: {e}")
                continue
        
        df = pd.DataFrame(tickers)
        
        if not df.empty:
            # 숫자 변환
            numeric_cols = ['priceChange', 'priceChangePercent', 'weightedAvgPrice',
                           'lastPrice', 'lastQty', 'openPrice', 'highPrice', 
                           'lowPrice', 'volume', 'quoteVolume']
            
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    
    async def _check_rate_limit(self):
        """Rate limit 체크 및 대기"""
        current_time = time.time()
        
        # 1분마다 리셋
        if current_time - self.last_reset > 60:
            self.request_weight = 0
            self.last_reset = current_time
        
        # 가중치 증가
        self.request_weight += 1
        
        # 한계 도달 시 대기
        if self.request_weight >= self.rate_limit:
            wait_time = 60 - (current_time - self.last_reset)
            if wait_time > 0:
                logger.warning(f"Rate limit 도달, {wait_time:.1f}초 대기")
                await asyncio.sleep(wait_time)
                self.request_weight = 0
                self.last_reset = time.time()
    
    def save_to_db(self, df: pd.DataFrame, table_name: str):
        """DuckDB에 데이터 저장"""
        
        if df.empty:
            logger.warning(f"빈 DataFrame, {table_name} 저장 스킵")
            return
        
        try:
            # 테이블 존재 여부 확인
            existing_tables = self.conn.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_schema='trading'"
            ).fetchall()
            
            table_exists = any(table_name in t for t in existing_tables)
            
            if table_exists:
                # 기존 데이터와 병합 (중복 제거)
                temp_table = f"temp_{table_name}_{int(time.time())}"
                
                # 임시 테이블에 저장
                self.conn.execute(f"CREATE TEMP TABLE {temp_table} AS SELECT * FROM df")
                
                # UPSERT 수행
                if 'symbol' in df.columns and ('ts' in df.columns or 'open_time' in df.columns):
                    time_col = 'ts' if 'ts' in df.columns else 'open_time'
                    
                    self.conn.execute(f"""
                        INSERT INTO trading.{table_name}
                        SELECT * FROM {temp_table}
                        ON CONFLICT (symbol, {time_col})
                        DO UPDATE SET
                            open = EXCLUDED.open,
                            high = EXCLUDED.high,
                            low = EXCLUDED.low,
                            close = EXCLUDED.close,
                            volume = EXCLUDED.volume
                    """)
                else:
                    # 단순 추가
                    self.conn.execute(f"INSERT INTO trading.{table_name} SELECT * FROM {temp_table}")
                
                # 임시 테이블 삭제
                self.conn.execute(f"DROP TABLE {temp_table}")
            else:
                # 새 테이블 생성
                self.conn.execute(f"CREATE TABLE trading.{table_name} AS SELECT * FROM df")
            
            # 통계 업데이트
            row_count = self.conn.execute(f"SELECT COUNT(*) FROM trading.{table_name}").fetchone()[0]
            logger.info(f"{table_name} 저장 완료: 총 {row_count} rows")
            
        except Exception as e:
            logger.error(f"DB 저장 실패 {table_name}: {e}")
            raise
    
    async def ingest_symbols(self,
                            symbols: List[str],
                            start_date: str,
                            end_date: Optional[str] = None,
                            intervals: List[str] = ["1m"]):
        """여러 심볼 데이터 일괄 수집"""
        
        start_time = pd.to_datetime(start_date)
        end_time = pd.to_datetime(end_date) if end_date else datetime.now()
        
        logger.info(f"데이터 수집 시작: {symbols}")
        logger.info(f"기간: {start_time} ~ {end_time}")
        
        for symbol in symbols:
            logger.info(f"\n{'='*50}")
            logger.info(f"{symbol} 처리 시작")
            
            # Klines 수집
            for interval in intervals:
                try:
                    df_klines = await self.fetch_klines(
                        symbol=symbol,
                        interval=interval,
                        start_time=start_time,
                        end_time=end_time
                    )
                    
                    if not df_klines.empty:
                        table_name = f"klines_{interval}"
                        self.save_to_db(df_klines, table_name)
                        
                except Exception as e:
                    logger.error(f"{symbol} {interval} klines 수집 실패: {e}")
            
            # Funding rate 수집
            try:
                df_funding = await self.fetch_funding_rate(
                    symbol=symbol,
                    start_time=start_time,
                    end_time=end_time
                )
                
                if not df_funding.empty:
                    self.save_to_db(df_funding, "funding")
                    
            except Exception as e:
                logger.error(f"{symbol} funding rate 수집 실패: {e}")
            
            # 잠시 대기
            await asyncio.sleep(1)
        
        logger.info(f"\n{'='*50}")
        logger.info("모든 데이터 수집 완료!")
        
        # 최종 통계
        self._print_statistics()
    
    def _print_statistics(self):
        """수집된 데이터 통계 출력"""
        
        stats = {}
        
        tables = ['klines_1m', 'funding']
        
        for table in tables:
            try:
                result = self.conn.execute(f"""
                    SELECT 
                        COUNT(*) as total_rows,
                        COUNT(DISTINCT symbol) as symbols,
                        MIN(open_time) as start_time,
                        MAX(open_time) as end_time
                    FROM trading.{table}
                """).fetchone()
                
                if result:
                    stats[table] = {
                        'total_rows': result[0],
                        'symbols': result[1],
                        'start_time': result[2],
                        'end_time': result[3]
                    }
            except:
                pass
        
        logger.info("\n=== 데이터 수집 통계 ===")
        for table, stat in stats.items():
            logger.info(f"\n{table}:")
            logger.info(f"  총 레코드: {stat['total_rows']:,}")
            logger.info(f"  심볼 수: {stat['symbols']}")
            logger.info(f"  시작: {stat['start_time']}")
            logger.info(f"  종료: {stat['end_time']}")

# CLI 실행용
async def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='바이낸스 데이터 수집')
    parser.add_argument('--symbols', nargs='+', required=True, help='심볼 리스트')
    parser.add_argument('--since', required=True, help='시작 날짜 (YYYY-MM-DD)')
    parser.add_argument('--until', help='종료 날짜 (YYYY-MM-DD)')
    parser.add_argument('--intervals', nargs='+', default=['1m'], help='시간 간격')
    parser.add_argument('--db', default='data/trading.db', help='DB 경로')
    
    args = parser.parse_args()
    
    ingester = BinanceDataIngester(db_path=args.db)
    
    await ingester.ingest_symbols(
        symbols=args.symbols,
        start_date=args.since,
        end_date=args.until,
        intervals=args.intervals
    )

if __name__ == "__main__":
    asyncio.run(main())