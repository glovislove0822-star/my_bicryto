"""Order Flow Imbalance (OFI) 계산 모듈"""

import pandas as pd
import numpy as np
import duckdb
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from datetime import datetime, timedelta
import logging
from collections import deque
import asyncio
import websockets
import json

from ..utils.logging import Logger
from ..utils.math import MathUtils

logger = Logger.get_logger(__name__)

class OFICalculator:
    """Order Flow Imbalance 계산기"""
    
    def __init__(self, db_path: str = "data/trading.db"):
        """
        Args:
            db_path: DuckDB 데이터베이스 경로
        """
        self.db_path = Path(db_path)
        self.conn = duckdb.connect(str(self.db_path))
        self.math_utils = MathUtils()
        
        # OFI 계산 파라미터
        self.params = {
            'depth_levels': 5,  # 사용할 호가 레벨 수
            'decay_factor': 0.95,  # 시간 감쇠 계수
            'min_depth': 100,  # 최소 심도 임계값
            'outlier_threshold': 5  # 이상치 제거 임계값 (표준편차)
        }
        
        # 실시간 버퍼 (심볼별)
        self.depth_buffer = {}
        self.trade_buffer = {}
    
    def calculate_ofi(self,
                     symbol: str,
                     start_time: datetime,
                     end_time: datetime,
                     interval: str = '1m') -> pd.DataFrame:
        """OFI 계산
        
        Args:
            symbol: 심볼
            start_time: 시작 시간
            end_time: 종료 시간
            interval: 집계 간격
        
        Returns:
            OFI 데이터프레임
        """
        
        # 호가 데이터 로드
        depth_df = self._load_depth_data(symbol, start_time, end_time)
        
        if depth_df.empty:
            logger.warning(f"{symbol} 호가 데이터 없음, 시뮬레이션 데이터 생성")
            depth_df = self._generate_simulated_depth(symbol, start_time, end_time)
        
        # 거래 데이터 로드
        trades_df = self._load_trades_data(symbol, start_time, end_time)
        
        if trades_df.empty:
            logger.warning(f"{symbol} 거래 데이터 없음, 시뮬레이션 데이터 생성")
            trades_df = self._generate_simulated_trades(symbol, start_time, end_time)
        
        # OFI 계산
        ofi_df = self._compute_ofi(depth_df, trades_df, interval)
        
        # 추가 마이크로구조 지표
        ofi_df = self._add_microstructure_metrics(ofi_df)
        
        return ofi_df
    
    def _load_depth_data(self, 
                        symbol: str,
                        start_time: datetime,
                        end_time: datetime) -> pd.DataFrame:
        """호가 데이터 로드"""
        
        query = f"""
            SELECT 
                ts,
                best_bid,
                best_ask,
                bid_sz,
                ask_sz,
                spread,
                bid_levels,
                ask_levels
            FROM trading.depth
            WHERE symbol = '{symbol}'
                AND ts >= '{start_time}'
                AND ts <= '{end_time}'
            ORDER BY ts
        """
        
        try:
            df = self.conn.execute(query).df()
            if not df.empty:
                df['ts'] = pd.to_datetime(df['ts'])
                df.set_index('ts', inplace=True)
            return df
        except:
            return pd.DataFrame()
    
    def _load_trades_data(self,
                         symbol: str,
                         start_time: datetime,
                         end_time: datetime) -> pd.DataFrame:
        """거래 데이터 로드"""
        
        query = f"""
            SELECT 
                ts,
                price,
                qty,
                side
            FROM trading.trades
            WHERE symbol = '{symbol}'
                AND ts >= '{start_time}'
                AND ts <= '{end_time}'
            ORDER BY ts
        """
        
        try:
            df = self.conn.execute(query).df()
            if not df.empty:
                df['ts'] = pd.to_datetime(df['ts'])
                df.set_index('ts', inplace=True)
            return df
        except:
            return pd.DataFrame()
    
    def _generate_simulated_depth(self,
                                 symbol: str,
                                 start_time: datetime,
                                 end_time: datetime) -> pd.DataFrame:
        """시뮬레이션 호가 데이터 생성"""
        
        # 가격 데이터 기반으로 호가 시뮬레이션
        price_query = f"""
            SELECT 
                open_time as ts,
                close as price,
                volume
            FROM trading.klines_1m
            WHERE symbol = '{symbol}'
                AND open_time >= '{start_time}'
                AND open_time <= '{end_time}'
            ORDER BY open_time
        """
        
        price_df = self.conn.execute(price_query).df()
        
        if price_df.empty:
            return pd.DataFrame()
        
        price_df['ts'] = pd.to_datetime(price_df['ts'])
        price_df.set_index('ts', inplace=True)
        
        # 호가 시뮬레이션
        depth_data = []
        
        for idx, row in price_df.iterrows():
            price = row['price']
            volume = row['volume']
            
            # 스프레드 시뮬레이션 (변동성 기반)
            volatility = price_df['price'].pct_change().rolling(20).std().loc[idx]
            if pd.isna(volatility):
                volatility = 0.001
            
            spread = max(0.01, price * volatility * 2)  # 변동성의 2배
            
            best_bid = price - spread / 2
            best_ask = price + spread / 2
            
            # 심도 시뮬레이션 (볼륨 기반)
            base_depth = volume * 10  # 거래량의 10배
            bid_sz = base_depth * np.random.uniform(0.8, 1.2)
            ask_sz = base_depth * np.random.uniform(0.8, 1.2)
            
            # 레벨별 호가 생성
            bid_levels = []
            ask_levels = []
            
            for i in range(self.params['depth_levels']):
                level_spread = spread * (i + 1)
                
                bid_levels.append({
                    'price': best_bid - level_spread,
                    'size': bid_sz * np.exp(-i * 0.3)  # 지수 감소
                })
                
                ask_levels.append({
                    'price': best_ask + level_spread,
                    'size': ask_sz * np.exp(-i * 0.3)
                })
            
            depth_data.append({
                'ts': idx,
                'best_bid': best_bid,
                'best_ask': best_ask,
                'bid_sz': bid_sz,
                'ask_sz': ask_sz,
                'spread': spread,
                'bid_levels': json.dumps(bid_levels),
                'ask_levels': json.dumps(ask_levels)
            })
        
        return pd.DataFrame(depth_data).set_index('ts')
    
    def _generate_simulated_trades(self,
                                  symbol: str,
                                  start_time: datetime,
                                  end_time: datetime) -> pd.DataFrame:
        """시뮬레이션 거래 데이터 생성"""
        
        # 가격/볼륨 데이터 로드
        price_query = f"""
            SELECT 
                open_time as ts,
                open,
                high,
                low,
                close,
                volume,
                trade_count
            FROM trading.klines_1m
            WHERE symbol = '{symbol}'
                AND open_time >= '{start_time}'
                AND open_time <= '{end_time}'
            ORDER BY open_time
        """
        
        price_df = self.conn.execute(price_query).df()
        
        if price_df.empty:
            return pd.DataFrame()
        
        price_df['ts'] = pd.to_datetime(price_df['ts'])
        
        # 거래 시뮬레이션
        trades_data = []
        
        for _, row in price_df.iterrows():
            # 분당 거래 수
            num_trades = max(1, int(row['trade_count']) if row['trade_count'] > 0 else 10)
            
            # 시간 분산
            timestamps = pd.date_range(
                start=row['ts'],
                end=row['ts'] + pd.Timedelta(minutes=1),
                periods=num_trades
            )[:-1]  # 마지막 제외
            
            # 가격 분산 (OHLC 내)
            prices = np.random.uniform(row['low'], row['high'], num_trades)
            
            # 거래량 분산
            avg_qty = row['volume'] / num_trades
            quantities = np.random.exponential(avg_qty, num_trades)
            
            # 매수/매도 방향 (가격 방향 기반)
            if row['close'] > row['open']:
                buy_prob = 0.6  # 상승 시 매수 확률 높음
            else:
                buy_prob = 0.4
            
            sides = np.random.choice(['buy', 'sell'], num_trades, p=[buy_prob, 1-buy_prob])
            
            for ts, price, qty, side in zip(timestamps, prices, quantities, sides):
                trades_data.append({
                    'ts': ts,
                    'price': price,
                    'qty': qty,
                    'side': side
                })
        
        return pd.DataFrame(trades_data).set_index('ts')
    
    def _compute_ofi(self,
                    depth_df: pd.DataFrame,
                    trades_df: pd.DataFrame,
                    interval: str) -> pd.DataFrame:
        """OFI 계산 로직"""
        
        if depth_df.empty:
            return pd.DataFrame()
        
        # 시간 간격별 리샘플링
        freq_map = {'1m': '1T', '3m': '3T', '5m': '5T'}
        freq = freq_map.get(interval, '1T')
        
        # OFI 계산
        ofi_values = []
        
        for ts in depth_df.index:
            # 현재 호가
            bid = depth_df.loc[ts, 'best_bid']
            ask = depth_df.loc[ts, 'best_ask']
            bid_size = depth_df.loc[ts, 'bid_sz']
            ask_size = depth_df.loc[ts, 'ask_sz']
            
            # 이전 호가
            if ts > depth_df.index[0]:
                prev_idx = depth_df.index[depth_df.index < ts][-1]
                prev_bid = depth_df.loc[prev_idx, 'best_bid']
                prev_ask = depth_df.loc[prev_idx, 'best_ask']
                prev_bid_size = depth_df.loc[prev_idx, 'bid_sz']
                prev_ask_size = depth_df.loc[prev_idx, 'ask_sz']
            else:
                prev_bid = bid
                prev_ask = ask
                prev_bid_size = bid_size
                prev_ask_size = ask_size
            
            # OFI 계산 (Cont & Larrard 2013 방식)
            # OFI = ΔBid_size * I(Bid_up) - ΔBid_size * I(Bid_down) 
            #     - ΔAsk_size * I(Ask_up) + ΔAsk_size * I(Ask_down)
            
            ofi = 0
            
            # Bid 변화
            if bid > prev_bid:  # Bid 상승
                ofi += bid_size
            elif bid < prev_bid:  # Bid 하락
                ofi -= prev_bid_size
            else:  # Bid 동일
                ofi += (bid_size - prev_bid_size)
            
            # Ask 변화
            if ask < prev_ask:  # Ask 하락 (좋음)
                ofi += ask_size
            elif ask > prev_ask:  # Ask 상승
                ofi -= prev_ask_size
            else:  # Ask 동일
                ofi -= (ask_size - prev_ask_size)
            
            ofi_values.append({
                'ts': ts,
                'ofi': ofi,
                'bid': bid,
                'ask': ask,
                'spread': ask - bid,
                'mid': (bid + ask) / 2,
                'bid_size': bid_size,
                'ask_size': ask_size
            })
        
        ofi_df = pd.DataFrame(ofi_values).set_index('ts')
        
        # 리샘플링
        ofi_resampled = ofi_df.resample(freq).agg({
            'ofi': 'sum',
            'bid': 'last',
            'ask': 'last',
            'spread': 'mean',
            'mid': 'last',
            'bid_size': 'mean',
            'ask_size': 'mean'
        })
        
        # 정규화
        ofi_resampled['ofi_normalized'] = ofi_resampled['ofi'] / (
            ofi_resampled['bid_size'] + ofi_resampled['ask_size']
        )
        
        return ofi_resampled
    
    def _add_microstructure_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """추가 마이크로구조 지표"""
        
        if df.empty:
            return df
        
        # Queue Imbalance
        df['queue_imbalance'] = df['bid_size'] / (df['bid_size'] + df['ask_size'])
        
        # Spread (basis points)
        df['spread_bps'] = (df['spread'] / df['mid']) * 10000
        
        # Total Depth
        df['depth_total'] = df['bid_size'] + df['ask_size']
        
        # OFI 이동평균
        df['ofi_ma_5'] = df['ofi'].rolling(5).mean()
        df['ofi_ma_20'] = df['ofi'].rolling(20).mean()
        
        # OFI 표준편차
        df['ofi_std'] = df['ofi'].rolling(20).std()
        
        # OFI Z-score
        df['ofi_z'] = (df['ofi'] - df['ofi_ma_20']) / df['ofi_std']
        df['ofi_z'] = df['ofi_z'].fillna(0)
        
        # 누적 OFI
        df['ofi_cumsum'] = df['ofi'].cumsum()
        
        # OFI 모멘텀
        df['ofi_momentum'] = df['ofi'] - df['ofi'].shift(5)
        
        # Liquidity Consumption (유동성 소비율)
        df['liquidity_consumption'] = abs(df['ofi']) / df['depth_total']
        
        # Price Impact 추정
        df['price_impact'] = df['mid'].pct_change() / (df['ofi_normalized'] + 0.0001)
        
        return df
    
    async def stream_ofi_realtime(self, symbol: str, ws_url: str):
        """실시간 OFI 스트리밍 (WebSocket)"""
        
        async with websockets.connect(ws_url) as websocket:
            # 구독 메시지
            subscribe_msg = {
                "method": "SUBSCRIBE",
                "params": [
                    f"{symbol.lower()}@depth@100ms",
                    f"{symbol.lower()}@trade"
                ],
                "id": 1
            }
            
            await websocket.send(json.dumps(subscribe_msg))
            
            # 버퍼 초기화
            if symbol not in self.depth_buffer:
                self.depth_buffer[symbol] = deque(maxlen=1000)
                self.trade_buffer[symbol] = deque(maxlen=1000)
            
            while True:
                try:
                    msg = await websocket.recv()
                    data = json.loads(msg)
                    
                    if 'e' in data:
                        if data['e'] == 'depthUpdate':
                            # 호가 업데이트
                            self._process_depth_update(symbol, data)
                        elif data['e'] == 'trade':
                            # 체결 업데이트
                            self._process_trade_update(symbol, data)
                        
                        # OFI 계산 (매 100ms)
                        if len(self.depth_buffer[symbol]) >= 2:
                            current_ofi = self._calculate_realtime_ofi(symbol)
                            
                            # DB 저장 또는 시그널 생성
                            if abs(current_ofi) > self.params['outlier_threshold']:
                                logger.info(f"{symbol} High OFI: {current_ofi:.2f}")
                
                except Exception as e:
                    logger.error(f"OFI 스트리밍 에러: {e}")
                    await asyncio.sleep(1)
    
    def _process_depth_update(self, symbol: str, data: Dict):
        """호가 업데이트 처리"""
        
        depth_snapshot = {
            'ts': datetime.now(),
            'bids': data.get('b', []),
            'asks': data.get('a', []),
            'event_time': data.get('E', 0)
        }
        
        self.depth_buffer[symbol].append(depth_snapshot)
    
    def _process_trade_update(self, symbol: str, data: Dict):
        """체결 업데이트 처리"""
        
        trade = {
            'ts': datetime.now(),
            'price': float(data.get('p', 0)),
            'qty': float(data.get('q', 0)),
            'is_buyer_maker': data.get('m', False),
            'trade_time': data.get('T', 0)
        }
        
        self.trade_buffer[symbol].append(trade)
    
    def _calculate_realtime_ofi(self, symbol: str) -> float:
        """실시간 OFI 계산"""
        
        if len(self.depth_buffer[symbol]) < 2:
            return 0
        
        current = self.depth_buffer[symbol][-1]
        previous = self.depth_buffer[symbol][-2]
        
        # 간단한 OFI 계산
        ofi = 0
        
        # Bid 변화
        if current['bids'] and previous['bids']:
            curr_bid = float(current['bids'][0][0])
            prev_bid = float(previous['bids'][0][0])
            curr_bid_size = float(current['bids'][0][1])
            prev_bid_size = float(previous['bids'][0][1])
            
            if curr_bid > prev_bid:
                ofi += curr_bid_size
            elif curr_bid < prev_bid:
                ofi -= prev_bid_size
        
        # Ask 변화
        if current['asks'] and previous['asks']:
            curr_ask = float(current['asks'][0][0])
            prev_ask = float(previous['asks'][0][0])
            curr_ask_size = float(current['asks'][0][1])
            prev_ask_size = float(previous['asks'][0][1])
            
            if curr_ask < prev_ask:
                ofi += curr_ask_size
            elif curr_ask > prev_ask:
                ofi -= prev_ask_size
        
        return ofi

# CLI 실행용
def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='OFI 계산')
    parser.add_argument('--symbol', required=True, help='심볼')
    parser.add_argument('--start', required=True, help='시작 시간')
    parser.add_argument('--end', required=True, help='종료 시간')
    parser.add_argument('--interval', default='1m', help='집계 간격')
    parser.add_argument('--db', default='data/trading.db', help='DB 경로')
    
    args = parser.parse_args()
    
    calculator = OFICalculator(db_path=args.db)
    
    ofi_df = calculator.calculate_ofi(
        symbol=args.symbol,
        start_time=pd.to_datetime(args.start),
        end_time=pd.to_datetime(args.end),
        interval=args.interval
    )
    
    if not ofi_df.empty:
        logger.info("\n=== OFI 통계 ===")
        logger.info(f"평균 OFI: {ofi_df['ofi'].mean():.2f}")
        logger.info(f"OFI 표준편차: {ofi_df['ofi'].std():.2f}")
        logger.info(f"평균 스프레드: {ofi_df['spread_bps'].mean():.2f} bps")
        logger.info(f"평균 Queue Imbalance: {ofi_df['queue_imbalance'].mean():.3f}")

if __name__ == "__main__":
    main()