"""웹소켓 스트림 처리"""

import asyncio
import websockets
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Callable, Any
from datetime import datetime
from collections import deque
import logging
import traceback

from ..utils.logging import Logger

logger = Logger.get_logger(__name__)

class WebSocketStream:
    """바이낸스 웹소켓 스트림 관리
    
    실시간 가격, 호가, 체결 데이터 수신 및 처리
    """
    
    def __init__(self, 
                 symbols: List[str],
                 testnet: bool = False):
        """
        Args:
            symbols: 심볼 리스트
            testnet: 테스트넷 여부
        """
        self.symbols = [s.lower() for s in symbols]
        self.testnet = testnet
        
        # 웹소켓 URL
        if testnet:
            self.ws_base = "wss://stream.binancefuture.com"
        else:
            self.ws_base = "wss://fstream.binance.com"
        
        # 연결 상태
        self.ws = None
        self.connected = False
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 10
        
        # 콜백 함수
        self.callbacks = {
            'trade': [],
            'depth': [],
            'kline': [],
            'mark_price': [],
            'funding': []
        }
        
        # 데이터 버퍼
        self.data_buffer = {
            'trades': deque(maxlen=1000),
            'depth': {},
            'klines': {},
            'mark_prices': {},
            'funding_rates': {}
        }
        
        # 통계
        self.stats = {
            'messages_received': 0,
            'messages_processed': 0,
            'errors': 0,
            'last_message_time': None,
            'connection_start': None
        }
    
    async def connect(self):
        """웹소켓 연결"""
        
        # 스트림 엔드포인트 생성
        streams = []
        
        for symbol in self.symbols:
            # 체결 스트림
            streams.append(f"{symbol}@trade")
            
            # 호가 스트림 (10 레벨)
            streams.append(f"{symbol}@depth10@100ms")
            
            # 1분봉 스트림
            streams.append(f"{symbol}@kline_1m")
            
            # 마크 가격 스트림
            streams.append(f"{symbol}@markPrice@1s")
        
        # 결합 스트림
        stream_url = f"{self.ws_base}/stream?streams={'/'.join(streams)}"
        
        try:
            logger.info(f"웹소켓 연결 시도: {stream_url}")
            
            self.ws = await websockets.connect(
                stream_url,
                ping_interval=20,
                ping_timeout=10,
                close_timeout=10
            )
            
            self.connected = True
            self.reconnect_attempts = 0
            self.stats['connection_start'] = datetime.now()
            
            logger.info("웹소켓 연결 성공")
            
        except Exception as e:
            logger.error(f"웹소켓 연결 실패: {e}")
            self.connected = False
            raise
    
    async def disconnect(self):
        """웹소켓 연결 해제"""
        
        if self.ws:
            await self.ws.close()
            self.ws = None
            self.connected = False
            
            logger.info("웹소켓 연결 해제")
    
    async def reconnect(self):
        """재연결"""
        
        if self.reconnect_attempts >= self.max_reconnect_attempts:
            logger.error("최대 재연결 시도 횟수 초과")
            return False
        
        self.reconnect_attempts += 1
        
        # 지수 백오프
        wait_time = min(2 ** self.reconnect_attempts, 60)
        
        logger.info(f"재연결 시도 {self.reconnect_attempts}/{self.max_reconnect_attempts} "
                   f"({wait_time}초 대기)")
        
        await asyncio.sleep(wait_time)
        
        try:
            await self.connect()
            return True
        except Exception as e:
            logger.error(f"재연결 실패: {e}")
            return False
    
    async def start_stream(self):
        """스트림 시작"""
        
        if not self.connected:
            await self.connect()
        
        logger.info("스트림 시작")
        
        while self.connected:
            try:
                # 메시지 수신
                message = await asyncio.wait_for(
                    self.ws.recv(),
                    timeout=30  # 30초 타임아웃
                )
                
                # 메시지 처리
                await self.process_message(message)
                
            except asyncio.TimeoutError:
                logger.warning("메시지 수신 타임아웃")
                
                # 핑 전송
                await self.ws.ping()
                
            except websockets.ConnectionClosed:
                logger.warning("웹소켓 연결 끊김")
                self.connected = False
                
                # 재연결 시도
                if await self.reconnect():
                    continue
                else:
                    break
                    
            except Exception as e:
                logger.error(f"스트림 에러: {e}")
                logger.error(traceback.format_exc())
                self.stats['errors'] += 1
                
                if self.stats['errors'] > 100:
                    logger.error("에러 횟수 초과, 스트림 중단")
                    break
    
    async def process_message(self, message: str):
        """메시지 처리
        
        Args:
            message: 웹소켓 메시지
        """
        
        self.stats['messages_received'] += 1
        self.stats['last_message_time'] = datetime.now()
        
        try:
            data = json.loads(message)
            
            # 스트림 타입 확인
            if 'stream' not in data:
                return
            
            stream = data['stream']
            payload = data['data']
            
            # 스트림 타입별 처리
            if '@trade' in stream:
                await self.process_trade(payload)
                
            elif '@depth' in stream:
                await self.process_depth(payload)
                
            elif '@kline' in stream:
                await self.process_kline(payload)
                
            elif '@markPrice' in stream:
                await self.process_mark_price(payload)
            
            self.stats['messages_processed'] += 1
            
        except json.JSONDecodeError:
            logger.error(f"JSON 디코드 에러: {message[:100]}")
        except Exception as e:
            logger.error(f"메시지 처리 에러: {e}")
    
    async def process_trade(self, data: Dict):
        """체결 데이터 처리"""
        
        trade = {
            'symbol': data['s'],
            'price': float(data['p']),
            'quantity': float(data['q']),
            'time': pd.to_datetime(data['T'], unit='ms'),
            'is_buyer_maker': data['m'],
            'trade_id': data['t']
        }
        
        # 버퍼에 추가
        self.data_buffer['trades'].append(trade)
        
        # 콜백 실행
        for callback in self.callbacks['trade']:
            try:
                await callback(trade)
            except Exception as e:
                logger.error(f"Trade 콜백 에러: {e}")
    
    async def process_depth(self, data: Dict):
        """호가 데이터 처리"""
        
        symbol = data['s']
        
        depth = {
            'symbol': symbol,
            'time': pd.to_datetime(data['T'], unit='ms'),
            'bids': [(float(p), float(q)) for p, q in data['b']],
            'asks': [(float(p), float(q)) for p, q in data['a']]
        }
        
        # 최우선 호가
        if depth['bids']:
            depth['best_bid'] = depth['bids'][0][0]
            depth['best_bid_qty'] = depth['bids'][0][1]
        else:
            depth['best_bid'] = 0
            depth['best_bid_qty'] = 0
        
        if depth['asks']:
            depth['best_ask'] = depth['asks'][0][0]
            depth['best_ask_qty'] = depth['asks'][0][1]
        else:
            depth['best_ask'] = 0
            depth['best_ask_qty'] = 0
        
        # 스프레드
        if depth['best_bid'] > 0 and depth['best_ask'] > 0:
            depth['spread'] = depth['best_ask'] - depth['best_bid']
            depth['spread_bps'] = depth['spread'] / depth['best_bid'] * 10000
        else:
            depth['spread'] = 0
            depth['spread_bps'] = 0
        
        # 심도
        depth['bid_depth'] = sum(q for _, q in depth['bids'])
        depth['ask_depth'] = sum(q for _, q in depth['asks'])
        depth['total_depth'] = depth['bid_depth'] + depth['ask_depth']
        
        # 버퍼 업데이트
        self.data_buffer['depth'][symbol] = depth
        
        # 콜백 실행
        for callback in self.callbacks['depth']:
            try:
                await callback(depth)
            except Exception as e:
                logger.error(f"Depth 콜백 에러: {e}")
    
    async def process_kline(self, data: Dict):
        """K라인 데이터 처리"""
        
        kline_data = data['k']
        symbol = kline_data['s']
        
        kline = {
            'symbol': symbol,
            'open_time': pd.to_datetime(kline_data['t'], unit='ms'),
            'close_time': pd.to_datetime(kline_data['T'], unit='ms'),
            'open': float(kline_data['o']),
            'high': float(kline_data['h']),
            'low': float(kline_data['l']),
            'close': float(kline_data['c']),
            'volume': float(kline_data['v']),
            'trades': int(kline_data['n']),
            'is_closed': kline_data['x']
        }
        
        # 버퍼 업데이트
        if symbol not in self.data_buffer['klines']:
            self.data_buffer['klines'][symbol] = deque(maxlen=100)
        
        self.data_buffer['klines'][symbol].append(kline)
        
        # 콜백 실행 (종료된 캔들만)
        if kline['is_closed']:
            for callback in self.callbacks['kline']:
                try:
                    await callback(kline)
                except Exception as e:
                    logger.error(f"Kline 콜백 에러: {e}")
    
    async def process_mark_price(self, data: Dict):
        """마크 가격 및 펀딩 데이터 처리"""
        
        symbol = data['s']
        
        mark_data = {
            'symbol': symbol,
            'mark_price': float(data['p']),
            'index_price': float(data.get('i', data['p'])),
            'funding_rate': float(data.get('r', 0)),
            'next_funding_time': pd.to_datetime(data.get('T', 0), unit='ms'),
            'time': pd.to_datetime(data['E'], unit='ms')
        }
        
        # 버퍼 업데이트
        self.data_buffer['mark_prices'][symbol] = mark_data['mark_price']
        
        if mark_data['funding_rate'] != 0:
            self.data_buffer['funding_rates'][symbol] = mark_data['funding_rate']
        
        # 콜백 실행
        for callback in self.callbacks['mark_price']:
            try:
                await callback(mark_data)
            except Exception as e:
                logger.error(f"Mark price 콜백 에러: {e}")
        
        # 펀딩 콜백
        if mark_data['funding_rate'] != 0:
            for callback in self.callbacks['funding']:
                try:
                    await callback(mark_data)
                except Exception as e:
                    logger.error(f"Funding 콜백 에러: {e}")
    
    def register_callback(self, 
                         event_type: str,
                         callback: Callable):
        """콜백 등록
        
        Args:
            event_type: 이벤트 타입 (trade, depth, kline, mark_price, funding)
            callback: 콜백 함수
        """
        
        if event_type in self.callbacks:
            self.callbacks[event_type].append(callback)
            logger.debug(f"콜백 등록: {event_type}")
        else:
            logger.warning(f"알 수 없는 이벤트 타입: {event_type}")
    
    def get_latest_price(self, symbol: str) -> Optional[float]:
        """최신 가격 조회"""
        
        # 호가에서 중간 가격
        if symbol in self.data_buffer['depth']:
            depth = self.data_buffer['depth'][symbol]
            if depth['best_bid'] > 0 and depth['best_ask'] > 0:
                return (depth['best_bid'] + depth['best_ask']) / 2
        
        # 마크 가격
        if symbol in self.data_buffer['mark_prices']:
            return self.data_buffer['mark_prices'][symbol]
        
        return None
    
    def get_latest_depth(self, symbol: str) -> Optional[Dict]:
        """최신 호가 조회"""
        
        return self.data_buffer['depth'].get(symbol)
    
    def get_recent_trades(self, n: int = 100) -> List[Dict]:
        """최근 체결 조회"""
        
        return list(self.data_buffer['trades'])[-n:]
    
    def get_stats(self) -> Dict:
        """통계 조회"""
        
        stats = self.stats.copy()
        
        # 연결 시간
        if stats['connection_start']:
            uptime = (datetime.now() - stats['connection_start']).total_seconds()
            stats['uptime_seconds'] = uptime
        
        # 메시지 처리율
        if stats['messages_received'] > 0:
            stats['process_rate'] = stats['messages_processed'] / stats['messages_received']
        
        # 에러율
        if stats['messages_received'] > 0:
            stats['error_rate'] = stats['errors'] / stats['messages_received']
        
        return stats