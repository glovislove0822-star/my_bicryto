"""바이낸스 거래소 클라이언트"""

import asyncio
from binance import AsyncClient, BinanceSocketManager
from binance.enums import *
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging
import hmac
import hashlib
import time

from ..utils.logging import Logger

logger = Logger.get_logger(__name__)

class BinanceClient:
    """바이낸스 선물 거래소 클라이언트
    
    실제 거래소 API 연동
    """
    
    def __init__(self, config: Dict):
        """
        Args:
            config: 거래소 설정
        """
        self.config = config
        
        # API 키
        self.api_key = config.get('api_key')
        self.api_secret = config.get('api_secret')
        
        # 테스트넷 설정
        self.testnet = config.get('testnet', False)
        
        if self.testnet:
            self.base_url = "https://testnet.binancefuture.com"
            self.ws_url = "wss://stream.binancefuture.com"
        else:
            self.base_url = "https://fapi.binance.com"
            self.ws_url = "wss://fstream.binance.com"
        
        # 클라이언트
        self.client = None
        self.bm = None
        
        # 레이트 리미트
        self.rate_limits = {
            'weight': {'limit': 2400, 'window': 60, 'current': 0},
            'orders': {'limit': 300, 'window': 10, 'current': 0}
        }
        
        logger.info(f"바이낸스 클라이언트 초기화 (testnet={self.testnet})")
    
    async def connect(self):
        """연결"""
        
        self.client = await AsyncClient.create(
            api_key=self.api_key,
            api_secret=self.api_secret,
            testnet=self.testnet
        )
        
        # 소켓 매니저
        self.bm = BinanceSocketManager(self.client)
        
        # 계정 정보 확인
        account = await self.get_account_info()
        logger.info(f"바이낸스 연결 성공: Balance=${account['balance']:,.2f}")
    
    async def disconnect(self):
        """연결 해제"""
        
        if self.client:
            await self.client.close_connection()
            logger.info("바이낸스 연결 해제")
    
    async def get_account_info(self) -> Dict:
        """계정 정보 조회"""
        
        try:
            account = await self.client.futures_account()
            
            # USDT 잔고
            usdt_balance = 0
            for asset in account['assets']:
                if asset['asset'] == 'USDT':
                    usdt_balance = float(asset['availableBalance'])
                    break
            
            return {
                'balance': usdt_balance,
                'total_margin_balance': float(account['totalMarginBalance']),
                'total_wallet_balance': float(account['totalWalletBalance']),
                'total_unrealized_profit': float(account['totalUnrealizedProfit']),
                'positions': len([p for p in account['positions'] if float(p['positionAmt']) != 0])
            }
            
        except Exception as e:
            logger.error(f"계정 정보 조회 실패: {e}")
            return {'balance': 0}
    
    async def get_positions(self) -> List[Dict]:
        """포지션 조회"""
        
        try:
            positions = await self.client.futures_position_information()
            
            # 활성 포지션만 필터링
            active_positions = []
            
            for pos in positions:
                if float(pos['positionAmt']) != 0:
                    active_positions.append({
                        'symbol': pos['symbol'],
                        'positionAmt': float(pos['positionAmt']),
                        'entryPrice': float(pos['entryPrice']),
                        'markPrice': float(pos['markPrice']),
                        'unrealizedPnl': float(pos['unRealizedProfit']),
                        'marginType': pos['marginType'],
                        'leverage': int(pos['leverage'])
                    })
            
            return active_positions
            
        except Exception as e:
            logger.error(f"포지션 조회 실패: {e}")
            return []
    
    async def create_order(self, **params) -> Dict:
        """주문 생성
        
        Args:
            **params: 주문 파라미터
            
        Returns:
            주문 결과
        """
        
        try:
            # 레이트 리미트 체크
            await self._check_rate_limit('orders')
            
            # 필수 파라미터
            symbol = params['symbol']
            side = params['side']  # BUY or SELL
            order_type = params.get('type', 'MARKET')
            quantity = params['quantity']
            
            # 주문 파라미터
            order_params = {
                'symbol': symbol,
                'side': side,
                'type': order_type,
                'quantity': quantity
            }
            
            # 옵션 파라미터
            if order_type == 'LIMIT':
                order_params['price'] = params['price']
                order_params['timeInForce'] = params.get('timeInForce', 'GTC')
            
            if params.get('reduceOnly'):
                order_params['reduceOnly'] = True
            
            if params.get('postOnly'):
                order_params['timeInForce'] = 'GTX'
            
            # 주문 실행
            if order_type == 'MARKET':
                result = await self.client.futures_create_order(
                    symbol=symbol,
                    side=side,
                    type=ORDER_TYPE_MARKET,
                    quantity=quantity,
                    reduceOnly=params.get('reduceOnly', False)
                )
            else:
                result = await self.client.futures_create_order(**order_params)
            
            logger.info(f"주문 생성: {symbol} {side} {quantity} @ {order_type}")
            
            return result
            
        except Exception as e:
            logger.error(f"주문 생성 실패: {e}")
            return {'success': False, 'error': str(e)}
    
    async def cancel_order(self, symbol: str, orderId: int) -> Dict:
        """주문 취소
        
        Args:
            symbol: 심볼
            orderId: 주문 ID
            
        Returns:
            취소 결과
        """
        
        try:
            result = await self.client.futures_cancel_order(
                symbol=symbol,
                orderId=orderId
            )
            
            logger.info(f"주문 취소: {symbol} {orderId}")
            
            return result
            
        except Exception as e:
            logger.error(f"주문 취소 실패: {e}")
            return {'success': False, 'error': str(e)}
    
    async def get_order(self, symbol: str, orderId: int) -> Dict:
        """주문 조회
        
        Args:
            symbol: 심볼
            orderId: 주문 ID
            
        Returns:
            주문 정보
        """
        
        try:
            order = await self.client.futures_get_order(
                symbol=symbol,
                orderId=orderId
            )
            
            return order
            
        except Exception as e:
            logger.error(f"주문 조회 실패: {e}")
            return {}
    
    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict]:
        """미체결 주문 조회
        
        Args:
            symbol: 심볼 (선택)
            
        Returns:
            미체결 주문 리스트
        """
        
        try:
            if symbol:
                orders = await self.client.futures_get_open_orders(symbol=symbol)
            else:
                orders = await self.client.futures_get_open_orders()
            
            return orders
            
        except Exception as e:
            logger.error(f"미체결 주문 조회 실패: {e}")
            return []
    
    async def get_order_book(self, symbol: str, limit: int = 10) -> Dict:
        """호가 조회
        
        Args:
            symbol: 심볼
            limit: 호가 레벨 수
            
        Returns:
            호가 정보
        """
        
        try:
            depth = await self.client.futures_order_book(
                symbol=symbol,
                limit=limit
            )
            
            return {
                'bids': [[float(p), float(q)] for p, q in depth['bids']],
                'asks': [[float(p), float(q)] for p, q in depth['asks']],
                'lastUpdateId': depth['lastUpdateId']
            }
            
        except Exception as e:
            logger.error(f"호가 조회 실패: {e}")
            return {'bids': [], 'asks': []}
    
    async def get_recent_trades(self, symbol: str, limit: int = 100) -> List[Dict]:
        """최근 체결 조회
        
        Args:
            symbol: 심볼
            limit: 체결 수
            
        Returns:
            체결 리스트
        """
        
        try:
            trades = await self.client.futures_recent_trades(
                symbol=symbol,
                limit=limit
            )
            
            return [
                {
                    'id': t['id'],
                    'price': float(t['price']),
                    'qty': float(t['qty']),
                    'time': t['time'],
                    'isBuyerMaker': t['isBuyerMaker']
                }
                for t in trades
            ]
            
        except Exception as e:
            logger.error(f"체결 조회 실패: {e}")
            return []
    
    async def get_klines(self,
                        symbol: str,
                        interval: str,
                        start_time: Optional[int] = None,
                        end_time: Optional[int] = None,
                        limit: int = 500) -> List[List]:
        """K라인 조회
        
        Args:
            symbol: 심볼
            interval: 간격 (1m, 3m, 5m, 15m, 30m, 1h, 4h, 1d)
            start_time: 시작 시간 (밀리초)
            end_time: 종료 시간 (밀리초)
            limit: 개수
            
        Returns:
            K라인 데이터
        """
        
        try:
            klines = await self.client.futures_klines(
                symbol=symbol,
                interval=interval,
                startTime=start_time,
                endTime=end_time,
                limit=limit
            )
            
            return klines
            
        except Exception as e:
            logger.error(f"K라인 조회 실패: {e}")
            return []
    
    async def get_funding_rate(self, symbol: str) -> Dict:
        """펀딩 레이트 조회
        
        Args:
            symbol: 심볼
            
        Returns:
            펀딩 정보
        """
        
        try:
            funding = await self.client.futures_funding_rate(
                symbol=symbol,
                limit=1
            )
            
            if funding:
                return {
                    'symbol': funding[0]['symbol'],
                    'fundingRate': float(funding[0]['fundingRate']),
                    'fundingTime': funding[0]['fundingTime']
                }
            
            return {}
            
        except Exception as e:
            logger.error(f"펀딩 레이트 조회 실패: {e}")
            return {}
    
    async def set_leverage(self, symbol: str, leverage: int) -> bool:
        """레버리지 설정
        
        Args:
            symbol: 심볼
            leverage: 레버리지
            
        Returns:
            성공 여부
        """
        
        try:
            result = await self.client.futures_change_leverage(
                symbol=symbol,
                leverage=leverage
            )
            
            logger.info(f"레버리지 설정: {symbol} {leverage}x")
            
            return True
            
        except Exception as e:
            logger.error(f"레버리지 설정 실패: {e}")
            return False
    
    async def set_margin_type(self, symbol: str, margin_type: str) -> bool:
        """마진 타입 설정
        
        Args:
            symbol: 심볼
            margin_type: ISOLATED or CROSSED
            
        Returns:
            성공 여부
        """
        
        try:
            result = await self.client.futures_change_margin_type(
                symbol=symbol,
                marginType=margin_type
            )
            
            logger.info(f"마진 타입 설정: {symbol} {margin_type}")
            
            return True
            
        except Exception as e:
            # 이미 설정된 경우 에러 무시
            if "No need to change margin type" in str(e):
                return True
            
            logger.error(f"마진 타입 설정 실패: {e}")
            return False
    
    async def _check_rate_limit(self, limit_type: str):
        """레이트 리미트 체크
        
        Args:
            limit_type: 리미트 타입 (weight, orders)
        """
        
        limit_info = self.rate_limits[limit_type]
        
        # 시간 윈도우 체크
        # TODO: 실제 구현 필요
        
        # 리미트 도달 시 대기
        if limit_info['current'] >= limit_info['limit']:
            wait_time = limit_info['window']
            logger.warning(f"레이트 리미트 도달, {wait_time}초 대기")
            await asyncio.sleep(wait_time)
            limit_info['current'] = 0
        
        limit_info['current'] += 1