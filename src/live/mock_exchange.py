"""모의 거래소 클라이언트 (드라이런용)"""

import asyncio
import random
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging

from ..utils.logging import Logger

logger = Logger.get_logger(__name__)

class MockExchangeClient:
    """모의 거래소 클라이언트
    
    드라이런 모드에서 실제 거래소 대신 사용
    """
    
    def __init__(self, config: Dict):
        """
        Args:
            config: 설정
        """
        self.config = config
        
        # 모의 계정
        self.balance = config.get('initial_capital', 100000)
        self.positions = {}
        self.orders = {}
        self.order_counter = 0
        
        # 모의 시장 데이터
        self.mock_prices = {
            'BTCUSDT': 50000,
            'ETHUSDT': 3000
        }
        
        logger.info("모의 거래소 클라이언트 초기화")
    
    async def get_account_info(self) -> Dict:
        """계정 정보 조회"""
        
        return {
            'balance': self.balance,
            'positions': len(self.positions),
            'margin_ratio': 0.1
        }
    
    async def get_positions(self) -> List[Dict]:
        """포지션 조회"""
        
        positions = []
        
        for symbol, pos in self.positions.items():
            positions.append({
                'symbol': symbol,
                'positionAmt': pos['size'] * (1 if pos['side'] == 'long' else -1),
                'entryPrice': pos['entry_price'],
                'unrealizedPnl': pos.get('unrealized_pnl', 0)
            })
        
        return positions
    
    async def create_order(self, **params) -> Dict:
        """주문 생성"""
        
        self.order_counter += 1
        order_id = f"MOCK_{self.order_counter}"
        
        # 모의 체결
        symbol = params['symbol']
        side = params['side'].lower()
        quantity = params['quantity']
        
        # 현재 가격 (랜덤 변동)
        current_price = self.mock_prices.get(symbol, 50000)
        current_price *= (1 + random.uniform(-0.001, 0.001))
        
        # 슬리피지
        if side == 'buy':
            fill_price = current_price * 1.0001
        else:
            fill_price = current_price * 0.9999
        
        # 수수료
        fee = quantity * fill_price * 0.0005
        
        # 포지션 업데이트
        if symbol not in self.positions:
            self.positions[symbol] = {
                'side': 'long' if side == 'buy' else 'short',
                'size': quantity,
                'entry_price': fill_price
            }
        else:
            # 기존 포지션 조정
            pos = self.positions[symbol]
            
            if (pos['side'] == 'long' and side == 'sell') or \
               (pos['side'] == 'short' and side == 'buy'):
                # 청산
                if quantity >= pos['size']:
                    del self.positions[symbol]
                else:
                    pos['size'] -= quantity
        
        # 잔고 업데이트
        if side == 'buy':
            self.balance -= quantity * fill_price + fee
        else:
            self.balance += quantity * fill_price - fee
        
        logger.debug(f"모의 주문 체결: {symbol} {side} {quantity}@{fill_price:.2f}")
        
        return {
            'orderId': order_id,
            'status': 'FILLED',
            'executedQty': quantity,
            'cummulativeQuoteQty': quantity * fill_price
        }
    
    async def cancel_order(self, symbol: str, orderId: str) -> Dict:
        """주문 취소"""
        
        return {
            'status': 'CANCELED',
            'orderId': orderId
        }
    
    async def get_order(self, symbol: str, orderId: str) -> Dict:
        """주문 조회"""
        
        return {
            'orderId': orderId,
            'status': 'FILLED',
            'executedQty': 1.0,
            'cummulativeQuoteQty': 50000
        }