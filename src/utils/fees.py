"""수수료 및 비용 계산 유틸리티"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

@dataclass
class TradeInfo:
    """거래 정보"""
    symbol: str
    side: str  # 'buy' or 'sell'
    price: float
    quantity: float
    is_maker: bool
    
class FeeCalculator:
    """수수료 계산기"""
    
    def __init__(self, maker_bp: float = 2.0, taker_bp: float = 5.0):
        """
        Args:
            maker_bp: 메이커 수수료 (basis points)
            taker_bp: 테이커 수수료 (basis points)
        """
        self.maker_rate = maker_bp / 10000
        self.taker_rate = taker_bp / 10000
        
    def calculate_fee(self, trade: TradeInfo) -> float:
        """거래 수수료 계산"""
        notional = trade.price * trade.quantity
        fee_rate = self.maker_rate if trade.is_maker else self.taker_rate
        return notional * fee_rate
    
    def calculate_slippage(self, 
                          intended_price: float, 
                          executed_price: float,
                          quantity: float,
                          depth: float,
                          model: str = 'depth_linear',
                          k: float = 0.0001) -> float:
        """슬리피지 계산
        
        Args:
            intended_price: 의도한 가격
            executed_price: 실제 체결 가격
            quantity: 거래량
            depth: 호가 심도
            model: 슬리피지 모델 ('fixed', 'depth_linear', 'sqrt')
            k: 슬리피지 계수
        """
        if model == 'fixed':
            slippage_pct = k
        elif model == 'depth_linear':
            # 거래량/심도에 비례
            slippage_pct = k * (quantity / depth)
        elif model == 'sqrt':
            # 거래량/심도의 제곱근에 비례
            slippage_pct = k * np.sqrt(quantity / depth)
        else:
            slippage_pct = 0
            
        slippage_price = intended_price * slippage_pct
        actual_slippage = abs(executed_price - intended_price)
        
        return min(actual_slippage, slippage_price)
    
    def calculate_funding_cost(self, 
                              position_size: float,
                              funding_rate: float,
                              hours_held: float = 8) -> float:
        """펀딩 비용 계산
        
        Args:
            position_size: 포지션 크기 (USD)
            funding_rate: 펀딩 레이트 (8시간 기준)
            hours_held: 보유 시간
        """
        periods = hours_held / 8  # 8시간 단위
        return position_size * funding_rate * periods
    
    def calculate_total_cost(self,
                            entry_trade: TradeInfo,
                            exit_trade: TradeInfo,
                            funding_rate: float = 0,
                            holding_hours: float = 0,
                            slippage_model: str = 'depth_linear',
                            depth: float = 10000) -> Dict[str, float]:
        """총 거래 비용 계산"""
        
        # 진입 수수료
        entry_fee = self.calculate_fee(entry_trade)
        
        # 청산 수수료
        exit_fee = self.calculate_fee(exit_trade)
        
        # 슬리피지 (간단화: 고정 슬리피지)
        entry_slippage = self.calculate_slippage(
            entry_trade.price, 
            entry_trade.price * 1.0001,  # 가정: 0.01% 슬리피지
            entry_trade.quantity,
            depth,
            slippage_model
        )
        
        exit_slippage = self.calculate_slippage(
            exit_trade.price,
            exit_trade.price * 0.9999,  # 가정: 0.01% 슬리피지
            exit_trade.quantity,
            depth,
            slippage_model
        )
        
        # 펀딩 비용
        position_value = entry_trade.price * entry_trade.quantity
        funding_cost = self.calculate_funding_cost(position_value, funding_rate, holding_hours)
        
        # 총 비용
        total_cost = entry_fee + exit_fee + entry_slippage + exit_slippage + abs(funding_cost)
        
        return {
            'entry_fee': entry_fee,
            'exit_fee': exit_fee,
            'entry_slippage': entry_slippage,
            'exit_slippage': exit_slippage,
            'funding_cost': funding_cost,
            'total_cost': total_cost,
            'total_cost_pct': total_cost / position_value
        }
    
    def estimate_break_even_move(self,
                                 is_maker: bool = False,
                                 funding_rate: float = 0,
                                 holding_hours: float = 8) -> float:
        """손익분기점 가격 변동률 계산"""
        
        # 왕복 수수료
        fee_rate = self.maker_rate if is_maker else self.taker_rate
        round_trip_fee = fee_rate * 2
        
        # 펀딩 비용 (비율)
        funding_cost_rate = funding_rate * (holding_hours / 8)
        
        # 슬리피지 가정 (0.02%)
        slippage_rate = 0.0002
        
        # 손익분기점
        break_even = round_trip_fee + abs(funding_cost_rate) + slippage_rate
        
        return break_even