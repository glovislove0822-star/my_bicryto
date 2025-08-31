"""거래 비용 모델"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
from datetime import datetime
import logging

from ..utils.logging import Logger
from ..utils.fees import FeeCalculator, TradeInfo

logger = Logger.get_logger(__name__)

class CostModel:
    """정교한 거래 비용 모델
    
    수수료, 슬리피지, 펀딩, 시장 충격 등을 모델링
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Args:
            config: 비용 모델 설정
        """
        self.config = config or {}
        
        # 기본 파라미터
        self.params = {
            'maker_fee_bp': 2.0,  # 메이커 수수료 (basis points)
            'taker_fee_bp': 5.0,  # 테이커 수수료
            'slippage_model': 'depth_linear',  # 슬리피지 모델
            'slippage_k': 0.0001,  # 슬리피지 계수
            'market_impact_model': 'sqrt',  # 시장 충격 모델
            'impact_k': 0.001,  # 시장 충격 계수
            'funding_interval_hours': 8,  # 펀딩 간격
            'latency_ms': 50,  # 지연시간 (밀리초)
            'rejection_rate': 0.01,  # 주문 거부율
            'partial_fill_rate': 0.05  # 부분 체결률
        }
        
        if config:
            self.params.update(config)
        
        # 수수료 계산기
        self.fee_calculator = FeeCalculator(
            maker_bp=self.params['maker_fee_bp'],
            taker_bp=self.params['taker_fee_bp']
        )
        
        # 비용 통계
        self.cost_stats = {
            'total_fees': 0,
            'total_slippage': 0,
            'total_funding': 0,
            'total_impact': 0,
            'trade_count': 0
        }
    
    def calculate_trade_cost(self,
                            symbol: str,
                            side: str,
                            price: float,
                            quantity: float,
                            is_maker: bool,
                            spread: float,
                            depth: float,
                            funding_rate: float = 0,
                            holding_hours: float = 0,
                            volatility: float = 0.01) -> Dict[str, float]:
        """거래 비용 계산
        
        Args:
            symbol: 심볼
            side: 거래 방향 ('buy' or 'sell')
            price: 의도한 가격
            quantity: 거래량
            is_maker: 메이커 여부
            spread: 현재 스프레드
            depth: 호가 심도
            funding_rate: 펀딩 레이트
            holding_hours: 보유 시간
            volatility: 변동성
            
        Returns:
            비용 딕셔너리
        """
        
        costs = {}
        
        # 1. 수수료
        trade_info = TradeInfo(
            symbol=symbol,
            side=side,
            price=price,
            quantity=quantity,
            is_maker=is_maker
        )
        costs['fee'] = self.fee_calculator.calculate_fee(trade_info)
        
        # 2. 슬리피지
        costs['slippage'] = self._calculate_slippage(
            price, quantity, spread, depth, volatility
        )
        
        # 3. 시장 충격
        costs['market_impact'] = self._calculate_market_impact(
            price, quantity, depth, volatility
        )
        
        # 4. 펀딩 비용
        if holding_hours > 0:
            costs['funding'] = self._calculate_funding_cost(
                price * quantity, funding_rate, holding_hours
            )
        else:
            costs['funding'] = 0
        
        # 5. 지연 비용
        costs['latency_cost'] = self._calculate_latency_cost(
            price, volatility, self.params['latency_ms']
        )
        
        # 총 비용
        costs['total'] = sum(costs.values())
        costs['total_bps'] = costs['total'] / (price * quantity) * 10000
        
        # 통계 업데이트
        self._update_stats(costs)
        
        return costs
    
    def _calculate_slippage(self,
                          price: float,
                          quantity: float,
                          spread: float,
                          depth: float,
                          volatility: float) -> float:
        """슬리피지 계산"""
        
        model = self.params['slippage_model']
        k = self.params['slippage_k']
        
        if model == 'fixed':
            # 고정 슬리피지
            slippage_pct = k
            
        elif model == 'depth_linear':
            # 거래량/심도에 선형 비례
            if depth > 0:
                slippage_pct = k * (quantity / depth)
            else:
                slippage_pct = k * 10  # 심도 없으면 큰 슬리피지
            
        elif model == 'sqrt':
            # 거래량/심도의 제곱근에 비례
            if depth > 0:
                slippage_pct = k * np.sqrt(quantity / depth)
            else:
                slippage_pct = k * 10
            
        elif model == 'almgren':
            # Almgren-Chriss 모델 (단순화)
            if depth > 0:
                lambda_param = volatility * np.sqrt(quantity / depth)
                slippage_pct = k * lambda_param
            else:
                slippage_pct = k * volatility * 10
            
        else:
            slippage_pct = 0
        
        # 스프레드 기반 최소 슬리피지
        min_slippage = spread / 2 / price
        slippage_pct = max(slippage_pct, min_slippage)
        
        # 금액으로 변환
        slippage = price * quantity * slippage_pct
        
        return slippage
    
    def _calculate_market_impact(self,
                                price: float,
                                quantity: float,
                                depth: float,
                                volatility: float) -> float:
        """시장 충격 계산
        
        큰 주문이 시장 가격에 미치는 영향
        """
        
        model = self.params['market_impact_model']
        k = self.params['impact_k']
        
        if model == 'linear':
            # 선형 충격
            if depth > 0:
                impact_pct = k * (quantity / depth)
            else:
                impact_pct = k * 10
            
        elif model == 'sqrt':
            # Kyle 모델 (제곱근)
            if depth > 0:
                impact_pct = k * volatility * np.sqrt(quantity / depth)
            else:
                impact_pct = k * volatility * 10
            
        elif model == 'power':
            # 멱법칙 (power law)
            if depth > 0:
                alpha = 0.5  # 지수
                impact_pct = k * (quantity / depth) ** alpha
            else:
                impact_pct = k * 10
            
        else:
            impact_pct = 0
        
        # 금액으로 변환
        impact = price * quantity * impact_pct
        
        return impact
    
    def _calculate_funding_cost(self,
                               position_value: float,
                               funding_rate: float,
                               holding_hours: float) -> float:
        """펀딩 비용 계산"""
        
        # 펀딩 주기
        funding_periods = holding_hours / self.params['funding_interval_hours']
        
        # 총 펀딩 비용
        funding_cost = position_value * funding_rate * funding_periods
        
        return abs(funding_cost)  # 항상 비용으로 처리
    
    def _calculate_latency_cost(self,
                               price: float,
                               volatility: float,
                               latency_ms: float) -> float:
        """지연 비용 계산
        
        주문 전송 지연으로 인한 불리한 가격 변동
        """
        
        # 지연 시간 (초)
        latency_sec = latency_ms / 1000
        
        # 예상 가격 변동 (브라운 운동 가정)
        expected_move = volatility * np.sqrt(latency_sec / (24 * 3600))
        
        # 비용 (50% 확률로 불리한 방향)
        latency_cost = price * expected_move * 0.5
        
        return latency_cost
    
    def calculate_portfolio_cost(self,
                                trades: pd.DataFrame,
                                market_data: pd.DataFrame) -> pd.DataFrame:
        """포트폴리오 레벨 비용 계산
        
        Args:
            trades: 거래 데이터프레임
            market_data: 시장 데이터
            
        Returns:
            비용이 추가된 거래 데이터프레임
        """
        
        # 비용 컬럼 초기화
        cost_columns = ['fee', 'slippage', 'market_impact', 'funding', 'latency_cost', 'total_cost']
        for col in cost_columns:
            trades[col] = 0.0
        
        # 각 거래에 대해 비용 계산
        for idx, trade in trades.iterrows():
            # 시장 데이터 매칭
            market_snapshot = self._get_market_snapshot(
                market_data, trade['symbol'], trade['timestamp']
            )
            
            # 비용 계산
            costs = self.calculate_trade_cost(
                symbol=trade['symbol'],
                side=trade['side'],
                price=trade['price'],
                quantity=trade['quantity'],
                is_maker=trade.get('is_maker', False),
                spread=market_snapshot.get('spread', 0.0001),
                depth=market_snapshot.get('depth', 10000),
                funding_rate=market_snapshot.get('funding_rate', 0),
                holding_hours=trade.get('holding_hours', 0),
                volatility=market_snapshot.get('volatility', 0.01)
            )
            
            # 비용 할당
            for cost_type, cost_value in costs.items():
                if cost_type in cost_columns:
                    trades.loc[idx, cost_type] = cost_value
        
        # 순 PnL 계산
        trades['gross_pnl'] = trades['pnl']
        trades['net_pnl'] = trades['gross_pnl'] - trades['total_cost']
        
        return trades
    
    def _get_market_snapshot(self,
                           market_data: pd.DataFrame,
                           symbol: str,
                           timestamp: datetime) -> Dict:
        """특정 시점의 시장 데이터 스냅샷"""
        
        # 심볼과 시간으로 필터링
        mask = (market_data['symbol'] == symbol) & \
               (market_data.index <= timestamp)
        
        if mask.sum() == 0:
            # 기본값 반환
            return {
                'spread': 0.0001,
                'depth': 10000,
                'funding_rate': 0,
                'volatility': 0.01
            }
        
        # 가장 최근 데이터
        snapshot = market_data[mask].iloc[-1]
        
        return {
            'spread': snapshot.get('spread_bps', 1) / 10000,
            'depth': snapshot.get('depth_total', 10000),
            'funding_rate': snapshot.get('funding_rate', 0),
            'volatility': snapshot.get('realized_vol', 0.01)
        }
    
    def _update_stats(self, costs: Dict[str, float]):
        """비용 통계 업데이트"""
        
        self.cost_stats['total_fees'] += costs.get('fee', 0)
        self.cost_stats['total_slippage'] += costs.get('slippage', 0)
        self.cost_stats['total_funding'] += costs.get('funding', 0)
        self.cost_stats['total_impact'] += costs.get('market_impact', 0)
        self.cost_stats['trade_count'] += 1
    
    def get_cost_summary(self) -> Dict:
        """비용 요약 통계"""
        
        if self.cost_stats['trade_count'] == 0:
            return {}
        
        total_cost = (
            self.cost_stats['total_fees'] +
            self.cost_stats['total_slippage'] +
            self.cost_stats['total_funding'] +
            self.cost_stats['total_impact']
        )
        
        return {
            'total_cost': total_cost,
            'avg_cost_per_trade': total_cost / self.cost_stats['trade_count'],
            'fee_pct': self.cost_stats['total_fees'] / total_cost * 100 if total_cost > 0 else 0,
            'slippage_pct': self.cost_stats['total_slippage'] / total_cost * 100 if total_cost > 0 else 0,
            'funding_pct': self.cost_stats['total_funding'] / total_cost * 100 if total_cost > 0 else 0,
            'impact_pct': self.cost_stats['total_impact'] / total_cost * 100 if total_cost > 0 else 0,
            'trade_count': self.cost_stats['trade_count']
        }
    
    def optimize_execution(self,
                         order_size: float,
                         available_depth: float,
                         urgency: float = 0.5) -> Dict:
        """주문 실행 최적화
        
        Args:
            order_size: 주문 크기
            available_depth: 가용 심도
            urgency: 긴급도 (0-1)
            
        Returns:
            최적 실행 전략
        """
        
        # VWAP vs TWAP 결정
        if order_size / available_depth > 0.1:
            # 큰 주문: 분할 실행
            execution_strategy = 'TWAP'
            
            # 분할 수 계산
            n_slices = max(2, int(order_size / available_depth * 10))
            
            # 시간 간격
            if urgency > 0.7:
                interval_seconds = 10
            elif urgency > 0.3:
                interval_seconds = 30
            else:
                interval_seconds = 60
            
            return {
                'strategy': execution_strategy,
                'n_slices': n_slices,
                'slice_size': order_size / n_slices,
                'interval_seconds': interval_seconds,
                'expected_cost_reduction': 0.3  # 30% 비용 절감 예상
            }
        
        else:
            # 작은 주문: 즉시 실행
            return {
                'strategy': 'IMMEDIATE',
                'n_slices': 1,
                'slice_size': order_size,
                'interval_seconds': 0,
                'expected_cost_reduction': 0
            }