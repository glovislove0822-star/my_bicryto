"""펀딩 차익거래 전략"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass

from ..utils.logging import Logger

logger = Logger.get_logger(__name__)

@dataclass
class FundingSignal:
    """펀딩 신호"""
    symbol: str
    side: str  # 'long' or 'short'
    confidence: float
    reason: str
    expected_profit: float
    funding_rate: float
    next_funding_time: datetime
    position_size: float = 0

class FundingArbitrage:
    """펀딩 레이트 차익거래 전략
    
    극단적인 펀딩 레이트를 활용한 차익거래
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Args:
            config: 전략 설정
        """
        self.config = config or {}
        
        # 기본 파라미터
        self.params = {
            'funding_z_threshold': 2.0,  # 펀딩 Z-score 임계값
            'funding_harvest_min': 0.01,  # 최소 하베스팅 펀딩 (1%)
            'momentum_window': 20,  # 모멘텀 계산 윈도우
            'min_confidence': 0.5,  # 최소 신뢰도
            'max_position_ratio': 0.3,  # 최대 포지션 비율
            'hedge_ratio': 0.8  # 헤지 비율
        }
        
        if config:
            self.params.update(config)
        
        # 상태
        self.active_positions = {}
        self.funding_history = {}
        self.performance = []
    
    def generate_signal(self,
                       funding_rate: float,
                       funding_ma: float,
                       funding_std: float,
                       momentum: float,
                       next_funding_time: datetime,
                       symbol: str,
                       current_price: float,
                       market_regime: Optional[str] = None) -> Optional[FundingSignal]:
        """펀딩 신호 생성
        
        Args:
            funding_rate: 현재 펀딩 레이트
            funding_ma: 펀딩 이동평균
            funding_std: 펀딩 표준편차
            momentum: 가격 모멘텀
            next_funding_time: 다음 펀딩 시간
            symbol: 심볼
            current_price: 현재 가격
            market_regime: 시장 레짐
            
        Returns:
            펀딩 신호 또는 None
        """
        
        # Z-score 계산
        if funding_std > 0:
            funding_z = (funding_rate - funding_ma) / funding_std
        else:
            funding_z = 0
        
        # 극단적 펀딩 체크
        if abs(funding_z) > self.params['funding_z_threshold']:
            signal = self._extreme_funding_signal(
                funding_rate, funding_z, momentum, 
                next_funding_time, symbol, current_price
            )
            if signal:
                return signal
        
        # 펀딩 하베스팅 체크
        if abs(funding_rate) > self.params['funding_harvest_min']:
            signal = self._funding_harvesting_signal(
                funding_rate, momentum, next_funding_time, 
                symbol, current_price, market_regime
            )
            if signal:
                return signal
        
        # 펀딩 다이버전스 체크
        signal = self._funding_divergence_signal(
            funding_rate, momentum, funding_ma,
            next_funding_time, symbol, current_price
        )
        
        return signal
    
    def _extreme_funding_signal(self,
                               funding_rate: float,
                               funding_z: float,
                               momentum: float,
                               next_funding_time: datetime,
                               symbol: str,
                               current_price: float) -> Optional[FundingSignal]:
        """극단적 펀딩 신호"""
        
        confidence = min(1.0, abs(funding_z) / 3.0)
        
        # 시간 체크 (펀딩까지 남은 시간)
        time_to_funding = (next_funding_time - datetime.now()).total_seconds() / 3600
        
        if time_to_funding > 8 or time_to_funding < 0.1:
            # 너무 이르거나 너무 늦음
            return None
        
        if funding_z > self.params['funding_z_threshold'] and momentum < 0:
            # 과도한 롱 펀딩 + 음의 모멘텀 = 숏 기회
            expected_profit = abs(funding_rate) * self.params['hedge_ratio']
            
            return FundingSignal(
                symbol=symbol,
                side='short',
                confidence=confidence,
                reason='excessive_long_funding',
                expected_profit=expected_profit,
                funding_rate=funding_rate,
                next_funding_time=next_funding_time,
                position_size=self._calculate_position_size(confidence, current_price)
            )
        
        elif funding_z < -self.params['funding_z_threshold'] and momentum > 0:
            # 과도한 숏 펀딩 + 양의 모멘텀 = 롱 기회
            expected_profit = abs(funding_rate) * self.params['hedge_ratio']
            
            return FundingSignal(
                symbol=symbol,
                side='long',
                confidence=confidence,
                reason='excessive_short_funding',
                expected_profit=expected_profit,
                funding_rate=funding_rate,
                next_funding_time=next_funding_time,
                position_size=self._calculate_position_size(confidence, current_price)
            )
        
        return None
    
    def _funding_harvesting_signal(self,
                                  funding_rate: float,
                                  momentum: float,
                                  next_funding_time: datetime,
                                  symbol: str,
                                  current_price: float,
                                  market_regime: Optional[str]) -> Optional[FundingSignal]:
        """펀딩 하베스팅 신호
        
        지속적으로 높은 펀딩을 수집하는 전략
        """
        
        # 시장 레짐 체크
        if market_regime == 'extreme':
            # 극단적 변동성에서는 회피
            return None
        
        # 펀딩 방향과 모멘텀 일치 체크
        funding_momentum_aligned = (
            (funding_rate > 0 and momentum < 0) or  # 롱 펀딩, 하락 모멘텀
            (funding_rate < 0 and momentum > 0)      # 숏 펀딩, 상승 모멘텀
        )
        
        if not funding_momentum_aligned:
            return None
        
        # 신뢰도 계산
        confidence = min(1.0, abs(funding_rate) / 0.02)  # 2% 기준
        
        if confidence < self.params['min_confidence']:
            return None
        
        # 포지션 방향 결정
        if funding_rate > self.params['funding_harvest_min']:
            # 양의 펀딩 = 숏으로 펀딩 받기
            side = 'short'
            reason = 'funding_harvest_short'
        else:
            # 음의 펀딩 = 롱으로 펀딩 받기
            side = 'long'
            reason = 'funding_harvest_long'
        
        # 예상 수익
        hours_to_funding = (next_funding_time - datetime.now()).total_seconds() / 3600
        periods = max(1, int(24 / 8))  # 하루 기준
        expected_profit = abs(funding_rate) * periods * self.params['hedge_ratio']
        
        return FundingSignal(
            symbol=symbol,
            side=side,
            confidence=confidence,
            reason=reason,
            expected_profit=expected_profit,
            funding_rate=funding_rate,
            next_funding_time=next_funding_time,
            position_size=self._calculate_position_size(confidence, current_price)
        )
    
    def _funding_divergence_signal(self,
                                  funding_rate: float,
                                  momentum: float,
                                  funding_ma: float,
                                  next_funding_time: datetime,
                                  symbol: str,
                                  current_price: float) -> Optional[FundingSignal]:
        """펀딩-가격 다이버전스 신호"""
        
        # 펀딩과 가격 방향 다이버전스
        divergence = False
        
        if funding_rate > funding_ma * 1.5 and momentum < -0.01:
            # 펀딩 상승 but 가격 하락
            divergence = True
            side = 'short'
            reason = 'funding_price_divergence_short'
            
        elif funding_rate < funding_ma * 0.5 and momentum > 0.01:
            # 펀딩 하락 but 가격 상승
            divergence = True
            side = 'long'
            reason = 'funding_price_divergence_long'
        
        if not divergence:
            return None
        
        # 신뢰도
        confidence = 0.6  # 중간 신뢰도
        
        # 예상 수익
        expected_profit = abs(funding_rate - funding_ma) * self.params['hedge_ratio']
        
        return FundingSignal(
            symbol=symbol,
            side=side,
            confidence=confidence,
            reason=reason,
            expected_profit=expected_profit,
            funding_rate=funding_rate,
            next_funding_time=next_funding_time,
            position_size=self._calculate_position_size(confidence, current_price)
        )
    
    def _calculate_position_size(self, confidence: float, current_price: float) -> float:
        """포지션 크기 계산"""
        
        # 기본 크기
        base_size = self.params['max_position_ratio']
        
        # 신뢰도 기반 조정
        adjusted_size = base_size * confidence
        
        # 최소/최대 제한
        min_size = 0.05
        max_size = self.params['max_position_ratio']
        
        position_size = np.clip(adjusted_size, min_size, max_size)
        
        return position_size
    
    def manage_positions(self,
                        current_time: datetime,
                        current_prices: Dict[str, float]) -> List[Dict]:
        """포지션 관리
        
        Args:
            current_time: 현재 시간
            current_prices: 현재 가격들
            
        Returns:
            액션 리스트
        """
        
        actions = []
        
        for symbol, position in self.active_positions.items():
            if symbol not in current_prices:
                continue
            
            current_price = current_prices[symbol]
            
            # 펀딩 시간 체크
            if current_time >= position['next_funding_time']:
                # 펀딩 수령 후 포지션 재평가
                action = self._evaluate_position_after_funding(
                    position, current_price, current_time
                )
                if action:
                    actions.append(action)
            
            # 손실 제한
            pnl = self._calculate_pnl(position, current_price)
            if pnl < -0.02:  # 2% 손실
                actions.append({
                    'symbol': symbol,
                    'action': 'close',
                    'reason': 'stop_loss',
                    'pnl': pnl
                })
            
            # 수익 실현
            elif pnl > 0.03:  # 3% 수익
                actions.append({
                    'symbol': symbol,
                    'action': 'close',
                    'reason': 'take_profit',
                    'pnl': pnl
                })
        
        return actions
    
    def _evaluate_position_after_funding(self,
                                        position: Dict,
                                        current_price: float,
                                        current_time: datetime) -> Optional[Dict]:
        """펀딩 후 포지션 평가"""
        
        # 펀딩 수익 계산
        funding_income = abs(position['funding_rate']) * position['position_size']
        
        # 총 PnL
        price_pnl = self._calculate_pnl(position, current_price)
        total_pnl = price_pnl + funding_income
        
        # 다음 펀딩까지 홀딩 여부 결정
        if total_pnl > 0 and position['confidence'] > 0.7:
            # 계속 홀딩
            return None
        else:
            # 포지션 종료
            return {
                'symbol': position['symbol'],
                'action': 'close',
                'reason': 'funding_collected',
                'pnl': total_pnl,
                'funding_income': funding_income
            }
    
    def _calculate_pnl(self, position: Dict, current_price: float) -> float:
        """PnL 계산"""
        
        entry_price = position['entry_price']
        size = position['position_size']
        
        if position['side'] == 'long':
            pnl = (current_price - entry_price) / entry_price * size
        else:
            pnl = (entry_price - current_price) / entry_price * size
        
        return pnl
    
    def update_performance(self, trade_result: Dict):
        """성과 업데이트"""
        
        self.performance.append({
            'timestamp': datetime.now(),
            'symbol': trade_result['symbol'],
            'pnl': trade_result['pnl'],
            'funding_income': trade_result.get('funding_income', 0),
            'reason': trade_result.get('reason', '')
        })
    
    def get_statistics(self) -> Dict:
        """전략 통계"""
        
        if not self.performance:
            return {}
        
        df = pd.DataFrame(self.performance)
        
        stats = {
            'total_trades': len(df),
            'total_pnl': df['pnl'].sum(),
            'avg_pnl': df['pnl'].mean(),
            'win_rate': (df['pnl'] > 0).mean(),
            'total_funding_income': df['funding_income'].sum(),
            'sharpe_ratio': df['pnl'].mean() / df['pnl'].std() * np.sqrt(252) if df['pnl'].std() > 0 else 0
        }
        
        return stats