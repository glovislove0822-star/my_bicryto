"""리스크 관리"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging
from collections import deque

from ..utils.logging import Logger

logger = Logger.get_logger(__name__)

class RiskLevel(Enum):
    """리스크 레벨"""
    LOW = "low"
    NORMAL = "normal"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

class RiskManager:
    """리스크 관리자
    
    포지션, 계정, 시장 리스크 모니터링 및 관리
    """
    
    def __init__(self, config: Dict):
        """
        Args:
            config: 리스크 관리 설정
        """
        self.config = config
        
        # 리스크 한도
        self.limits = {
            'max_leverage': config.get('risk', {}).get('max_leverage', 3.0),
            'max_position_size': config.get('risk', {}).get('max_position_size', 0.3),
            'max_daily_loss': config.get('risk', {}).get('daily_stop_pct', 0.02),
            'max_daily_profit': config.get('risk', {}).get('daily_take_pct', 0.04),
            'max_drawdown': config.get('risk', {}).get('max_drawdown', 0.2),
            'max_correlation': config.get('risk', {}).get('max_correlation', 0.7),
            'var_limit': config.get('risk', {}).get('var_limit', 0.05),
            'stress_test_threshold': config.get('risk', {}).get('stress_threshold', 0.1)
        }
        
        # 리스크 메트릭
        self.metrics = {
            'current_leverage': 0,
            'var_95': 0,
            'cvar_95': 0,
            'max_drawdown': 0,
            'current_drawdown': 0,
            'correlation_risk': 0,
            'concentration_risk': 0,
            'liquidity_risk': 0,
            'funding_risk': 0
        }
        
        # 일일 통계
        self.daily_stats = {
            'pnl': 0,
            'trades': 0,
            'max_loss': 0,
            'max_profit': 0,
            'start_equity': 0,
            'high_equity': 0,
            'low_equity': 0
        }
        
        # 리스크 히스토리
        self.risk_history = deque(maxlen=1000)
        self.pnl_history = deque(maxlen=1000)
        
        # 리스크 레벨
        self.current_risk_level = RiskLevel.NORMAL
        
        # 경고/알림
        self.alerts = deque(maxlen=100)
        
        # 서킷 브레이커
        self.circuit_breaker = {
            'triggered': False,
            'trigger_time': None,
            'trigger_reason': None,
            'cooldown_minutes': 30
        }
        
        # 스트레스 테스트
        self.stress_scenarios = self._define_stress_scenarios()
    
    def check_pre_trade_risk(self,
                            symbol: str,
                            side: str,
                            size: float,
                            price: float,
                            current_positions: Dict,
                            account_balance: float) -> Tuple[bool, Optional[str]]:
        """거래 전 리스크 체크
        
        Args:
            symbol: 심볼
            side: 방향
            size: 크기
            price: 가격
            current_positions: 현재 포지션
            account_balance: 계정 잔고
            
        Returns:
            (허용 여부, 거부 사유)
        """
        
        # 서킷 브레이커 체크
        if self.circuit_breaker['triggered']:
            if not self._check_circuit_breaker_cooldown():
                return False, "circuit_breaker_active"
        
        # 포지션 크기 체크
        position_value = size * price
        position_ratio = position_value / account_balance
        
        if position_ratio > self.limits['max_position_size']:
            return False, f"position_too_large: {position_ratio:.2%}"
        
        # 레버리지 체크
        total_exposure = sum(p.get('notional', 0) for p in current_positions.values())
        new_exposure = total_exposure + position_value
        new_leverage = new_exposure / account_balance
        
        if new_leverage > self.limits['max_leverage']:
            return False, f"leverage_exceeded: {new_leverage:.1f}x"
        
        # 상관관계 리스크 체크
        if self._check_correlation_risk(symbol, current_positions) > self.limits['max_correlation']:
            return False, "correlation_risk_too_high"
        
        # VaR 체크
        estimated_var = self._estimate_var_impact(symbol, side, size, current_positions)
        
        if estimated_var > self.limits['var_limit'] * account_balance:
            return False, f"var_exceeded: {estimated_var/account_balance:.2%}"
        
        # 일일 손실 한도 체크
        if self.daily_stats['pnl'] < -self.limits['max_daily_loss'] * self.daily_stats['start_equity']:
            return False, "daily_loss_limit_reached"
        
        # 리스크 레벨 체크
        if self.current_risk_level in [RiskLevel.CRITICAL, RiskLevel.EMERGENCY]:
            return False, f"risk_level_{self.current_risk_level.value}"
        
        return True, None
    
    def update_position_risk(self,
                            positions: Dict,
                            market_prices: Dict,
                            account_balance: float):
        """포지션 리스크 업데이트
        
        Args:
            positions: 현재 포지션
            market_prices: 시장 가격
            account_balance: 계정 잔고
        """
        
        # 총 노출 계산
        total_exposure = 0
        position_pnls = []
        
        for symbol, position in positions.items():
            if symbol in market_prices:
                current_price = market_prices[symbol]
                
                # PnL 계산
                if position['side'] == 'long':
                    pnl = (current_price - position['entry_price']) * position['size']
                else:
                    pnl = (position['entry_price'] - current_price) * position['size']
                
                position_pnls.append(pnl)
                total_exposure += abs(position['size'] * current_price)
        
        # 레버리지 업데이트
        self.metrics['current_leverage'] = total_exposure / account_balance if account_balance > 0 else 0
        
        # VaR/CVaR 계산
        if position_pnls:
            self.metrics['var_95'] = np.percentile(position_pnls, 5)
            self.metrics['cvar_95'] = np.mean([p for p in position_pnls if p <= self.metrics['var_95']])
        
        # 드로우다운 업데이트
        current_equity = account_balance + sum(position_pnls)
        
        if current_equity > self.daily_stats.get('high_equity', current_equity):
            self.daily_stats['high_equity'] = current_equity
        
        drawdown = (self.daily_stats['high_equity'] - current_equity) / self.daily_stats['high_equity']
        self.metrics['current_drawdown'] = drawdown
        self.metrics['max_drawdown'] = max(self.metrics['max_drawdown'], drawdown)
        
        # 집중도 리스크
        if positions:
            position_values = [abs(p['size'] * market_prices.get(s, p['entry_price'])) 
                             for s, p in positions.items()]
            herfindahl = sum((v/sum(position_values))**2 for v in position_values)
            self.metrics['concentration_risk'] = herfindahl
        
        # 리스크 레벨 업데이트
        self._update_risk_level()
        
        # 히스토리 기록
        self.risk_history.append({
            'timestamp': datetime.now(),
            'metrics': self.metrics.copy(),
            'risk_level': self.current_risk_level
        })
    
    def _update_risk_level(self):
        """리스크 레벨 업데이트"""
        
        old_level = self.current_risk_level
        
        # 레벨 결정 로직
        if self.metrics['current_leverage'] > self.limits['max_leverage'] * 0.9:
            self.current_risk_level = RiskLevel.CRITICAL
            
        elif self.metrics['current_drawdown'] > self.limits['max_drawdown'] * 0.8:
            self.current_risk_level = RiskLevel.WARNING
            
        elif self.metrics['var_95'] < -self.limits['var_limit'] * 0.8:
            self.current_risk_level = RiskLevel.WARNING
            
        elif self.circuit_breaker['triggered']:
            self.current_risk_level = RiskLevel.EMERGENCY
            
        else:
            # 정상 범위
            if self.metrics['current_leverage'] < self.limits['max_leverage'] * 0.5:
                self.current_risk_level = RiskLevel.LOW
            else:
                self.current_risk_level = RiskLevel.NORMAL
        
        # 레벨 변경 시 알림
        if old_level != self.current_risk_level:
            self._create_alert(
                f"Risk level changed: {old_level.value} -> {self.current_risk_level.value}",
                severity='warning' if self.current_risk_level.value in ['warning', 'critical'] else 'info'
            )
    
    def check_stop_conditions(self,
                            current_equity: float) -> Optional[str]:
        """중단 조건 체크
        
        Args:
            current_equity: 현재 자산
            
        Returns:
            중단 사유 또는 None
        """
        
        # 일일 손실 한도
        daily_loss = current_equity - self.daily_stats['start_equity']
        daily_loss_pct = daily_loss / self.daily_stats['start_equity']
        
        if daily_loss_pct < -self.limits['max_daily_loss']:
            self._trigger_circuit_breaker('daily_loss_limit')
            return 'daily_loss_limit'
        
        # 일일 수익 한도
        if daily_loss_pct > self.limits['max_daily_profit']:
            logger.info(f"일일 수익 한도 도달: {daily_loss_pct:.2%}")
            return 'daily_profit_limit'
        
        # 최대 드로우다운
        if self.metrics['max_drawdown'] > self.limits['max_drawdown']:
            self._trigger_circuit_breaker('max_drawdown')
            return 'max_drawdown'
        
        # 연속 손실
        if self._check_consecutive_losses() > 5:
            self._trigger_circuit_breaker('consecutive_losses')
            return 'consecutive_losses'
        
        return None
    
    def _trigger_circuit_breaker(self, reason: str):
        """서킷 브레이커 발동
        
        Args:
            reason: 발동 사유
        """
        
        self.circuit_breaker['triggered'] = True
        self.circuit_breaker['trigger_time'] = datetime.now()
        self.circuit_breaker['trigger_reason'] = reason
        
        self._create_alert(
            f"Circuit breaker triggered: {reason}",
            severity='critical'
        )
        
        logger.critical(f"서킷 브레이커 발동: {reason}")
    
    def _check_circuit_breaker_cooldown(self) -> bool:
        """서킷 브레이커 쿨다운 체크
        
        Returns:
            쿨다운 완료 여부
        """
        
        if not self.circuit_breaker['triggered']:
            return True
        
        if not self.circuit_breaker['trigger_time']:
            return True
        
        elapsed = (datetime.now() - self.circuit_breaker['trigger_time']).total_seconds() / 60
        
        if elapsed > self.circuit_breaker['cooldown_minutes']:
            self.circuit_breaker['triggered'] = False
            self.circuit_breaker['trigger_time'] = None
            self.circuit_breaker['trigger_reason'] = None
            
            logger.info("서킷 브레이커 해제")
            return True
        
        return False
    
    def _check_correlation_risk(self,
                               symbol: str,
                               current_positions: Dict) -> float:
        """상관관계 리스크 체크
        
        Args:
            symbol: 심볼
            current_positions: 현재 포지션
            
        Returns:
            최대 상관관계
        """
        
        # TODO: 실제 상관관계 매트릭스 사용
        # 임시로 고정값 반환
        return 0.5
    
    def _estimate_var_impact(self,
                            symbol: str,
                            side: str,
                            size: float,
                            current_positions: Dict) -> float:
        """VaR 영향 추정
        
        Args:
            symbol: 심볼
            side: 방향
            size: 크기
            current_positions: 현재 포지션
            
        Returns:
            예상 VaR
        """
        
        # TODO: 실제 VaR 계산 구현
        # 임시로 간단한 추정
        return size * 0.01  # 1% 변동성 가정
    
    def _check_consecutive_losses(self) -> int:
        """연속 손실 체크
        
        Returns:
            연속 손실 수
        """
        
        if not self.pnl_history:
            return 0
        
        consecutive = 0
        for pnl in reversed(self.pnl_history):
            if pnl < 0:
                consecutive += 1
            else:
                break
        
        return consecutive
    
    def _define_stress_scenarios(self) -> List[Dict]:
        """스트레스 시나리오 정의
        
        Returns:
            스트레스 시나리오 리스트
        """
        
        return [
            {
                'name': 'flash_crash',
                'description': '순간 폭락 (-10%)',
                'price_change': -0.10,
                'volatility_multiplier': 3.0
            },
            {
                'name': 'black_swan',
                'description': '블랙 스완 (-20%)',
                'price_change': -0.20,
                'volatility_multiplier': 5.0
            },
            {
                'name': 'liquidity_crisis',
                'description': '유동성 위기',
                'spread_multiplier': 10.0,
                'depth_reduction': 0.9
            },
            {
                'name': 'correlation_breakdown',
                'description': '상관관계 붕괴',
                'correlation_shock': 1.0
            }
        ]
    
    def run_stress_test(self,
                       positions: Dict,
                       market_prices: Dict,
                       account_balance: float) -> Dict:
        """스트레스 테스트 실행
        
        Args:
            positions: 현재 포지션
            market_prices: 시장 가격
            account_balance: 계정 잔고
            
        Returns:
            스트레스 테스트 결과
        """
        
        results = {}
        
        for scenario in self.stress_scenarios:
            scenario_pnl = 0
            
            for symbol, position in positions.items():
                current_price = market_prices.get(symbol, position['entry_price'])
                
                # 시나리오 적용
                if 'price_change' in scenario:
                    stressed_price = current_price * (1 + scenario['price_change'])
                else:
                    stressed_price = current_price
                
                # PnL 계산
                if position['side'] == 'long':
                    pnl = (stressed_price - position['entry_price']) * position['size']
                else:
                    pnl = (position['entry_price'] - stressed_price) * position['size']
                
                scenario_pnl += pnl
            
            # 결과 저장
            results[scenario['name']] = {
                'description': scenario['description'],
                'pnl': scenario_pnl,
                'pnl_pct': scenario_pnl / account_balance,
                'survival': (account_balance + scenario_pnl) > 0
            }
        
        return results
    
    def update_daily_stats(self, pnl: float, equity: float):
        """일일 통계 업데이트
        
        Args:
            pnl: 손익
            equity: 자산
        """
        
        self.daily_stats['pnl'] += pnl
        self.daily_stats['trades'] += 1
        
        if pnl < 0:
            self.daily_stats['max_loss'] = min(self.daily_stats['max_loss'], pnl)
        else:
            self.daily_stats['max_profit'] = max(self.daily_stats['max_profit'], pnl)
        
        self.daily_stats['high_equity'] = max(self.daily_stats.get('high_equity', equity), equity)
        self.daily_stats['low_equity'] = min(self.daily_stats.get('low_equity', equity), equity)
        
        # PnL 히스토리
        self.pnl_history.append(pnl)
    
    def reset_daily_stats(self, starting_equity: float):
        """일일 통계 리셋
        
        Args:
            starting_equity: 시작 자산
        """
        
        self.daily_stats = {
            'pnl': 0,
            'trades': 0,
            'max_loss': 0,
            'max_profit': 0,
            'start_equity': starting_equity,
            'high_equity': starting_equity,
            'low_equity': starting_equity
        }
        
        logger.info("일일 리스크 통계 리셋")
    
    def _create_alert(self, message: str, severity: str = 'info'):
        """알림 생성
        
        Args:
            message: 알림 메시지
            severity: 심각도 (info, warning, critical)
        """
        
        alert = {
            'timestamp': datetime.now(),
            'message': message,
            'severity': severity,
            'risk_level': self.current_risk_level.value
        }
        
        self.alerts.append(alert)
        
        # 로깅
        if severity == 'critical':
            logger.critical(message)
        elif severity == 'warning':
            logger.warning(message)
        else:
            logger.info(message)
    
    def get_risk_summary(self) -> Dict:
        """리스크 요약 조회
        
        Returns:
            리스크 요약
        """
        
        return {
            'risk_level': self.current_risk_level.value,
            'metrics': self.metrics.copy(),
            'daily_stats': self.daily_stats.copy(),
            'circuit_breaker': self.circuit_breaker.copy(),
            'recent_alerts': list(self.alerts)[-10:],
            'limits': self.limits.copy()
        }