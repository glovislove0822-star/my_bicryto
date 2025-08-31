"""동적 포트폴리오 리밸런싱 모듈"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging
from scipy.optimize import minimize
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')

from ..utils.logging import Logger
from ..utils.math import MathUtils

logger = Logger.get_logger(__name__)

class DynamicRebalancer:
    """동적 포트폴리오 리밸런서
    
    켈리 기준과 상관관계를 고려한 최적 자산 배분
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Args:
            config: 리밸런서 설정
        """
        self.config = config or {}
        self.math_utils = MathUtils()
        
        # 기본 파라미터
        self.params = {
            'kelly_fraction': 0.25,  # 보수적 켈리 (1/4)
            'max_position': 0.30,  # 최대 단일 포지션
            'min_position': 0.05,  # 최소 단일 포지션
            'max_correlation': 0.7,  # 최대 허용 상관관계
            'rebalance_threshold': 0.1,  # 리밸런싱 임계값 (10% 이탈)
            'rebalance_hours': 4,  # 리밸런싱 주기
            'risk_parity': False,  # 리스크 패리티 사용
            'use_black_litterman': False,  # Black-Litterman 모델
            'confidence_weighting': True,  # 신뢰도 가중
            'regime_adaptive': True  # 레짐 적응형
        }
        
        if config:
            self.params.update(config)
        
        # 포트폴리오 상태
        self.current_allocations = {}
        self.target_allocations = {}
        self.performance_history = {}
        
        # 상관관계 매트릭스
        self.correlation_matrix = None
        self.covariance_matrix = None
        
        # 리밸런싱 히스토리
        self.rebalance_history = []
        self.last_rebalance = None
    
    def rebalance(self,
                 symbols_performance: Dict[str, Dict],
                 correlations: pd.DataFrame,
                 total_capital: float,
                 current_positions: Optional[Dict] = None,
                 market_regime: Optional[str] = None,
                 constraints: Optional[Dict] = None) -> Dict[str, float]:
        """포트폴리오 리밸런싱
        
        Args:
            symbols_performance: 심볼별 성과 정보
            correlations: 상관관계 매트릭스
            total_capital: 총 자본
            current_positions: 현재 포지션
            market_regime: 시장 레짐
            constraints: 제약 조건
            
        Returns:
            심볼별 목표 배분 (비율)
        """
        
        if not symbols_performance:
            return {}
        
        # 현재 포지션
        if current_positions:
            self.current_allocations = current_positions
        
        # 상관관계 업데이트
        self.correlation_matrix = correlations
        
        # 1. 켈리 엣지 계산
        kelly_allocations = self._calculate_kelly_allocations(symbols_performance)
        
        # 2. 상관관계 조정
        corr_adjusted = self._adjust_for_correlations(kelly_allocations, correlations)
        
        # 3. 리스크 패리티 (선택적)
        if self.params['risk_parity']:
            risk_parity_alloc = self._calculate_risk_parity(
                symbols_performance, correlations
            )
            # 켈리와 리스크 패리티 블렌딩
            allocations = self._blend_allocations(
                corr_adjusted, risk_parity_alloc, weight=0.5
            )
        else:
            allocations = corr_adjusted
        
        # 4. Black-Litterman (선택적)
        if self.params['use_black_litterman']:
            bl_allocations = self._black_litterman_optimization(
                symbols_performance, correlations, allocations
            )
            allocations = bl_allocations
        
        # 5. 레짐 조정
        if self.params['regime_adaptive'] and market_regime:
            allocations = self._adjust_for_regime(allocations, market_regime)
        
        # 6. 제약 조건 적용
        if constraints:
            allocations = self._apply_constraints(allocations, constraints)
        
        # 7. 정규화
        allocations = self._normalize_allocations(allocations)
        
        # 8. 리밸런싱 필요성 체크
        if self._should_rebalance(allocations):
            self.target_allocations = allocations
            self._record_rebalance(allocations, symbols_performance)
            return allocations
        else:
            # 현재 유지
            return self.current_allocations
    
    def _calculate_kelly_allocations(self, 
                                    symbols_performance: Dict[str, Dict]) -> Dict[str, float]:
        """켈리 기준 배분 계산"""
        
        allocations = {}
        
        for symbol, perf in symbols_performance.items():
            # 켈리 엣지 계산
            win_rate = perf.get('win_rate', 0.5)
            avg_win = perf.get('avg_win', 0)
            avg_loss = abs(perf.get('avg_loss', 1))
            
            if avg_loss == 0:
                edge = 0
            else:
                # Kelly = (p*b - q) / b
                # p = win_rate, q = 1-win_rate, b = avg_win/avg_loss
                b = avg_win / avg_loss if avg_loss > 0 else 0
                q = 1 - win_rate
                
                if b > 0:
                    kelly = (win_rate * b - q) / b
                else:
                    kelly = 0
            
            # 보수적 켈리
            conservative_kelly = kelly * self.params['kelly_fraction']
            
            # 신뢰도 가중 (선택적)
            if self.params['confidence_weighting']:
                confidence = perf.get('confidence', 1.0)
                conservative_kelly *= confidence
            
            # 범위 제한
            allocation = np.clip(
                conservative_kelly,
                0,  # 음수 배분 없음
                self.params['max_position']
            )
            
            allocations[symbol] = allocation
        
        return allocations
    
    def _adjust_for_correlations(self,
                                allocations: Dict[str, float],
                                correlations: pd.DataFrame) -> Dict[str, float]:
        """상관관계 조정"""
        
        adjusted = {}
        symbols = list(allocations.keys())
        
        for symbol in symbols:
            base_allocation = allocations[symbol]
            
            # 다른 심볼과의 상관관계 체크
            correlation_penalty = 0
            
            for other_symbol in symbols:
                if other_symbol == symbol:
                    continue
                
                if symbol in correlations.index and other_symbol in correlations.columns:
                    corr = correlations.loc[symbol, other_symbol]
                    
                    # 높은 상관관계 페널티
                    if abs(corr) > self.params['max_correlation']:
                        other_alloc = allocations.get(other_symbol, 0)
                        
                        # 상관관계가 높을수록 페널티 증가
                        penalty_factor = (abs(corr) - self.params['max_correlation']) / \
                                       (1 - self.params['max_correlation'])
                        
                        correlation_penalty += penalty_factor * other_alloc * 0.3
            
            # 조정된 배분
            adjusted_allocation = base_allocation * (1 - correlation_penalty)
            
            # 최소값 보장
            if adjusted_allocation > 0:
                adjusted_allocation = max(
                    adjusted_allocation,
                    self.params['min_position']
                )
            
            adjusted[symbol] = adjusted_allocation
        
        return adjusted
    
    def _calculate_risk_parity(self,
                              symbols_performance: Dict[str, Dict],
                              correlations: pd.DataFrame) -> Dict[str, float]:
        """리스크 패리티 배분"""
        
        symbols = list(symbols_performance.keys())
        n = len(symbols)
        
        if n == 0:
            return {}
        
        # 변동성 벡터
        volatilities = np.array([
            symbols_performance[s].get('volatility', 0.01) 
            for s in symbols
        ])
        
        # 공분산 매트릭스 구성
        cov_matrix = np.outer(volatilities, volatilities)
        
        for i, sym1 in enumerate(symbols):
            for j, sym2 in enumerate(symbols):
                if i != j and sym1 in correlations.index and sym2 in correlations.columns:
                    cov_matrix[i, j] *= correlations.loc[sym1, sym2]
        
        # 리스크 패리티 최적화
        def risk_parity_objective(weights):
            """리스크 기여도 균등화 목적 함수"""
            portfolio_vol = np.sqrt(weights @ cov_matrix @ weights)
            
            # 각 자산의 리스크 기여도
            marginal_contrib = cov_matrix @ weights
            contrib = weights * marginal_contrib / portfolio_vol
            
            # 리스크 기여도의 분산 (균등할수록 0에 가까움)
            target_contrib = portfolio_vol / n
            return np.sum((contrib - target_contrib) ** 2)
        
        # 초기 추정 (균등 가중)
        initial_weights = np.ones(n) / n
        
        # 제약 조건
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},  # 합 = 1
        ]
        
        bounds = [(0, self.params['max_position']) for _ in range(n)]
        
        # 최적화
        result = minimize(
            risk_parity_objective,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'disp': False}
        )
        
        if result.success:
            allocations = {
                symbols[i]: result.x[i] 
                for i in range(n)
            }
        else:
            # 실패 시 균등 배분
            allocations = {s: 1/n for s in symbols}
        
        return allocations
    
    def _black_litterman_optimization(self,
                                     symbols_performance: Dict[str, Dict],
                                     correlations: pd.DataFrame,
                                     prior_allocations: Dict[str, float]) -> Dict[str, float]:
        """Black-Litterman 모델 최적화"""
        
        symbols = list(symbols_performance.keys())
        n = len(symbols)
        
        if n == 0:
            return {}
        
        # 시장 균형 수익률 (CAPM 가정)
        market_returns = np.array([
            symbols_performance[s].get('expected_return', 0.1)
            for s in symbols
        ])
        
        # 투자자 관점 (Views)
        P = np.eye(n)  # 각 자산에 대한 직접적 관점
        Q = np.array([
            symbols_performance[s].get('expected_return', 0.1) * 1.2  # 20% 상향
            if symbols_performance[s].get('confidence', 0.5) > 0.7
            else symbols_performance[s].get('expected_return', 0.1)
            for s in symbols
        ])
        
        # 불확실성
        tau = 0.05  # 스케일링 파라미터
        omega = np.diag([
            (symbols_performance[s].get('volatility', 0.01) ** 2) / 
            symbols_performance[s].get('confidence', 0.5)
            for s in symbols
        ])
        
        # 공분산 매트릭스
        volatilities = np.array([
            symbols_performance[s].get('volatility', 0.01)
            for s in symbols
        ])
        
        sigma = np.outer(volatilities, volatilities)
        for i, sym1 in enumerate(symbols):
            for j, sym2 in enumerate(symbols):
                if i != j and sym1 in correlations.index and sym2 in correlations.columns:
                    sigma[i, j] *= correlations.loc[sym1, sym2]
        
        # Black-Litterman 수익률
        inv_sigma = np.linalg.inv(sigma)
        inv_omega = np.linalg.inv(omega)
        
        posterior_precision = inv_sigma + tau * P.T @ inv_omega @ P
        posterior_mean = np.linalg.inv(posterior_precision) @ (
            inv_sigma @ market_returns + tau * P.T @ inv_omega @ Q
        )
        
        # 최적 포트폴리오
        risk_aversion = 2.5  # 위험 회피 계수
        optimal_weights = np.linalg.inv(risk_aversion * sigma) @ posterior_mean
        
        # 정규화 및 제약
        optimal_weights = np.maximum(optimal_weights, 0)  # Long only
        optimal_weights = optimal_weights / optimal_weights.sum()
        
        allocations = {
            symbols[i]: min(optimal_weights[i], self.params['max_position'])
            for i in range(n)
        }
        
        return allocations
    
    def _adjust_for_regime(self, 
                          allocations: Dict[str, float],
                          market_regime: str) -> Dict[str, float]:
        """시장 레짐에 따른 조정"""
        
        adjusted = {}
        
        # 레짐별 조정 계수
        regime_multipliers = {
            'bull': 1.2,  # 상승장: 공격적
            'bear': 0.7,  # 하락장: 보수적
            'high_vol': 0.6,  # 고변동성: 축소
            'low_vol': 1.1,  # 저변동성: 확대
            'trending': 1.15,  # 트렌드: 추종
            'ranging': 0.85  # 횡보: 축소
        }
        
        multiplier = regime_multipliers.get(market_regime, 1.0)
        
        for symbol, allocation in allocations.items():
            # 레짐 조정
            adjusted_alloc = allocation * multiplier
            
            # 범위 제한
            adjusted_alloc = np.clip(
                adjusted_alloc,
                self.params['min_position'],
                self.params['max_position']
            )
            
            adjusted[symbol] = adjusted_alloc
        
        return adjusted
    
    def _blend_allocations(self,
                         alloc1: Dict[str, float],
                         alloc2: Dict[str, float],
                         weight: float = 0.5) -> Dict[str, float]:
        """두 배분 전략 블렌딩"""
        
        blended = {}
        all_symbols = set(alloc1.keys()) | set(alloc2.keys())
        
        for symbol in all_symbols:
            w1 = alloc1.get(symbol, 0)
            w2 = alloc2.get(symbol, 0)
            
            blended[symbol] = weight * w1 + (1 - weight) * w2
        
        return blended
    
    def _apply_constraints(self,
                          allocations: Dict[str, float],
                          constraints: Dict) -> Dict[str, float]:
        """제약 조건 적용"""
        
        constrained = allocations.copy()
        
        # 섹터별 제한
        if 'sector_limits' in constraints:
            # TODO: 섹터별 제한 구현
            pass
        
        # 개별 심볼 제한
        if 'symbol_limits' in constraints:
            for symbol, limits in constraints['symbol_limits'].items():
                if symbol in constrained:
                    min_alloc = limits.get('min', 0)
                    max_alloc = limits.get('max', 1)
                    
                    constrained[symbol] = np.clip(
                        constrained[symbol],
                        min_alloc,
                        max_alloc
                    )
        
        # 총 포지션 수 제한
        if 'max_positions' in constraints:
            max_positions = constraints['max_positions']
            
            if len(constrained) > max_positions:
                # 상위 N개만 선택
                sorted_symbols = sorted(
                    constrained.items(),
                    key=lambda x: x[1],
                    reverse=True
                )
                
                constrained = dict(sorted_symbols[:max_positions])
        
        return constrained
    
    def _normalize_allocations(self, allocations: Dict[str, float]) -> Dict[str, float]:
        """배분 정규화"""
        
        total = sum(allocations.values())
        
        if total == 0:
            # 균등 배분
            n = len(allocations)
            return {s: 1/n for s in allocations.keys()}
        
        # 비율 조정
        normalized = {
            s: w/total 
            for s, w in allocations.items()
        }
        
        return normalized
    
    def _should_rebalance(self, new_allocations: Dict[str, float]) -> bool:
        """리밸런싱 필요성 판단"""
        
        # 시간 체크
        if self.last_rebalance:
            elapsed = (datetime.now() - self.last_rebalance).total_seconds() / 3600
            
            if elapsed < self.params['rebalance_hours']:
                return False
        
        # 이탈도 체크
        if not self.current_allocations:
            return True
        
        max_deviation = 0
        
        for symbol in new_allocations:
            current = self.current_allocations.get(symbol, 0)
            target = new_allocations[symbol]
            
            deviation = abs(target - current)
            max_deviation = max(max_deviation, deviation)
        
        # 임계값 초과 시 리밸런싱
        return max_deviation > self.params['rebalance_threshold']
    
    def _record_rebalance(self, 
                         allocations: Dict[str, float],
                         performance: Dict[str, Dict]):
        """리밸런싱 기록"""
        
        record = {
            'timestamp': datetime.now(),
            'allocations': allocations.copy(),
            'performance_snapshot': {
                s: {
                    'win_rate': p.get('win_rate', 0),
                    'sharpe': p.get('sharpe_ratio', 0),
                    'volatility': p.get('volatility', 0)
                }
                for s, p in performance.items()
            }
        }
        
        self.rebalance_history.append(record)
        self.last_rebalance = datetime.now()
        
        logger.info(f"포트폴리오 리밸런싱 완료: {len(allocations)} 자산")
        for symbol, weight in allocations.items():
            logger.info(f"  {symbol}: {weight:.1%}")
    
    def get_portfolio_metrics(self, 
                             allocations: Dict[str, float],
                             symbols_performance: Dict[str, Dict],
                             correlations: pd.DataFrame) -> Dict:
        """포트폴리오 메트릭 계산"""
        
        if not allocations or not symbols_performance:
            return {}
        
        # 가중 평균 수익률
        portfolio_return = sum(
            allocations.get(s, 0) * perf.get('expected_return', 0)
            for s, perf in symbols_performance.items()
        )
        
        # 포트폴리오 변동성
        symbols = list(allocations.keys())
        n = len(symbols)
        
        if n > 0:
            weights = np.array([allocations.get(s, 0) for s in symbols])
            volatilities = np.array([
                symbols_performance.get(s, {}).get('volatility', 0.01)
                for s in symbols
            ])
            
            # 공분산 매트릭스
            cov_matrix = np.outer(volatilities, volatilities)
            
            for i, sym1 in enumerate(symbols):
                for j, sym2 in enumerate(symbols):
                    if i != j and sym1 in correlations.index and sym2 in correlations.columns:
                        cov_matrix[i, j] *= correlations.loc[sym1, sym2]
            
            portfolio_vol = np.sqrt(weights @ cov_matrix @ weights)
        else:
            portfolio_vol = 0
        
        # Sharpe Ratio
        if portfolio_vol > 0:
            portfolio_sharpe = portfolio_return / portfolio_vol
        else:
            portfolio_sharpe = 0
        
        # 다각화 비율
        if n > 1:
            weighted_vol_sum = sum(
                allocations.get(s, 0) * symbols_performance.get(s, {}).get('volatility', 0.01)
                for s in symbols
            )
            
            if weighted_vol_sum > 0:
                diversification_ratio = weighted_vol_sum / portfolio_vol
            else:
                diversification_ratio = 1
        else:
            diversification_ratio = 1
        
        # 집중도 (HHI)
        herfindahl = sum(w**2 for w in allocations.values())
        
        # 유효 자산 수
        if herfindahl > 0:
            effective_n = 1 / herfindahl
        else:
            effective_n = 0
        
        return {
            'expected_return': portfolio_return,
            'volatility': portfolio_vol,
            'sharpe_ratio': portfolio_sharpe,
            'diversification_ratio': diversification_ratio,
            'concentration_hhi': herfindahl,
            'effective_n_assets': effective_n,
            'n_assets': n,
            'max_weight': max(allocations.values()) if allocations else 0,
            'min_weight': min(allocations.values()) if allocations else 0
        }