"""피라미드 진입 최적화 모듈"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import logging
from dataclasses import dataclass
from collections import deque

from ..utils.logging import Logger
from ..utils.math import MathUtils

logger = Logger.get_logger(__name__)

@dataclass
class EntryLevel:
    """진입 레벨"""
    price: float
    size: float
    size_pct: float
    trigger_condition: str
    urgency: float
    expected_slippage: float

@dataclass
class PyramidPlan:
    """피라미드 진입 계획"""
    symbol: str
    side: str
    total_size: float
    entry_levels: List[EntryLevel]
    max_levels: int
    time_interval: int  # seconds
    adaptive: bool
    liquidity_based: bool

class PyramidEntryOptimizer:
    """피라미드 진입 최적화
    
    유동성과 시장 상황에 따른 다층 진입 전략
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Args:
            config: 설정
        """
        self.config = config or {}
        self.math_utils = MathUtils()
        
        # 기본 파라미터
        self.params = {
            'max_splits': 3,
            'default_split_ratio': [0.5, 0.3, 0.2],
            'liquidity_based': True,
            'adaptive_sizing': True,
            'min_interval_seconds': 5,
            'max_interval_seconds': 60,
            'urgency_threshold': 0.7,
            'vwap_bands': [0.5, 1.0, 1.5],  # ATR 배수
            'depth_threshold': 0.1,  # 심도 대비 주문 크기 비율
            'impact_model': 'sqrt'
        }
        
        if config:
            self.params.update(config)
        
        # 실행 상태
        self.active_pyramids = {}
        self.execution_history = []
        
        # 유동성 프로파일
        self.liquidity_profile = {}
        
        # 시장 충격 추정기
        self.impact_estimator = MarketImpactEstimator()
    
    def optimize_entries(self,
                        signal_strength: float,
                        market_depth: Dict,
                        vwap: float,
                        atr: float,
                        current_price: float,
                        total_size: float,
                        side: str,
                        urgency: float = 0.5,
                        volatility: float = 0.01) -> PyramidPlan:
        """진입 레벨 최적화
        
        Args:
            signal_strength: 신호 강도 (0-1)
            market_depth: 호가 심도 정보
            vwap: VWAP
            atr: ATR
            current_price: 현재 가격
            total_size: 총 주문 크기
            side: 'buy' or 'sell'
            urgency: 긴급도 (0-1)
            volatility: 변동성
            
        Returns:
            피라미드 진입 계획
        """
        
        # 분할 수 결정
        n_splits = self._determine_split_count(
            signal_strength, market_depth, total_size, urgency
        )
        
        # 유동성 포켓 찾기
        liquidity_pockets = self._find_liquidity_pockets(market_depth, side)
        
        # 진입 레벨 생성
        if self.params['liquidity_based'] and liquidity_pockets:
            entry_levels = self._liquidity_based_entries(
                liquidity_pockets, total_size, n_splits, 
                current_price, side, vwap, atr
            )
        else:
            entry_levels = self._vwap_based_entries(
                n_splits, total_size, current_price, 
                side, vwap, atr, volatility
            )
        
        # 시장 충격 최적화
        if self.params['adaptive_sizing']:
            entry_levels = self._optimize_for_impact(
                entry_levels, market_depth, volatility
            )
        
        # 시간 간격 계산
        time_interval = self._calculate_time_interval(urgency, volatility)
        
        # 피라미드 계획 생성
        plan = PyramidPlan(
            symbol='',  # 심볼은 나중에 설정
            side=side,
            total_size=total_size,
            entry_levels=entry_levels,
            max_levels=n_splits,
            time_interval=time_interval,
            adaptive=self.params['adaptive_sizing'],
            liquidity_based=self.params['liquidity_based']
        )
        
        logger.debug(f"피라미드 계획 생성: {n_splits} 레벨, "
                    f"시간 간격: {time_interval}초")
        
        return plan
    
    def _determine_split_count(self,
                              signal_strength: float,
                              market_depth: Dict,
                              total_size: float,
                              urgency: float) -> int:
        """분할 수 결정"""
        
        # 기본 분할 수
        base_splits = 1
        
        # 신호 강도 기반
        if signal_strength > 0.8:
            base_splits = 3
        elif signal_strength > 0.6:
            base_splits = 2
        
        # 시장 심도 기반
        total_depth = market_depth.get('total_depth', 0)
        if total_depth > 0:
            size_depth_ratio = total_size / total_depth
            
            if size_depth_ratio > self.params['depth_threshold']:
                # 큰 주문은 더 많이 분할
                base_splits = min(base_splits + 1, self.params['max_splits'])
        
        # 긴급도 조정
        if urgency > self.params['urgency_threshold']:
            # 긴급하면 분할 축소
            base_splits = max(1, base_splits - 1)
        
        return min(base_splits, self.params['max_splits'])
    
    def _find_liquidity_pockets(self, 
                               market_depth: Dict,
                               side: str) -> List[Dict]:
        """유동성 포켓 찾기
        
        호가창에서 유동성이 집중된 가격 레벨 탐색
        """
        
        liquidity_pockets = []
        
        # 호가 레벨 분석
        if side == 'buy':
            levels = market_depth.get('ask_levels', [])
        else:
            levels = market_depth.get('bid_levels', [])
        
        if not levels:
            return []
        
        # 유동성 집중도 계산
        total_liquidity = sum(level.get('size', 0) for level in levels)
        
        if total_liquidity == 0:
            return []
        
        # 포켓 식별
        for i, level in enumerate(levels):
            size = level.get('size', 0)
            price = level.get('price', 0)
            
            # 상대적 유동성
            liquidity_ratio = size / total_liquidity
            
            # 임계값 이상이면 포켓으로 간주
            if liquidity_ratio > 0.15:  # 15% 이상
                # 인접 레벨과의 관계 분석
                is_peak = True
                
                if i > 0:
                    prev_size = levels[i-1].get('size', 0)
                    if prev_size > size:
                        is_peak = False
                
                if i < len(levels) - 1:
                    next_size = levels[i+1].get('size', 0)
                    if next_size > size:
                        is_peak = False
                
                liquidity_pockets.append({
                    'price': price,
                    'size': size,
                    'ratio': liquidity_ratio,
                    'level': i,
                    'is_peak': is_peak,
                    'score': liquidity_ratio * (1.5 if is_peak else 1.0)
                })
        
        # 점수순 정렬
        liquidity_pockets.sort(key=lambda x: x['score'], reverse=True)
        
        return liquidity_pockets
    
    def _liquidity_based_entries(self,
                                liquidity_pockets: List[Dict],
                                total_size: float,
                                n_splits: int,
                                current_price: float,
                                side: str,
                                vwap: float,
                                atr: float) -> List[EntryLevel]:
        """유동성 기반 진입 레벨"""
        
        entry_levels = []
        remaining_size = total_size
        
        # 상위 포켓 선택
        selected_pockets = liquidity_pockets[:n_splits]
        
        if not selected_pockets:
            # 폴백: VWAP 기반
            return self._vwap_based_entries(
                n_splits, total_size, current_price,
                side, vwap, atr, 0.01
            )
        
        # 각 포켓에 크기 할당
        total_score = sum(p['score'] for p in selected_pockets)
        
        for i, pocket in enumerate(selected_pockets):
            # 크기 비례 할당
            if total_score > 0:
                size_ratio = pocket['score'] / total_score
            else:
                size_ratio = 1.0 / len(selected_pockets)
            
            size = total_size * size_ratio
            
            # 진입 가격 조정
            if side == 'buy':
                # 포켓 직전 가격
                entry_price = pocket['price'] - 0.0001
            else:
                # 포켓 직후 가격
                entry_price = pocket['price'] + 0.0001
            
            # 예상 슬리피지
            expected_slippage = self._estimate_slippage(
                size, pocket['size'], 0.01
            )
            
            entry_levels.append(EntryLevel(
                price=entry_price,
                size=size,
                size_pct=size_ratio,
                trigger_condition=f'liquidity_pocket_{i}',
                urgency=0.5 + i * 0.1,  # 순차적 긴급도
                expected_slippage=expected_slippage
            ))
            
            remaining_size -= size
        
        # 남은 크기 처리
        if remaining_size > 0 and entry_levels:
            entry_levels[-1].size += remaining_size
        
        return entry_levels
    
    def _vwap_based_entries(self,
                          n_splits: int,
                          total_size: float,
                          current_price: float,
                          side: str,
                          vwap: float,
                          atr: float,
                          volatility: float) -> List[EntryLevel]:
        """VWAP 기반 진입 레벨"""
        
        entry_levels = []
        
        # 분할 비율
        if n_splits <= len(self.params['default_split_ratio']):
            split_ratios = self.params['default_split_ratio'][:n_splits]
        else:
            # 균등 분할
            split_ratios = [1.0 / n_splits] * n_splits
        
        # 정규화
        total_ratio = sum(split_ratios)
        split_ratios = [r / total_ratio for r in split_ratios]
        
        # VWAP 밴드
        vwap_bands = self.params['vwap_bands'][:n_splits]
        
        for i in range(n_splits):
            size = total_size * split_ratios[i]
            
            # VWAP 기준 가격
            if side == 'buy':
                # 과매도 영역에서 매수
                entry_price = vwap - vwap_bands[i] * atr
                
                # 현재가보다 높으면 조정
                if entry_price > current_price:
                    entry_price = current_price - i * atr * 0.1
                    
            else:
                # 과매수 영역에서 매도
                entry_price = vwap + vwap_bands[i] * atr
                
                # 현재가보다 낮으면 조정
                if entry_price < current_price:
                    entry_price = current_price + i * atr * 0.1
            
            # 예상 슬리피지
            expected_slippage = volatility * np.sqrt(size / total_size) * 0.01
            
            # 트리거 조건
            if i == 0:
                trigger = 'immediate'
            else:
                trigger = f'vwap_band_{i}'
            
            entry_levels.append(EntryLevel(
                price=entry_price,
                size=size,
                size_pct=split_ratios[i],
                trigger_condition=trigger,
                urgency=1.0 - i * 0.2,
                expected_slippage=expected_slippage
            ))
        
        return entry_levels
    
    def _optimize_for_impact(self,
                           entry_levels: List[EntryLevel],
                           market_depth: Dict,
                           volatility: float) -> List[EntryLevel]:
        """시장 충격 최적화"""
        
        optimized_levels = []
        
        total_depth = market_depth.get('total_depth', 10000)
        
        for level in entry_levels:
            # 예상 시장 충격
            impact = self.impact_estimator.estimate(
                level.size,
                total_depth,
                volatility,
                self.params['impact_model']
            )
            
            # 충격이 크면 크기 축소
            if impact > 0.002:  # 0.2% 이상
                reduction_factor = 0.002 / impact
                level.size *= reduction_factor
                level.size_pct *= reduction_factor
                
                logger.debug(f"시장 충격 최적화: 크기 {reduction_factor:.1%} 축소")
            
            optimized_levels.append(level)
        
        # 크기 재정규화
        total_optimized = sum(l.size for l in optimized_levels)
        if total_optimized > 0:
            for level in optimized_levels:
                level.size_pct = level.size / total_optimized
        
        return optimized_levels
    
    def _calculate_time_interval(self, urgency: float, volatility: float) -> int:
        """시간 간격 계산"""
        
        # 기본 간격
        base_interval = self.params['max_interval_seconds']
        
        # 긴급도 조정
        urgency_factor = 1.0 - urgency
        
        # 변동성 조정 (변동성 높으면 간격 축소)
        vol_factor = 1.0 - min(volatility * 10, 0.5)
        
        # 최종 간격
        interval = base_interval * urgency_factor * vol_factor
        
        # 범위 제한
        interval = np.clip(
            interval,
            self.params['min_interval_seconds'],
            self.params['max_interval_seconds']
        )
        
        return int(interval)
    
    def _estimate_slippage(self, 
                         order_size: float,
                         available_liquidity: float,
                         volatility: float) -> float:
        """슬리피지 추정"""
        
        if available_liquidity == 0:
            return volatility * 0.01
        
        # 유동성 소비 비율
        consumption_ratio = order_size / available_liquidity
        
        # 슬리피지 모델
        if consumption_ratio < 0.1:
            # 작은 주문
            slippage = volatility * 0.001
        elif consumption_ratio < 0.5:
            # 중간 주문
            slippage = volatility * 0.005 * consumption_ratio
        else:
            # 큰 주문
            slippage = volatility * 0.01 * np.sqrt(consumption_ratio)
        
        return slippage
    
    def execute_pyramid(self, 
                       plan: PyramidPlan,
                       current_price: float,
                       current_time: datetime) -> Optional[Dict]:
        """피라미드 실행
        
        Args:
            plan: 피라미드 계획
            current_price: 현재 가격
            current_time: 현재 시간
            
        Returns:
            실행 정보 또는 None
        """
        
        # 활성 피라미드 체크
        pyramid_id = f"{plan.symbol}_{plan.side}_{current_time.timestamp()}"
        
        if pyramid_id not in self.active_pyramids:
            # 새 피라미드 시작
            self.active_pyramids[pyramid_id] = {
                'plan': plan,
                'executed_levels': [],
                'remaining_levels': list(plan.entry_levels),
                'start_time': current_time,
                'last_execution': None,
                'total_executed': 0
            }
        
        pyramid = self.active_pyramids[pyramid_id]
        
        # 다음 레벨 체크
        if not pyramid['remaining_levels']:
            # 완료
            del self.active_pyramids[pyramid_id]
            return None
        
        next_level = pyramid['remaining_levels'][0]
        
        # 시간 간격 체크
        if pyramid['last_execution']:
            elapsed = (current_time - pyramid['last_execution']).total_seconds()
            if elapsed < plan.time_interval:
                return None
        
        # 가격 조건 체크
        should_execute = False
        
        if next_level.trigger_condition == 'immediate':
            should_execute = True
            
        elif 'vwap_band' in next_level.trigger_condition:
            if plan.side == 'buy' and current_price <= next_level.price:
                should_execute = True
            elif plan.side == 'sell' and current_price >= next_level.price:
                should_execute = True
                
        elif 'liquidity_pocket' in next_level.trigger_condition:
            # 유동성 포켓 근처 체크
            price_distance = abs(current_price - next_level.price) / current_price
            if price_distance < 0.001:  # 0.1% 이내
                should_execute = True
        
        if not should_execute:
            return None
        
        # 실행
        execution = {
            'pyramid_id': pyramid_id,
            'level_index': len(pyramid['executed_levels']),
            'price': current_price,
            'intended_price': next_level.price,
            'size': next_level.size,
            'size_pct': next_level.size_pct,
            'slippage': abs(current_price - next_level.price),
            'expected_slippage': next_level.expected_slippage,
            'timestamp': current_time
        }
        
        # 상태 업데이트
        pyramid['executed_levels'].append(execution)
        pyramid['remaining_levels'].pop(0)
        pyramid['last_execution'] = current_time
        pyramid['total_executed'] += next_level.size
        
        # 히스토리 기록
        self.execution_history.append(execution)
        
        logger.info(f"피라미드 레벨 실행: {execution['level_index']+1}/{plan.max_levels}, "
                   f"크기: {next_level.size:.4f} @ {current_price:.4f}")
        
        return execution
    
    def get_active_pyramids(self) -> Dict:
        """활성 피라미드 조회"""
        return self.active_pyramids.copy()
    
    def cancel_pyramid(self, pyramid_id: str) -> bool:
        """피라미드 취소"""
        
        if pyramid_id in self.active_pyramids:
            del self.active_pyramids[pyramid_id]
            logger.info(f"피라미드 취소: {pyramid_id}")
            return True
        
        return False
    
    def get_execution_stats(self) -> Dict:
        """실행 통계"""
        
        if not self.execution_history:
            return {}
        
        df = pd.DataFrame(self.execution_history)
        
        stats = {
            'total_executions': len(df),
            'avg_slippage': df['slippage'].mean(),
            'max_slippage': df['slippage'].max(),
            'slippage_vs_expected': (df['slippage'] / df['expected_slippage']).mean()
                                   if df['expected_slippage'].sum() > 0 else 0,
            'avg_levels_per_pyramid': df.groupby('pyramid_id')['level_index'].max().mean() + 1,
            'completion_rate': len(df[df['level_index'] == df.groupby('pyramid_id')['level_index'].transform('max')]) 
                             / df['pyramid_id'].nunique() if df['pyramid_id'].nunique() > 0 else 0
        }
        
        return stats


class MarketImpactEstimator:
    """시장 충격 추정기"""
    
    def estimate(self,
                order_size: float,
                market_depth: float,
                volatility: float,
                model: str = 'sqrt') -> float:
        """시장 충격 추정
        
        Args:
            order_size: 주문 크기
            market_depth: 시장 심도
            volatility: 변동성
            model: 충격 모델
            
        Returns:
            예상 충격 (비율)
        """
        
        if market_depth == 0:
            return volatility * 0.01
        
        # 주문 크기 비율
        size_ratio = order_size / market_depth
        
        if model == 'linear':
            # 선형 모델
            impact = 0.001 * size_ratio
            
        elif model == 'sqrt':
            # 제곱근 모델 (Kyle)
            impact = 0.001 * volatility * np.sqrt(size_ratio)
            
        elif model == 'power':
            # 멱법칙
            alpha = 0.6  # 경험적 지수
            impact = 0.001 * (size_ratio ** alpha)
            
        else:
            impact = 0
        
        return impact