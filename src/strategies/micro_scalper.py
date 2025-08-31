"""극초단 스캘핑 전략"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging
from collections import deque
from dataclasses import dataclass

from ..utils.logging import Logger

logger = Logger.get_logger(__name__)

@dataclass
class ScalpSignal:
    """스캘핑 신호"""
    action: str  # 'scalp'
    side: str  # 'buy' or 'sell'
    entry_price: float
    tp_price: float
    sl_price: float
    tp_bps: float
    sl_bps: float
    size_pct: float
    max_hold_seconds: int
    confidence: float
    reason: str

class MicroScalper:
    """극초단 스캘핑 전략
    
    틱 레벨 불균형과 미세 가격 움직임을 활용
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Args:
            config: 전략 설정
        """
        self.config = config or {}
        
        # 기본 파라미터
        self.params = {
            'max_spread_bps': 0.5,  # 최대 스프레드 (0.5 bps)
            'min_depth_imbalance': 0.7,  # 최소 심도 불균형
            'tick_momentum_threshold': 0.001,  # 틱 모멘텀 임계값
            'min_volume_spike': 2.0,  # 최소 볼륨 스파이크
            'max_position_pct': 0.1,  # 최대 포지션 비율
            'default_tp_bps': 1.0,  # 기본 TP (1 bps)
            'default_sl_bps': 0.8,  # 기본 SL (0.8 bps)
            'max_hold_seconds': 60,  # 최대 홀딩 시간
            'min_profit_bps': 0.3,  # 최소 수익 (수수료 후)
            'tick_window': 10  # 틱 윈도우
        }
        
        if config:
            self.params.update(config)
        
        # 틱 데이터 버퍼
        self.tick_buffer = deque(maxlen=1000)
        self.trade_buffer = deque(maxlen=100)
        
        # 상태
        self.active_scalps = {}
        self.performance = []
        
        # 마켓 마이크로구조 추적
        self.microstructure = {
            'avg_spread': 0,
            'avg_depth': 0,
            'tick_rate': 0,
            'volume_profile': {}
        }
    
    def find_opportunity(self,
                        spread_bps: float,
                        depth_imbalance: float,
                        recent_trades: List[Dict],
                        best_bid: float,
                        best_ask: float,
                        bid_depth: float,
                        ask_depth: float,
                        current_time: datetime) -> Optional[ScalpSignal]:
        """스캘핑 기회 탐색
        
        Args:
            spread_bps: 현재 스프레드 (bps)
            depth_imbalance: 심도 불균형 (bid_depth / total_depth)
            recent_trades: 최근 체결 내역
            best_bid: 최우선 매수 호가
            best_ask: 최우선 매도 호가
            bid_depth: 매수 심도
            ask_depth: 매도 심도
            current_time: 현재 시간
            
        Returns:
            스캘핑 신호 또는 None
        """
        
        # 스프레드 체크
        if spread_bps > self.params['max_spread_bps']:
            return None
        
        # 틱 데이터 업데이트
        self._update_tick_buffer(recent_trades)
        
        # 틱 모멘텀 계산
        tick_momentum = self._calculate_tick_momentum(recent_trades)
        
        if abs(tick_momentum) < self.params['tick_momentum_threshold']:
            return None
        
        # 심도 불균형 분석
        imbalance_signal = self._analyze_depth_imbalance(
            depth_imbalance, bid_depth, ask_depth
        )
        
        # 볼륨 프로파일 분석
        volume_signal = self._analyze_volume_profile(recent_trades)
        
        # 가격 레벨 분석
        level_signal = self._analyze_price_levels(best_bid, best_ask)
        
        # 종합 신호
        if imbalance_signal and abs(tick_momentum) > self.params['tick_momentum_threshold']:
            return self._generate_scalp_signal(
                tick_momentum,
                depth_imbalance,
                spread_bps,
                best_bid,
                best_ask,
                imbalance_signal,
                volume_signal,
                level_signal
            )
        
        return None
    
    def _update_tick_buffer(self, recent_trades: List[Dict]):
        """틱 버퍼 업데이트"""
        
        for trade in recent_trades:
            self.tick_buffer.append({
                'timestamp': trade.get('timestamp', datetime.now()),
                'price': trade['price'],
                'quantity': trade['quantity'],
                'side': trade.get('side', 'unknown')
            })
    
    def _calculate_tick_momentum(self, recent_trades: List[Dict]) -> float:
        """틱 모멘텀 계산"""
        
        if len(recent_trades) < self.params['tick_window']:
            return 0
        
        # 가격 변화
        prices = [t['price'] for t in recent_trades[-self.params['tick_window']:]]
        
        if len(prices) < 2:
            return 0
        
        # 가중 모멘텀 (최근 거래에 더 높은 가중치)
        weights = np.exp(np.linspace(-1, 0, len(prices)))
        weights /= weights.sum()
        
        price_changes = np.diff(prices)
        if len(price_changes) == 0:
            return 0
        
        # 가중치 조정
        if len(weights) > len(price_changes):
            weights = weights[-len(price_changes):]
        
        weighted_momentum = np.average(price_changes, weights=weights)
        
        # 정규화
        avg_price = np.mean(prices)
        if avg_price > 0:
            normalized_momentum = weighted_momentum / avg_price
        else:
            normalized_momentum = 0
        
        return normalized_momentum
    
    def _analyze_depth_imbalance(self,
                                depth_imbalance: float,
                                bid_depth: float,
                                ask_depth: float) -> Optional[str]:
        """심도 불균형 분석"""
        
        # 극단적 불균형
        if depth_imbalance > self.params['min_depth_imbalance']:
            # 매수 심도가 강함 → 가격 상승 가능성
            return 'buy_pressure'
        elif depth_imbalance < (1 - self.params['min_depth_imbalance']):
            # 매도 심도가 강함 → 가격 하락 가능성
            return 'sell_pressure'
        
        # 심도 절대량 체크
        total_depth = bid_depth + ask_depth
        if total_depth < self.microstructure.get('avg_depth', 10000) * 0.5:
            # 유동성 부족
            return None
        
        return 'balanced'
    
    def _analyze_volume_profile(self, recent_trades: List[Dict]) -> Optional[str]:
        """볼륨 프로파일 분석"""
        
        if not recent_trades:
            return None
        
        # 볼륨 집계
        buy_volume = sum(t['quantity'] for t in recent_trades 
                        if t.get('side') == 'buy')
        sell_volume = sum(t['quantity'] for t in recent_trades 
                         if t.get('side') == 'sell')
        
        total_volume = buy_volume + sell_volume
        
        if total_volume == 0:
            return None
        
        # 볼륨 불균형
        buy_ratio = buy_volume / total_volume
        
        if buy_ratio > 0.7:
            return 'heavy_buying'
        elif buy_ratio < 0.3:
            return 'heavy_selling'
        
        # 볼륨 스파이크
        avg_volume = np.mean([t['quantity'] for t in self.tick_buffer]) if self.tick_buffer else 0
        
        if avg_volume > 0 and total_volume > avg_volume * self.params['min_volume_spike']:
            return 'volume_spike'
        
        return 'normal_volume'
    
    def _analyze_price_levels(self, best_bid: float, best_ask: float) -> Optional[str]:
        """가격 레벨 분석"""
        
        mid_price = (best_bid + best_ask) / 2
        
        # 라운드 넘버 근처 체크
        round_level = round(mid_price, -1)  # 10 단위
        distance_to_round = abs(mid_price - round_level) / mid_price
        
        if distance_to_round < 0.001:  # 0.1% 이내
            return 'near_round_number'
        
        # 지지/저항 레벨 (간단한 구현)
        if self.tick_buffer:
            recent_prices = [t['price'] for t in list(self.tick_buffer)[-100:]]
            if recent_prices:
                price_counts = pd.Series(recent_prices).value_counts()
                
                if len(price_counts) > 0:
                    most_traded_price = price_counts.index[0]
                    
                    if abs(mid_price - most_traded_price) / mid_price < 0.001:
                        return 'at_support_resistance'
        
        return 'normal_level'
    
    def _generate_scalp_signal(self,
                              tick_momentum: float,
                              depth_imbalance: float,
                              spread_bps: float,
                              best_bid: float,
                              best_ask: float,
                              imbalance_signal: str,
                              volume_signal: Optional[str],
                              level_signal: Optional[str]) -> ScalpSignal:
        """스캘핑 신호 생성"""
        
        # 방향 결정
        if tick_momentum > 0 and imbalance_signal == 'buy_pressure':
            side = 'buy'
            entry_price = best_ask  # 테이커로 진입
            
        elif tick_momentum < 0 and imbalance_signal == 'sell_pressure':
            side = 'sell'
            entry_price = best_bid  # 테이커로 진입
            
        else:
            # 모멘텀과 불균형이 일치하지 않음
            return None
        
        # TP/SL 계산
        tp_bps = self.params['default_tp_bps']
        sl_bps = self.params['default_sl_bps']
        
        # 볼륨 신호에 따른 조정
        if volume_signal == 'volume_spike':
            tp_bps *= 1.5  # 더 큰 움직임 기대
            
        elif volume_signal in ['heavy_buying', 'heavy_selling']:
            tp_bps *= 1.2
        
        # 레벨 신호에 따른 조정
        if level_signal == 'near_round_number':
            sl_bps *= 1.2  # 라운드 넘버에서는 더 넓은 스탑
            
        elif level_signal == 'at_support_resistance':
            tp_bps *= 0.8  # 지지/저항에서는 보수적
        
        # 가격 계산
        if side == 'buy':
            tp_price = entry_price * (1 + tp_bps / 10000)
            sl_price = entry_price * (1 - sl_bps / 10000)
        else:
            tp_price = entry_price * (1 - tp_bps / 10000)
            sl_price = entry_price * (1 + sl_bps / 10000)
        
        # 신뢰도 계산
        confidence = self._calculate_confidence(
            tick_momentum, depth_imbalance, spread_bps,
            volume_signal, level_signal
        )
        
        # 사이즈 계산
        size_pct = min(
            self.params['max_position_pct'],
            self.params['max_position_pct'] * confidence
        )
        
        # 이유 생성
        reasons = []
        if abs(tick_momentum) > self.params['tick_momentum_threshold'] * 2:
            reasons.append('strong_momentum')
        if imbalance_signal in ['buy_pressure', 'sell_pressure']:
            reasons.append(imbalance_signal)
        if volume_signal == 'volume_spike':
            reasons.append('volume_spike')
        
        reason = '_'.join(reasons) if reasons else 'micro_inefficiency'
        
        return ScalpSignal(
            action='scalp',
            side=side,
            entry_price=entry_price,
            tp_price=tp_price,
            sl_price=sl_price,
            tp_bps=tp_bps,
            sl_bps=sl_bps,
            size_pct=size_pct,
            max_hold_seconds=self.params['max_hold_seconds'],
            confidence=confidence,
            reason=reason
        )
    
    def _calculate_confidence(self,
                             tick_momentum: float,
                             depth_imbalance: float,
                             spread_bps: float,
                             volume_signal: Optional[str],
                             level_signal: Optional[str]) -> float:
        """신뢰도 계산"""
        
        confidence = 0.5  # 기본
        
        # 모멘텀 강도
        momentum_strength = abs(tick_momentum) / self.params['tick_momentum_threshold']
        confidence += min(0.2, momentum_strength * 0.1)
        
        # 심도 불균형 강도
        imbalance_strength = abs(depth_imbalance - 0.5) * 2
        confidence += min(0.2, imbalance_strength * 0.2)
        
        # 스프레드 (낮을수록 좋음)
        spread_score = 1 - (spread_bps / self.params['max_spread_bps'])
        confidence += spread_score * 0.1
        
        # 볼륨 신호
        if volume_signal in ['volume_spike', 'heavy_buying', 'heavy_selling']:
            confidence += 0.1
        
        # 레벨 신호
        if level_signal == 'at_support_resistance':
            confidence += 0.05
        elif level_signal == 'near_round_number':
            confidence -= 0.05  # 라운드 넘버는 불확실성
        
        return np.clip(confidence, 0.1, 0.9)
    
    def manage_scalps(self, current_time: datetime, current_prices: Dict[str, float]) -> List[Dict]:
        """스캘핑 포지션 관리"""
        
        actions = []
        
        for scalp_id, scalp in list(self.active_scalps.items()):
            symbol = scalp['symbol']
            
            if symbol not in current_prices:
                continue
            
            current_price = current_prices[symbol]
            entry_time = scalp['entry_time']
            
            # 시간 체크
            hold_time = (current_time - entry_time).total_seconds()
            
            if hold_time > scalp['max_hold_seconds']:
                # 시간 종료
                actions.append({
                    'scalp_id': scalp_id,
                    'action': 'close',
                    'reason': 'timeout',
                    'pnl': self._calculate_scalp_pnl(scalp, current_price)
                })
                del self.active_scalps[scalp_id]
                continue
            
            # TP/SL 체크
            if scalp['side'] == 'buy':
                if current_price >= scalp['tp_price']:
                    actions.append({
                        'scalp_id': scalp_id,
                        'action': 'close',
                        'reason': 'take_profit',
                        'pnl': self._calculate_scalp_pnl(scalp, scalp['tp_price'])
                    })
                    del self.active_scalps[scalp_id]
                    
                elif current_price <= scalp['sl_price']:
                    actions.append({
                        'scalp_id': scalp_id,
                        'action': 'close',
                        'reason': 'stop_loss',
                        'pnl': self._calculate_scalp_pnl(scalp, scalp['sl_price'])
                    })
                    del self.active_scalps[scalp_id]
            
            else:  # sell
                if current_price <= scalp['tp_price']:
                    actions.append({
                        'scalp_id': scalp_id,
                        'action': 'close',
                        'reason': 'take_profit',
                        'pnl': self._calculate_scalp_pnl(scalp, scalp['tp_price'])
                    })
                    del self.active_scalps[scalp_id]
                    
                elif current_price >= scalp['sl_price']:
                    actions.append({
                        'scalp_id': scalp_id,
                        'action': 'close',
                        'reason': 'stop_loss',
                        'pnl': self._calculate_scalp_pnl(scalp, scalp['sl_price'])
                    })
                    del self.active_scalps[scalp_id]
        
        return actions
    
    def _calculate_scalp_pnl(self, scalp: Dict, exit_price: float) -> float:
        """스캘핑 PnL 계산 (bps)"""
        
        entry_price = scalp['entry_price']
        
        if scalp['side'] == 'buy':
            pnl_bps = (exit_price - entry_price) / entry_price * 10000
        else:
            pnl_bps = (entry_price - exit_price) / entry_price * 10000
        
        # 수수료 차감 (테이커 양방향)
        fee_bps = 5 * 2  # 5 bps * 2
        net_pnl_bps = pnl_bps - fee_bps
        
        return net_pnl_bps
    
    def update_microstructure(self, spread: float, depth: float, tick_rate: float):
        """마켓 마이크로구조 업데이트"""
        
        # 지수 이동 평균
        alpha = 0.1
        
        self.microstructure['avg_spread'] = (
            alpha * spread + 
            (1 - alpha) * self.microstructure.get('avg_spread', spread)
        )
        
        self.microstructure['avg_depth'] = (
            alpha * depth + 
            (1 - alpha) * self.microstructure.get('avg_depth', depth)
        )
        
        self.microstructure['tick_rate'] = tick_rate
    
    def get_statistics(self) -> Dict:
        """스캘핑 통계"""
        
        if not self.performance:
            return {}
        
        df = pd.DataFrame(self.performance)
        
        stats = {
            'total_scalps': len(df),
            'active_scalps': len(self.active_scalps),
            'total_pnl_bps': df['pnl_bps'].sum() if 'pnl_bps' in df else 0,
            'avg_pnl_bps': df['pnl_bps'].mean() if 'pnl_bps' in df else 0,
            'win_rate': (df['pnl_bps'] > 0).mean() if 'pnl_bps' in df else 0,
            'avg_hold_time': df['hold_time'].mean() if 'hold_time' in df else 0,
            'sharpe_ratio': (
                df['pnl_bps'].mean() / df['pnl_bps'].std() * np.sqrt(252 * 24 * 60) 
                if 'pnl_bps' in df and df['pnl_bps'].std() > 0 else 0
            )
        }
        
        return stats