"""거래 상태 관리"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from collections import deque
import threading
import json
import logging

from ..utils.logging import Logger

logger = Logger.get_logger(__name__)

class TradingState:
    """거래 상태 관리
    
    실시간 시장 상태, 신호, 포지션 등 전체 거래 상태 관리
    """
    
    def __init__(self, config: Dict):
        """
        Args:
            config: 거래 설정
        """
        self.config = config
        
        # 스레드 안전성을 위한 락
        self.lock = threading.RLock()
        
        # 시장 상태
        self.market_state = {
            'prices': {},  # symbol -> price
            'spreads': {},  # symbol -> spread_bps
            'depths': {},  # symbol -> depth_info
            'volumes': {},  # symbol -> volume
            'funding_rates': {},  # symbol -> funding_rate
            'volatilities': {},  # symbol -> volatility
            'last_update': {}  # symbol -> timestamp
        }
        
        # 특징 상태
        self.features = {}  # symbol -> features dict
        self.feature_history = {}  # symbol -> deque of features
        
        # 신호 상태
        self.signals = {}  # symbol -> signal dict
        self.signal_history = deque(maxlen=1000)
        
        # 포지션 상태
        self.positions = {}  # symbol -> position dict
        self.open_orders = {}  # order_id -> order dict
        
        # 성과 추적
        self.performance = {
            'realized_pnl': 0,
            'unrealized_pnl': 0,
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'fees_paid': 0,
            'funding_received': 0,
            'max_drawdown': 0,
            'current_drawdown': 0,
            'equity_high': self.config.get('initial_capital', 100000)
        }
        
        # 리스크 상태
        self.risk_state = {
            'total_exposure': 0,
            'leverage': 0,
            'var_95': 0,
            'daily_pnl': 0,
            'daily_trades': 0,
            'risk_level': 'normal',  # normal, warning, critical
            'circuit_breaker': False
        }
        
        # 레짐 상태 (v2.0)
        self.regime_state = {
            'current_regime': 'normal',
            'volatility_state': 'normal',
            'trend_state': 'neutral',
            'liquidity_state': 'normal',
            'regime_confidence': 1.0,
            'last_regime_change': None
        }
        
        # 실행 상태
        self.execution_state = {
            'pending_orders': {},  # order_id -> order
            'active_pyramids': {},  # pyramid_id -> pyramid_state
            'last_execution': {},  # symbol -> timestamp
            'execution_stats': {
                'total_orders': 0,
                'filled_orders': 0,
                'rejected_orders': 0,
                'avg_slippage': 0
            }
        }
        
        # 시스템 상태
        self.system_state = {
            'status': 'initializing',  # initializing, running, paused, stopped
            'start_time': datetime.now(),
            'last_heartbeat': datetime.now(),
            'errors': [],
            'warnings': [],
            'info_messages': []
        }
        
        # 상태 스냅샷 (복구용)
        self.snapshots = deque(maxlen=100)
        self.last_snapshot_time = None
    
    def update_market_data(self, symbol: str, data: Dict):
        """시장 데이터 업데이트
        
        Args:
            symbol: 심볼
            data: 시장 데이터
        """
        
        with self.lock:
            # 가격
            if 'price' in data:
                self.market_state['prices'][symbol] = data['price']
            
            # 스프레드
            if 'spread_bps' in data:
                self.market_state['spreads'][symbol] = data['spread_bps']
            
            # 심도
            if 'depth' in data:
                self.market_state['depths'][symbol] = {
                    'bid_depth': data.get('bid_depth', 0),
                    'ask_depth': data.get('ask_depth', 0),
                    'total_depth': data.get('total_depth', 0),
                    'imbalance': data.get('depth_imbalance', 0.5)
                }
            
            # 볼륨
            if 'volume' in data:
                self.market_state['volumes'][symbol] = data['volume']
            
            # 펀딩
            if 'funding_rate' in data:
                self.market_state['funding_rates'][symbol] = data['funding_rate']
            
            # 변동성
            if 'volatility' in data:
                self.market_state['volatilities'][symbol] = data['volatility']
            
            # 업데이트 시간
            self.market_state['last_update'][symbol] = datetime.now()
    
    def update_features(self, symbol: str, features: Dict):
        """특징 업데이트
        
        Args:
            symbol: 심볼
            features: 특징 딕셔너리
        """
        
        with self.lock:
            # 현재 특징
            self.features[symbol] = features
            
            # 히스토리
            if symbol not in self.feature_history:
                self.feature_history[symbol] = deque(maxlen=100)
            
            self.feature_history[symbol].append({
                'timestamp': datetime.now(),
                'features': features.copy()
            })
    
    def update_signal(self, symbol: str, signal: Dict):
        """신호 업데이트
        
        Args:
            symbol: 심볼
            signal: 신호 딕셔너리
        """
        
        with self.lock:
            # 현재 신호
            self.signals[symbol] = signal
            
            # 히스토리
            self.signal_history.append({
                'timestamp': datetime.now(),
                'symbol': symbol,
                'signal': signal.copy()
            })
            
            logger.info(f"신호 업데이트: {symbol} - {signal.get('action', 'none')}")
    
    def update_position(self, symbol: str, position: Optional[Dict]):
        """포지션 업데이트
        
        Args:
            symbol: 심볼
            position: 포지션 정보 또는 None (청산)
        """
        
        with self.lock:
            if position:
                # 포지션 업데이트
                self.positions[symbol] = position
                
                # 미실현 손익 계산
                if 'current_price' in position and 'entry_price' in position:
                    size = position.get('size', 0)
                    
                    if position['side'] == 'long':
                        unrealized = (position['current_price'] - position['entry_price']) * size
                    else:
                        unrealized = (position['entry_price'] - position['current_price']) * size
                    
                    position['unrealized_pnl'] = unrealized
            else:
                # 포지션 제거
                if symbol in self.positions:
                    del self.positions[symbol]
    
    def update_order(self, order_id: str, order: Dict):
        """주문 업데이트
        
        Args:
            order_id: 주문 ID
            order: 주문 정보
        """
        
        with self.lock:
            if order['status'] in ['pending', 'partial']:
                self.open_orders[order_id] = order
            else:
                # 완료/취소된 주문 제거
                if order_id in self.open_orders:
                    del self.open_orders[order_id]
                
                # 실행 통계 업데이트
                self.execution_state['execution_stats']['total_orders'] += 1
                
                if order['status'] == 'filled':
                    self.execution_state['execution_stats']['filled_orders'] += 1
                elif order['status'] == 'rejected':
                    self.execution_state['execution_stats']['rejected_orders'] += 1
    
    def update_performance(self, trade_result: Dict):
        """성과 업데이트
        
        Args:
            trade_result: 거래 결과
        """
        
        with self.lock:
            # 실현 손익
            pnl = trade_result.get('pnl', 0)
            self.performance['realized_pnl'] += pnl
            
            # 거래 수
            self.performance['total_trades'] += 1
            
            if pnl > 0:
                self.performance['winning_trades'] += 1
            else:
                self.performance['losing_trades'] += 1
            
            # 수수료
            self.performance['fees_paid'] += trade_result.get('fees', 0)
            
            # 펀딩
            self.performance['funding_received'] += trade_result.get('funding', 0)
            
            # 자산 업데이트
            total_equity = (
                self.config.get('initial_capital', 100000) +
                self.performance['realized_pnl'] +
                self.performance['unrealized_pnl']
            )
            
            # 최대 드로우다운
            if total_equity > self.performance['equity_high']:
                self.performance['equity_high'] = total_equity
            
            drawdown = (self.performance['equity_high'] - total_equity) / self.performance['equity_high']
            self.performance['current_drawdown'] = drawdown
            self.performance['max_drawdown'] = max(self.performance['max_drawdown'], drawdown)
            
            # 일일 통계
            self.risk_state['daily_pnl'] += pnl
            self.risk_state['daily_trades'] += 1
    
    def update_risk_state(self):
        """리스크 상태 업데이트"""
        
        with self.lock:
            # 총 노출
            total_exposure = sum(
                abs(p.get('notional', 0)) 
                for p in self.positions.values()
            )
            self.risk_state['total_exposure'] = total_exposure
            
            # 레버리지
            capital = self.config.get('initial_capital', 100000) + self.performance['realized_pnl']
            
            if capital > 0:
                self.risk_state['leverage'] = total_exposure / capital
            else:
                self.risk_state['leverage'] = 0
            
            # VaR 계산 (간단한 추정)
            if self.feature_history:
                recent_returns = []
                
                for symbol, history in self.feature_history.items():
                    if len(history) > 1:
                        returns = [
                            h['features'].get('return_1m', 0) 
                            for h in list(history)[-20:]
                        ]
                        recent_returns.extend(returns)
                
                if recent_returns:
                    self.risk_state['var_95'] = np.percentile(recent_returns, 5) * total_exposure
            
            # 리스크 레벨
            if self.risk_state['leverage'] > 5:
                self.risk_state['risk_level'] = 'critical'
            elif self.risk_state['leverage'] > 3:
                self.risk_state['risk_level'] = 'warning'
            else:
                self.risk_state['risk_level'] = 'normal'
            
            # 서킷 브레이커
            daily_loss_limit = self.config.get('risk', {}).get('daily_stop_pct', 0.02)
            
            if self.risk_state['daily_pnl'] < -capital * daily_loss_limit:
                self.risk_state['circuit_breaker'] = True
                logger.warning("서킷 브레이커 발동!")
    
    def update_regime(self, regime_info: Dict):
        """레짐 상태 업데이트 (v2.0)
        
        Args:
            regime_info: 레짐 정보
        """
        
        with self.lock:
            old_regime = self.regime_state['current_regime']
            new_regime = regime_info.get('regime', 'normal')
            
            # 레짐 변경
            if old_regime != new_regime:
                self.regime_state['last_regime_change'] = datetime.now()
                logger.info(f"레짐 변경: {old_regime} -> {new_regime}")
            
            # 상태 업데이트
            self.regime_state.update({
                'current_regime': new_regime,
                'volatility_state': regime_info.get('vol_state', 'normal'),
                'trend_state': regime_info.get('trend_state', 'neutral'),
                'liquidity_state': regime_info.get('liquidity_state', 'normal'),
                'regime_confidence': regime_info.get('confidence', 1.0)
            })
    
    def update_system_status(self, status: str, message: Optional[str] = None):
        """시스템 상태 업데이트
        
        Args:
            status: 상태 (running, paused, stopped, error)
            message: 메시지
        """
        
        with self.lock:
            self.system_state['status'] = status
            self.system_state['last_heartbeat'] = datetime.now()
            
            if message:
                if status == 'error':
                    self.system_state['errors'].append({
                        'timestamp': datetime.now(),
                        'message': message
                    })
                elif status == 'warning':
                    self.system_state['warnings'].append({
                        'timestamp': datetime.now(),
                        'message': message
                    })
                else:
                    self.system_state['info_messages'].append({
                        'timestamp': datetime.now(),
                        'message': message
                    })
    
    def get_position(self, symbol: str) -> Optional[Dict]:
        """포지션 조회
        
        Args:
            symbol: 심볼
            
        Returns:
            포지션 정보 또는 None
        """
        
        with self.lock:
            return self.positions.get(symbol)
    
    def get_all_positions(self) -> Dict[str, Dict]:
        """모든 포지션 조회
        
        Returns:
            포지션 딕셔너리
        """
        
        with self.lock:
            return self.positions.copy()
    
    def get_market_price(self, symbol: str) -> Optional[float]:
        """시장 가격 조회
        
        Args:
            symbol: 심볼
            
        Returns:
            가격 또는 None
        """
        
        with self.lock:
            return self.market_state['prices'].get(symbol)
    
    def get_total_equity(self) -> float:
        """총 자산 조회
        
        Returns:
            총 자산
        """
        
        with self.lock:
            # 미실현 손익 계산
            unrealized = sum(
                p.get('unrealized_pnl', 0) 
                for p in self.positions.values()
            )
            
            self.performance['unrealized_pnl'] = unrealized
            
            return (
                self.config.get('initial_capital', 100000) +
                self.performance['realized_pnl'] +
                unrealized
            )
    
    def get_performance_metrics(self) -> Dict:
        """성과 메트릭 조회
        
        Returns:
            성과 메트릭
        """
        
        with self.lock:
            metrics = self.performance.copy()
            
            # 추가 메트릭 계산
            if metrics['total_trades'] > 0:
                metrics['win_rate'] = metrics['winning_trades'] / metrics['total_trades']
                metrics['avg_pnl'] = metrics['realized_pnl'] / metrics['total_trades']
            else:
                metrics['win_rate'] = 0
                metrics['avg_pnl'] = 0
            
            # Sharpe Ratio (간단한 추정)
            if hasattr(self, '_daily_returns'):
                if len(self._daily_returns) > 1:
                    returns = np.array(self._daily_returns)
                    metrics['sharpe_ratio'] = (
                        np.mean(returns) / np.std(returns) * np.sqrt(252)
                        if np.std(returns) > 0 else 0
                    )
            
            return metrics
    
    def should_trade(self, symbol: str) -> bool:
        """거래 가능 여부 확인
        
        Args:
            symbol: 심볼
            
        Returns:
            거래 가능 여부
        """
        
        with self.lock:
            # 시스템 상태 체크
            if self.system_state['status'] != 'running':
                return False
            
            # 서킷 브레이커 체크
            if self.risk_state['circuit_breaker']:
                return False
            
            # 리스크 레벨 체크
            if self.risk_state['risk_level'] == 'critical':
                return False
            
            # 포지션 수 제한
            max_positions = self.config.get('max_positions', 10)
            if len(self.positions) >= max_positions:
                # 이미 포지션이 있는 심볼은 허용
                if symbol not in self.positions:
                    return False
            
            # 일일 거래 수 제한
            max_daily_trades = self.config.get('max_daily_trades', 50)
            if self.risk_state['daily_trades'] >= max_daily_trades:
                return False
            
            return True
    
    def create_snapshot(self) -> Dict:
        """상태 스냅샷 생성
        
        Returns:
            상태 스냅샷
        """
        
        with self.lock:
            snapshot = {
                'timestamp': datetime.now(),
                'market_state': self.market_state.copy(),
                'positions': self.positions.copy(),
                'performance': self.performance.copy(),
                'risk_state': self.risk_state.copy(),
                'regime_state': self.regime_state.copy(),
                'system_state': self.system_state.copy()
            }
            
            # 스냅샷 저장
            self.snapshots.append(snapshot)
            self.last_snapshot_time = datetime.now()
            
            return snapshot
    
    def restore_from_snapshot(self, snapshot: Dict):
        """스냅샷에서 복구
        
        Args:
            snapshot: 상태 스냅샷
        """
        
        with self.lock:
            self.market_state = snapshot.get('market_state', {})
            self.positions = snapshot.get('positions', {})
            self.performance = snapshot.get('performance', {})
            self.risk_state = snapshot.get('risk_state', {})
            self.regime_state = snapshot.get('regime_state', {})
            self.system_state = snapshot.get('system_state', {})
            
            logger.info(f"스냅샷 복구 완료: {snapshot.get('timestamp')}")
    
    def reset_daily_stats(self):
        """일일 통계 리셋"""
        
        with self.lock:
            self.risk_state['daily_pnl'] = 0
            self.risk_state['daily_trades'] = 0
            self.risk_state['circuit_breaker'] = False
            
            # 일일 수익률 기록
            if not hasattr(self, '_daily_returns'):
                self._daily_returns = deque(maxlen=252)
            
            self._daily_returns.append(self.risk_state['daily_pnl'])
            
            logger.info("일일 통계 리셋")
    
    def get_state_summary(self) -> Dict:
        """상태 요약 조회
        
        Returns:
            상태 요약
        """
        
        with self.lock:
            return {
                'system_status': self.system_state['status'],
                'total_equity': self.get_total_equity(),
                'n_positions': len(self.positions),
                'total_exposure': self.risk_state['total_exposure'],
                'leverage': self.risk_state['leverage'],
                'risk_level': self.risk_state['risk_level'],
                'current_regime': self.regime_state['current_regime'],
                'realized_pnl': self.performance['realized_pnl'],
                'unrealized_pnl': self.performance['unrealized_pnl'],
                'win_rate': self.get_performance_metrics()['win_rate'],
                'daily_pnl': self.risk_state['daily_pnl'],
                'circuit_breaker': self.risk_state['circuit_breaker']
            }