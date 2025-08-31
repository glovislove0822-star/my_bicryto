"""포지션 관리"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from enum import Enum

from ..utils.logging import Logger

logger = Logger.get_logger(__name__)

class PositionSide(Enum):
    """포지션 방향"""
    LONG = "long"
    SHORT = "short"

class PositionStatus(Enum):
    """포지션 상태"""
    OPEN = "open"
    CLOSING = "closing"
    CLOSED = "closed"

@dataclass
class Position:
    """포지션 데이터 클래스"""
    symbol: str
    side: PositionSide
    entry_price: float
    size: float
    entry_time: datetime
    status: PositionStatus = PositionStatus.OPEN
    
    # 선택적 필드
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    realized_pnl: float = 0
    unrealized_pnl: float = 0
    fees_paid: float = 0
    funding_received: float = 0
    
    # 리스크 관리
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    trailing_stop: Optional[float] = None
    max_pnl: float = 0
    
    # 메타 정보
    strategy: str = ""
    tags: List[str] = None
    metadata: Dict = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if self.metadata is None:
            self.metadata = {}
        
        # Enum 변환
        if isinstance(self.side, str):
            self.side = PositionSide(self.side.lower())
        if isinstance(self.status, str):
            self.status = PositionStatus(self.status.lower())
    
    @property
    def notional(self) -> float:
        """명목 가치"""
        return self.size * self.entry_price
    
    def update_pnl(self, current_price: float):
        """손익 업데이트"""
        if self.side == PositionSide.LONG:
            self.unrealized_pnl = (current_price - self.entry_price) * self.size
        else:
            self.unrealized_pnl = (self.entry_price - current_price) * self.size
        
        # 최대 손익 추적
        self.max_pnl = max(self.max_pnl, self.unrealized_pnl)

class PositionManager:
    """포지션 관리자
    
    포지션 생성, 업데이트, 청산 및 리스크 관리
    """
    
    def __init__(self, config: Dict):
        """
        Args:
            config: 포지션 관리 설정
        """
        self.config = config
        
        # 포지션 저장소
        self.positions = {}  # symbol -> Position
        self.position_history = []  # 청산된 포지션 히스토리
        
        # 리스크 파라미터
        self.risk_params = config.get('risk', {})
        self.max_positions = config.get('max_positions', 10)
        self.max_position_size = config.get('max_position_size', 0.3)
        
        # 포지션 통계
        self.stats = {
            'total_opened': 0,
            'total_closed': 0,
            'total_pnl': 0,
            'total_fees': 0,
            'total_funding': 0,
            'avg_holding_time': 0
        }
    
    def open_position(self,
                     symbol: str,
                     side: str,
                     size: float,
                     entry_price: float,
                     stop_loss: Optional[float] = None,
                     take_profit: Optional[float] = None,
                     trailing_stop: Optional[float] = None,
                     strategy: str = "",
                     metadata: Optional[Dict] = None) -> Optional[Position]:
        """포지션 오픈
        
        Args:
            symbol: 심볼
            side: 방향 (long/short)
            size: 크기
            entry_price: 진입 가격
            stop_loss: 손절가
            take_profit: 익절가
            trailing_stop: 트레일링 스탑
            strategy: 전략 이름
            metadata: 메타데이터
            
        Returns:
            생성된 포지션 또는 None
        """
        
        # 포지션 수 체크
        if len(self.positions) >= self.max_positions:
            if symbol not in self.positions:
                logger.warning(f"최대 포지션 수 도달: {self.max_positions}")
                return None
        
        # 기존 포지션 체크
        if symbol in self.positions:
            logger.warning(f"이미 포지션 존재: {symbol}")
            return None
        
        # 포지션 생성
        position = Position(
            symbol=symbol,
            side=PositionSide(side.lower()),
            entry_price=entry_price,
            size=size,
            entry_time=datetime.now(),
            stop_loss=stop_loss,
            take_profit=take_profit,
            trailing_stop=trailing_stop,
            strategy=strategy,
            metadata=metadata or {}
        )
        
        # 저장
        self.positions[symbol] = position
        self.stats['total_opened'] += 1
        
        logger.info(f"포지션 오픈: {symbol} {side} {size}@{entry_price:.4f}")
        
        return position
    
    def close_position(self,
                      symbol: str,
                      exit_price: float,
                      reason: str = "manual",
                      fees: float = 0) -> Optional[Dict]:
        """포지션 청산
        
        Args:
            symbol: 심볼
            exit_price: 청산 가격
            reason: 청산 사유
            fees: 수수료
            
        Returns:
            청산 결과 또는 None
        """
        
        if symbol not in self.positions:
            logger.warning(f"포지션 없음: {symbol}")
            return None
        
        position = self.positions[symbol]
        
        # 청산 정보 업데이트
        position.exit_price = exit_price
        position.exit_time = datetime.now()
        position.status = PositionStatus.CLOSED
        
        # 실현 손익 계산
        if position.side == PositionSide.LONG:
            position.realized_pnl = (exit_price - position.entry_price) * position.size
        else:
            position.realized_pnl = (position.entry_price - exit_price) * position.size
        
        # 수수료 차감
        position.fees_paid += fees
        position.realized_pnl -= fees
        
        # 보유 시간
        holding_time = (position.exit_time - position.entry_time).total_seconds() / 3600
        
        # 결과
        result = {
            'symbol': symbol,
            'side': position.side.value,
            'entry_price': position.entry_price,
            'exit_price': exit_price,
            'size': position.size,
            'pnl': position.realized_pnl,
            'pnl_pct': position.realized_pnl / position.notional,
            'fees': position.fees_paid,
            'funding': position.funding_received,
            'holding_time_hours': holding_time,
            'reason': reason
        }
        
        # 히스토리에 추가
        self.position_history.append(position)
        
        # 통계 업데이트
        self.stats['total_closed'] += 1
        self.stats['total_pnl'] += position.realized_pnl
        self.stats['total_fees'] += position.fees_paid
        self.stats['total_funding'] += position.funding_received
        
        # 평균 보유 시간 업데이트
        n = self.stats['total_closed']
        self.stats['avg_holding_time'] = (
            (self.stats['avg_holding_time'] * (n - 1) + holding_time) / n
        )
        
        # 포지션 제거
        del self.positions[symbol]
        
        logger.info(f"포지션 청산: {symbol} PnL={position.realized_pnl:.2f} ({reason})")
        
        return result
    
    def update_position(self,
                       symbol: str,
                       current_price: float,
                       funding_rate: Optional[float] = None) -> Optional[Position]:
        """포지션 업데이트
        
        Args:
            symbol: 심볼
            current_price: 현재 가격
            funding_rate: 펀딩 레이트
            
        Returns:
            업데이트된 포지션 또는 None
        """
        
        if symbol not in self.positions:
            return None
        
        position = self.positions[symbol]
        
        # PnL 업데이트
        position.update_pnl(current_price)
        
        # 펀딩 업데이트
        if funding_rate:
            if position.side == PositionSide.LONG:
                # 롱 포지션은 펀딩 지불
                position.funding_received -= position.notional * funding_rate
            else:
                # 숏 포지션은 펀딩 수취
                position.funding_received += position.notional * funding_rate
        
        return position
    
    def check_exit_conditions(self,
                             symbol: str,
                             current_price: float) -> Optional[str]:
        """청산 조건 체크
        
        Args:
            symbol: 심볼
            current_price: 현재 가격
            
        Returns:
            청산 사유 또는 None
        """
        
        if symbol not in self.positions:
            return None
        
        position = self.positions[symbol]
        
        # Stop Loss
        if position.stop_loss:
            if position.side == PositionSide.LONG:
                if current_price <= position.stop_loss:
                    return "stop_loss"
            else:
                if current_price >= position.stop_loss:
                    return "stop_loss"
        
        # Take Profit
        if position.take_profit:
            if position.side == PositionSide.LONG:
                if current_price >= position.take_profit:
                    return "take_profit"
            else:
                if current_price <= position.take_profit:
                    return "take_profit"
        
        # Trailing Stop
        if position.trailing_stop and position.trailing_stop > 0:
            position.update_pnl(current_price)
            
            # 최대 이익에서 일정 비율 하락 시
            drawdown = position.max_pnl - position.unrealized_pnl
            
            if position.max_pnl > 0 and drawdown > position.trailing_stop * position.notional:
                return "trailing_stop"
        
        # 시간 제한
        max_holding = self.config.get('max_holding_hours', 0)
        if max_holding > 0:
            holding_time = (datetime.now() - position.entry_time).total_seconds() / 3600
            if holding_time > max_holding:
                return "time_limit"
        
        return None
    
    def adjust_position_size(self,
                           symbol: str,
                           new_size: float,
                           current_price: float) -> bool:
        """포지션 크기 조정
        
        Args:
            symbol: 심볼
            new_size: 새 크기
            current_price: 현재 가격
            
        Returns:
            성공 여부
        """
        
        if symbol not in self.positions:
            return False
        
        position = self.positions[symbol]
        old_size = position.size
        
        if new_size == old_size:
            return True
        
        if new_size == 0:
            # 전체 청산
            self.close_position(symbol, current_price, "size_adjustment")
            return True
        
        # 부분 청산/추가
        size_diff = new_size - old_size
        
        if size_diff < 0:
            # 부분 청산
            closed_size = abs(size_diff)
            
            # 부분 PnL 계산
            if position.side == PositionSide.LONG:
                partial_pnl = (current_price - position.entry_price) * closed_size
            else:
                partial_pnl = (position.entry_price - current_price) * closed_size
            
            position.realized_pnl += partial_pnl
            position.size = new_size
            
            logger.info(f"포지션 부분 청산: {symbol} {closed_size} units, PnL={partial_pnl:.2f}")
            
        else:
            # 포지션 추가
            # 평균 진입가 재계산
            old_notional = position.notional
            new_notional = old_notional + size_diff * current_price
            position.entry_price = new_notional / new_size
            position.size = new_size
            
            logger.info(f"포지션 추가: {symbol} +{size_diff} units @{current_price:.4f}")
        
        return True
    
    def get_position(self, symbol: str) -> Optional[Position]:
        """포지션 조회
        
        Args:
            symbol: 심볼
            
        Returns:
            포지션 또는 None
        """
        
        return self.positions.get(symbol)
    
    def get_all_positions(self) -> Dict[str, Position]:
        """모든 포지션 조회
        
        Returns:
            포지션 딕셔너리
        """
        
        return self.positions.copy()
    
    def get_total_exposure(self) -> float:
        """총 노출 계산
        
        Returns:
            총 노출 금액
        """
        
        return sum(p.notional for p in self.positions.values())
    
    def get_position_stats(self) -> Dict:
        """포지션 통계 조회
        
        Returns:
            통계 딕셔너리
        """
        
        stats = self.stats.copy()
        
        # 현재 포지션 정보
        stats['open_positions'] = len(self.positions)
        stats['total_exposure'] = self.get_total_exposure()
        
        # 포지션별 손익
        position_pnls = {}
        for symbol, position in self.positions.items():
            position_pnls[symbol] = {
                'unrealized_pnl': position.unrealized_pnl,
                'realized_pnl': position.realized_pnl,
                'total_pnl': position.unrealized_pnl + position.realized_pnl
            }
        
        stats['position_pnls'] = position_pnls
        
        # 승률
        if self.position_history:
            winning = sum(1 for p in self.position_history if p.realized_pnl > 0)
            stats['win_rate'] = winning / len(self.position_history)
            
            # 평균 손익
            pnls = [p.realized_pnl for p in self.position_history]
            stats['avg_win'] = np.mean([p for p in pnls if p > 0]) if any(p > 0 for p in pnls) else 0
            stats['avg_loss'] = np.mean([p for p in pnls if p <= 0]) if any(p <= 0 for p in pnls) else 0
        
        return stats
    
    def export_history(self) -> pd.DataFrame:
        """포지션 히스토리 내보내기
        
        Returns:
            히스토리 DataFrame
        """
        
        if not self.position_history:
            return pd.DataFrame()
        
        data = []
        for position in self.position_history:
            data.append({
                'symbol': position.symbol,
                'side': position.side.value,
                'entry_price': position.entry_price,
                'exit_price': position.exit_price,
                'size': position.size,
                'entry_time': position.entry_time,
                'exit_time': position.exit_time,
                'holding_hours': (position.exit_time - position.entry_time).total_seconds() / 3600
                                if position.exit_time else None,
                'realized_pnl': position.realized_pnl,
                'fees': position.fees_paid,
                'funding': position.funding_received,
                'net_pnl': position.realized_pnl + position.funding_received - position.fees_paid,
                'strategy': position.strategy
            })
        
        return pd.DataFrame(data)