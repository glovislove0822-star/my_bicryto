"""주문 실행 관리"""

import asyncio
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from enum import Enum
import logging
import uuid
from collections import deque
import numpy as np

from ..utils.logging import Logger

logger = Logger.get_logger(__name__)

class OrderType(Enum):
    """주문 유형"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"

class OrderStatus(Enum):
    """주문 상태"""
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIAL = "partial"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"

class TimeInForce(Enum):
    """주문 유효 기간"""
    GTC = "GTC"  # Good Till Cancel
    IOC = "IOC"  # Immediate Or Cancel
    FOK = "FOK"  # Fill Or Kill
    GTX = "GTX"  # Good Till Crossing (Post Only)

class Order:
    """주문 클래스"""
    
    def __init__(self,
                 symbol: str,
                 side: str,
                 order_type: OrderType,
                 quantity: float,
                 price: Optional[float] = None,
                 stop_price: Optional[float] = None,
                 time_in_force: TimeInForce = TimeInForce.GTC,
                 reduce_only: bool = False,
                 post_only: bool = False):
        """
        Args:
            symbol: 심볼
            side: 방향 (buy/sell)
            order_type: 주문 유형
            quantity: 수량
            price: 지정가
            stop_price: 스탑 가격
            time_in_force: 유효 기간
            reduce_only: 포지션 축소만
            post_only: 메이커 주문만
        """
        self.order_id = str(uuid.uuid4())
        self.symbol = symbol
        self.side = side.lower()
        self.order_type = order_type
        self.quantity = quantity
        self.price = price
        self.stop_price = stop_price
        self.time_in_force = time_in_force
        self.reduce_only = reduce_only
        self.post_only = post_only
        
        # 상태
        self.status = OrderStatus.PENDING
        self.filled_quantity = 0
        self.avg_fill_price = 0
        self.fees = 0
        
        # 시간
        self.created_time = datetime.now()
        self.submitted_time = None
        self.filled_time = None
        
        # 메타데이터
        self.metadata = {}
        
    @property
    def is_filled(self) -> bool:
        """체결 완료 여부"""
        return self.status == OrderStatus.FILLED
    
    @property
    def is_active(self) -> bool:
        """활성 상태 여부"""
        return self.status in [OrderStatus.PENDING, OrderStatus.SUBMITTED, OrderStatus.PARTIAL]
    
    @property
    def fill_rate(self) -> float:
        """체결률"""
        return self.filled_quantity / self.quantity if self.quantity > 0 else 0

class OrderExecutor:
    """주문 실행 관리자
    
    주문 생성, 제출, 추적 및 실행 최적화
    """
    
    def __init__(self, exchange_client: Any, config: Dict):
        """
        Args:
            exchange_client: 거래소 클라이언트
            config: 실행 설정
        """
        self.exchange = exchange_client
        self.config = config
        
        # 주문 관리
        self.orders = {}  # order_id -> Order
        self.active_orders = {}  # symbol -> List[Order]
        self.order_history = deque(maxlen=1000)
        
        # 실행 설정
        self.execution_params = {
            'use_post_only': config.get('post_only', False),
            'use_iceberg': config.get('use_iceberg', False),
            'iceberg_ratio': config.get('iceberg_ratio', 0.1),
            'retry_attempts': config.get('retry_attempts', 3),
            'retry_delay': config.get('retry_delay', 1),
            'slippage_tolerance': config.get('slippage_tolerance', 0.001),
            'urgency_threshold': config.get('urgency_threshold', 0.7)
        }
        
        # 실행 통계
        self.execution_stats = {
            'total_orders': 0,
            'filled_orders': 0,
            'cancelled_orders': 0,
            'rejected_orders': 0,
            'total_slippage': 0,
            'avg_fill_time': 0,
            'maker_fills': 0,
            'taker_fills': 0
        }
        
        # 스마트 라우팅
        self.smart_routing = SmartOrderRouter(config)
        
        # 실행 큐
        self.execution_queue = asyncio.Queue()
        self.urgent_queue = asyncio.Queue()
    
    async def submit_order(self, order: Order) -> Dict:
        """주문 제출
        
        Args:
            order: 주문 객체
            
        Returns:
            제출 결과
        """
        
        try:
            # 주문 검증
            if not self._validate_order(order):
                order.status = OrderStatus.REJECTED
                return {'success': False, 'reason': 'validation_failed'}
            
            # 스마트 라우팅
            routed_orders = self.smart_routing.route_order(order)
            
            results = []
            for routed_order in routed_orders:
                # 거래소 제출
                result = await self._submit_to_exchange(routed_order)
                results.append(result)
                
                if result['success']:
                    routed_order.status = OrderStatus.SUBMITTED
                    routed_order.submitted_time = datetime.now()
                    
                    # 관리 등록
                    self._register_order(routed_order)
                else:
                    routed_order.status = OrderStatus.REJECTED
                    logger.error(f"주문 제출 실패: {result.get('reason')}")
            
            # 통계 업데이트
            self.execution_stats['total_orders'] += len(routed_orders)
            
            return {
                'success': all(r['success'] for r in results),
                'orders': routed_orders,
                'results': results
            }
            
        except Exception as e:
            logger.error(f"주문 제출 에러: {e}")
            order.status = OrderStatus.REJECTED
            return {'success': False, 'reason': str(e)}
    
    async def _submit_to_exchange(self, order: Order) -> Dict:
        """거래소에 주문 제출
        
        Args:
            order: 주문 객체
            
        Returns:
            제출 결과
        """
        
        # 주문 파라미터 구성
        params = {
            'symbol': order.symbol,
            'side': order.side.upper(),
            'type': order.order_type.value.upper(),
            'quantity': order.quantity,
            'timeInForce': order.time_in_force.value
        }
        
        # 가격 설정
        if order.order_type in [OrderType.LIMIT, OrderType.STOP_LIMIT]:
            params['price'] = order.price
        
        if order.order_type in [OrderType.STOP, OrderType.STOP_LIMIT]:
            params['stopPrice'] = order.stop_price
        
        # 옵션 설정
        if order.reduce_only:
            params['reduceOnly'] = True
        
        if order.post_only and order.time_in_force == TimeInForce.GTX:
            params['postOnly'] = True
        
        # 재시도 로직
        for attempt in range(self.execution_params['retry_attempts']):
            try:
                # API 호출
                response = await self.exchange.create_order(**params)
                
                # 주문 ID 업데이트
                if 'orderId' in response:
                    order.metadata['exchange_order_id'] = response['orderId']
                
                return {
                    'success': True,
                    'order_id': response.get('orderId'),
                    'response': response
                }
                
            except Exception as e:
                logger.warning(f"주문 제출 시도 {attempt + 1} 실패: {e}")
                
                if attempt < self.execution_params['retry_attempts'] - 1:
                    await asyncio.sleep(self.execution_params['retry_delay'])
                else:
                    return {
                        'success': False,
                        'reason': str(e)
                    }
        
        return {'success': False, 'reason': 'max_retries_exceeded'}
    
    async def cancel_order(self, order_id: str) -> bool:
        """주문 취소
        
        Args:
            order_id: 주문 ID
            
        Returns:
            취소 성공 여부
        """
        
        if order_id not in self.orders:
            logger.warning(f"주문 없음: {order_id}")
            return False
        
        order = self.orders[order_id]
        
        if not order.is_active:
            logger.warning(f"비활성 주문: {order_id}")
            return False
        
        try:
            # 거래소 취소 요청
            exchange_order_id = order.metadata.get('exchange_order_id')
            
            if exchange_order_id:
                response = await self.exchange.cancel_order(
                    symbol=order.symbol,
                    orderId=exchange_order_id
                )
                
                if response.get('status') == 'CANCELED':
                    order.status = OrderStatus.CANCELLED
                    self._unregister_order(order)
                    
                    # 통계 업데이트
                    self.execution_stats['cancelled_orders'] += 1
                    
                    logger.info(f"주문 취소: {order_id}")
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"주문 취소 에러: {e}")
            return False
    
    async def modify_order(self,
                          order_id: str,
                          new_price: Optional[float] = None,
                          new_quantity: Optional[float] = None) -> bool:
        """주문 수정
        
        Args:
            order_id: 주문 ID
            new_price: 새 가격
            new_quantity: 새 수량
            
        Returns:
            수정 성공 여부
        """
        
        # 취소 후 재제출 방식
        if await self.cancel_order(order_id):
            order = self.orders.get(order_id)
            
            if order:
                # 새 주문 생성
                new_order = Order(
                    symbol=order.symbol,
                    side=order.side,
                    order_type=order.order_type,
                    quantity=new_quantity or order.quantity,
                    price=new_price or order.price,
                    stop_price=order.stop_price,
                    time_in_force=order.time_in_force,
                    reduce_only=order.reduce_only,
                    post_only=order.post_only
                )
                
                # 메타데이터 복사
                new_order.metadata = order.metadata.copy()
                new_order.metadata['modified_from'] = order_id
                
                # 제출
                result = await self.submit_order(new_order)
                return result['success']
        
        return False
    
    async def update_order_status(self, order_id: str):
        """주문 상태 업데이트
        
        Args:
            order_id: 주문 ID
        """
        
        if order_id not in self.orders:
            return
        
        order = self.orders[order_id]
        exchange_order_id = order.metadata.get('exchange_order_id')
        
        if not exchange_order_id:
            return
        
        try:
            # 거래소에서 상태 조회
            response = await self.exchange.get_order(
                symbol=order.symbol,
                orderId=exchange_order_id
            )
            
            # 상태 업데이트
            status_map = {
                'NEW': OrderStatus.SUBMITTED,
                'PARTIALLY_FILLED': OrderStatus.PARTIAL,
                'FILLED': OrderStatus.FILLED,
                'CANCELED': OrderStatus.CANCELLED,
                'REJECTED': OrderStatus.REJECTED,
                'EXPIRED': OrderStatus.EXPIRED
            }
            
            new_status = status_map.get(response.get('status'))
            
            if new_status:
                order.status = new_status
                
                # 체결 정보 업데이트
                order.filled_quantity = float(response.get('executedQty', 0))
                
                if order.filled_quantity > 0 and float(response.get('cummulativeQuoteQty', 0)) > 0:
                    order.avg_fill_price = float(response['cummulativeQuoteQty']) / order.filled_quantity
                
                # 수수료 업데이트
                # 바이낸스 경우 별도 조회 필요
                
                # 체결 완료 처리
                if order.status == OrderStatus.FILLED:
                    order.filled_time = datetime.now()
                    self._handle_filled_order(order)
                
        except Exception as e:
            logger.error(f"주문 상태 업데이트 에러: {e}")
    
    def _validate_order(self, order: Order) -> bool:
        """주문 검증
        
        Args:
            order: 주문 객체
            
        Returns:
            유효 여부
        """
        
        # 기본 검증
        if order.quantity <= 0:
            logger.error("유효하지 않은 수량")
            return False
        
        if order.order_type in [OrderType.LIMIT, OrderType.STOP_LIMIT]:
            if not order.price or order.price <= 0:
                logger.error("유효하지 않은 가격")
                return False
        
        if order.order_type in [OrderType.STOP, OrderType.STOP_LIMIT]:
            if not order.stop_price or order.stop_price <= 0:
                logger.error("유효하지 않은 스탑 가격")
                return False
        
        # Post-only 검증
        if order.post_only and order.order_type != OrderType.LIMIT:
            logger.error("Post-only는 지정가 주문만 가능")
            return False
        
        return True
    
    def _register_order(self, order: Order):
        """주문 등록
        
        Args:
            order: 주문 객체
        """
        
        self.orders[order.order_id] = order
        
        if order.symbol not in self.active_orders:
            self.active_orders[order.symbol] = []
        
        self.active_orders[order.symbol].append(order)
    
    def _unregister_order(self, order: Order):
        """주문 등록 해제
        
        Args:
            order: 주문 객체
        """
        
        if order.symbol in self.active_orders:
            self.active_orders[order.symbol] = [
                o for o in self.active_orders[order.symbol]
                if o.order_id != order.order_id
            ]
        
        # 히스토리에 추가
        self.order_history.append(order)
    
    def _handle_filled_order(self, order: Order):
        """체결 완료 처리
        
        Args:
            order: 주문 객체
        """
        
        # 슬리피지 계산
        if order.order_type == OrderType.MARKET:
            # 마켓 주문은 예상 가격과 비교
            expected_price = order.metadata.get('expected_price', order.avg_fill_price)
            
            if order.side == 'buy':
                slippage = (order.avg_fill_price - expected_price) / expected_price
            else:
                slippage = (expected_price - order.avg_fill_price) / expected_price
            
            self.execution_stats['total_slippage'] += slippage
        
        # 체결 시간 계산
        if order.submitted_time:
            fill_time = (order.filled_time - order.submitted_time).total_seconds()
            
            n = self.execution_stats['filled_orders']
            self.execution_stats['avg_fill_time'] = (
                (self.execution_stats['avg_fill_time'] * n + fill_time) / (n + 1)
            )
        
        # 메이커/테이커 구분
        if order.post_only or order.order_type == OrderType.LIMIT:
            self.execution_stats['maker_fills'] += 1
        else:
            self.execution_stats['taker_fills'] += 1
        
        # 통계 업데이트
        self.execution_stats['filled_orders'] += 1
        
        # 등록 해제
        self._unregister_order(order)
        
        logger.info(f"주문 체결: {order.order_id} "
                   f"{order.filled_quantity}@{order.avg_fill_price:.4f}")
    
    async def execute_market_order(self,
                                  symbol: str,
                                  side: str,
                                  quantity: float,
                                  urgency: float = 0.5) -> Dict:
        """시장가 주문 실행
        
        Args:
            symbol: 심볼
            side: 방향
            quantity: 수량
            urgency: 긴급도 (0-1)
            
        Returns:
            실행 결과
        """
        
        # 긴급도에 따른 실행 전략
        if urgency > self.execution_params['urgency_threshold']:
            # 즉시 실행
            order = Order(
                symbol=symbol,
                side=side,
                order_type=OrderType.MARKET,
                quantity=quantity,
                time_in_force=TimeInForce.IOC
            )
        else:
            # 분할 실행 고려
            if quantity > self._get_optimal_order_size(symbol):
                return await self.execute_twap(symbol, side, quantity, duration=60)
            
            order = Order(
                symbol=symbol,
                side=side,
                order_type=OrderType.MARKET,
                quantity=quantity
            )
        
        return await self.submit_order(order)
    
    async def execute_limit_order(self,
                                 symbol: str,
                                 side: str,
                                 quantity: float,
                                 price: float,
                                 post_only: bool = None) -> Dict:
        """지정가 주문 실행
        
        Args:
            symbol: 심볼
            side: 방향
            quantity: 수량
            price: 가격
            post_only: Post-only 여부
            
        Returns:
            실행 결과
        """
        
        if post_only is None:
            post_only = self.execution_params['use_post_only']
        
        order = Order(
            symbol=symbol,
            side=side,
            order_type=OrderType.LIMIT,
            quantity=quantity,
            price=price,
            time_in_force=TimeInForce.GTX if post_only else TimeInForce.GTC,
            post_only=post_only
        )
        
        return await self.submit_order(order)
    
    async def execute_twap(self,
                          symbol: str,
                          side: str,
                          total_quantity: float,
                          duration: int,
                          interval: int = 10) -> Dict:
        """TWAP 실행
        
        Args:
            symbol: 심볼
            side: 방향
            total_quantity: 총 수량
            duration: 실행 기간 (초)
            interval: 실행 간격 (초)
            
        Returns:
            실행 결과
        """
        
        n_slices = duration // interval
        slice_quantity = total_quantity / n_slices
        
        results = []
        
        for i in range(n_slices):
            # 슬라이스 실행
            order = Order(
                symbol=symbol,
                side=side,
                order_type=OrderType.MARKET,
                quantity=slice_quantity
            )
            
            result = await self.submit_order(order)
            results.append(result)
            
            # 대기
            if i < n_slices - 1:
                await asyncio.sleep(interval)
        
        return {
            'success': all(r['success'] for r in results),
            'total_executed': sum(
                r['orders'][0].filled_quantity 
                for r in results 
                if r['success'] and r['orders']
            ),
            'results': results
        }
    
    def _get_optimal_order_size(self, symbol: str) -> float:
        """최적 주문 크기 계산
        
        Args:
            symbol: 심볼
            
        Returns:
            최적 크기
        """
        
        # TODO: 시장 심도 기반 계산
        return 1000  # 임시 값
    
    def get_active_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """활성 주문 조회
        
        Args:
            symbol: 심볼 (선택적)
            
        Returns:
            활성 주문 리스트
        """
        
        if symbol:
            return self.active_orders.get(symbol, [])
        
        all_orders = []
        for orders in self.active_orders.values():
            all_orders.extend(orders)
        
        return all_orders
    
    def get_execution_stats(self) -> Dict:
        """실행 통계 조회
        
        Returns:
            실행 통계
        """
        
        stats = self.execution_stats.copy()
        
        # 추가 계산
        if stats['filled_orders'] > 0:
            stats['fill_rate'] = stats['filled_orders'] / stats['total_orders']
            stats['avg_slippage_bps'] = stats['total_slippage'] / stats['filled_orders'] * 10000
            stats['maker_ratio'] = stats['maker_fills'] / stats['filled_orders']
        
        return stats


class SmartOrderRouter:
    """스마트 주문 라우터
    
    주문을 최적으로 분할하고 라우팅
    """
    
    def __init__(self, config: Dict):
        """
        Args:
            config: 라우팅 설정
        """
        self.config = config
        
        self.routing_params = {
            'use_iceberg': config.get('use_iceberg', False),
            'iceberg_ratio': config.get('iceberg_ratio', 0.1),
            'max_order_size': config.get('max_order_size', 10000),
            'min_order_size': config.get('min_order_size', 10)
        }
    
    def route_order(self, order: Order) -> List[Order]:
        """주문 라우팅
        
        Args:
            order: 원본 주문
            
        Returns:
            라우팅된 주문 리스트
        """
        
        # 아이스버그 주문
        if self.routing_params['use_iceberg'] and order.quantity > self.routing_params['max_order_size']:
            return self._create_iceberg_orders(order)
        
        # 단일 주문
        return [order]
    
    def _create_iceberg_orders(self, order: Order) -> List[Order]:
        """아이스버그 주문 생성
        
        Args:
            order: 원본 주문
            
        Returns:
            분할된 주문 리스트
        """
        
        visible_quantity = order.quantity * self.routing_params['iceberg_ratio']
        visible_quantity = max(visible_quantity, self.routing_params['min_order_size'])
        visible_quantity = min(visible_quantity, self.routing_params['max_order_size'])
        
        orders = []
        remaining = order.quantity
        
        while remaining > 0:
            slice_quantity = min(visible_quantity, remaining)
            
            slice_order = Order(
                symbol=order.symbol,
                side=order.side,
                order_type=order.order_type,
                quantity=slice_quantity,
                price=order.price,
                stop_price=order.stop_price,
                time_in_force=order.time_in_force,
                reduce_only=order.reduce_only,
                post_only=order.post_only
            )
            
            slice_order.metadata['iceberg'] = True
            slice_order.metadata['parent_order'] = order.order_id
            
            orders.append(slice_order)
            remaining -= slice_quantity
        
        return orders