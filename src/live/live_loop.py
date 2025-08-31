"""실시간 거래 메인 루프"""

import asyncio
import signal
import sys
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import logging
import traceback
import json
from pathlib import Path

from ..utils.logging import Logger
from ..utils.io import IOUtils
from ..data.feature_engineering import FeatureEngineer
from ..ml.lightgbm_train import LightGBMTrainer
from ..ml.regime_detector import AdaptiveRegimeDetector
from ..ml.threshold_learner import SelfLearningThreshold
from ..strategies.funding_arb import FundingArbitrage
from ..strategies.micro_scalper import MicroScalper
from ..execution.pyramid_entry import PyramidEntryOptimizer
from ..portfolio.rebalancer import DynamicRebalancer
from .ws_stream import WebSocketStream
from .state import TradingState
from .position import PositionManager
from .execution import OrderExecutor
from .risk import RiskManager

logger = Logger.get_logger(__name__)

class LiveTradingLoop:
    """실시간 거래 메인 루프
    
    전체 거래 시스템 조율 및 실행
    """
    
    def __init__(self, config_path: str, dry_run: bool = True):
        """
        Args:
            config_path: 설정 파일 경로
            dry_run: 드라이런 모드
        """
        # 설정 로드
        self.config = IOUtils.load_config(config_path)
        self.dry_run = dry_run
        
        # 심볼 리스트
        self.symbols = self.config.get('symbols', ['BTCUSDT'])
        
        # 컴포넌트 초기화
        self._initialize_components()
        
        # 실행 상태
        self.running = False
        self.tasks = []
        
        # 성능 모니터링
        self.performance_monitor = PerformanceMonitor()
        
        # 시그널 핸들러
        self._setup_signal_handlers()
    
    def _initialize_components(self):
        """컴포넌트 초기화"""
        
        logger.info("컴포넌트 초기화 시작...")
        
        # 거래소 클라이언트
        if self.dry_run:
            from .mock_exchange import MockExchangeClient
            self.exchange = MockExchangeClient(self.config)
        else:
            from .binance_client import BinanceClient
            self.exchange = BinanceClient(self.config)
        
        # 웹소켓 스트림
        self.ws_stream = WebSocketStream(
            symbols=self.symbols,
            testnet=self.config.get('testnet', False)
        )
        
        # 상태 관리
        self.state = TradingState(self.config)
        
        # 포지션 관리
        self.position_manager = PositionManager(self.config)
        
        # 주문 실행
        self.order_executor = OrderExecutor(self.exchange, self.config)
        
        # 리스크 관리
        self.risk_manager = RiskManager(self.config)
        
        # 특징 엔지니어링
        self.feature_engineer = FeatureEngineer(self.config)
        
        # ML 모델
        self._load_models()
        
        # 전략
        self._initialize_strategies()
        
        # 포트폴리오 리밸런서
        self.rebalancer = DynamicRebalancer(self.config.get('portfolio', {}))
        
        logger.info("컴포넌트 초기화 완료")
    
    def _load_models(self):
        """ML 모델 로드"""
        
        # LightGBM 모델
        model_path = Path('models') / 'lightgbm_best.pkl'
        if model_path.exists():
            self.ml_model = LightGBMTrainer.load_model(str(model_path))
            logger.info("ML 모델 로드 완료")
        else:
            self.ml_model = None
            logger.warning("ML 모델 없음")
        
        # 레짐 감지기
        self.regime_detector = AdaptiveRegimeDetector()
        
        # 자기학습 임계값
        self.threshold_learner = SelfLearningThreshold(
            alpha=self.config.get('adaptive', {}).get('alpha', 0.1)
        )
        
        # 임계값 상태 로드
        threshold_state_path = Path('models') / 'threshold_state.json'
        if threshold_state_path.exists():
            self.threshold_learner.load_state(str(threshold_state_path))
    
    def _initialize_strategies(self):
        """전략 초기화"""
        
        # 메인 전략 (TF-PB-OFI-VT)
        self.main_strategy = TrendFollowPullbackStrategy(self.config)
        
        # 펀딩 차익거래
        if self.config.get('funding', {}).get('enabled', False):
            self.funding_strategy = FundingArbitrage(self.config.get('funding', {}))
        else:
            self.funding_strategy = None
        
        # 극초단 스캘핑
        if self.config.get('scalping', {}).get('enabled', False):
            self.scalping_strategy = MicroScalper(self.config.get('scalping', {}))
        else:
            self.scalping_strategy = None
        
        # 피라미드 진입 최적화
        if self.config.get('pyramid', {}).get('enabled', False):
            self.pyramid_optimizer = PyramidEntryOptimizer(self.config.get('pyramid', {}))
        else:
            self.pyramid_optimizer = None
    
    def _setup_signal_handlers(self):
        """시그널 핸들러 설정"""
        
        def signal_handler(signum, frame):
            logger.info(f"시그널 수신: {signum}")
            asyncio.create_task(self.shutdown())
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    async def start(self):
        """거래 시작"""
        
        logger.info("="*60)
        logger.info("실시간 거래 시작")
        logger.info(f"모드: {'DRY RUN' if self.dry_run else 'LIVE'}")
        logger.info(f"심볼: {self.symbols}")
        logger.info("="*60)
        
        self.running = True
        
        try:
            # 초기화
            await self._initialize()
            
            # 태스크 시작
            self.tasks = [
                asyncio.create_task(self._websocket_task()),
                asyncio.create_task(self._trading_task()),
                asyncio.create_task(self._risk_monitoring_task()),
                asyncio.create_task(self._performance_monitoring_task()),
                asyncio.create_task(self._heartbeat_task())
            ]
            
            # 태스크 실행
            await asyncio.gather(*self.tasks)
            
        except Exception as e:
            logger.error(f"거래 루프 에러: {e}")
            logger.error(traceback.format_exc())
        finally:
            await self.shutdown()
    
    async def _initialize(self):
        """초기화"""
        
        # 계정 정보 조회
        account_info = await self.exchange.get_account_info()
        initial_balance = account_info.get('balance', self.config.get('initial_capital', 100000))
        
        logger.info(f"초기 자본: ${initial_balance:,.2f}")
        
        # 리스크 초기화
        self.risk_manager.reset_daily_stats(initial_balance)
        
        # 상태 초기화
        self.state.update_system_status('running', 'System initialized')
        
        # 웹소켓 콜백 등록
        self._register_ws_callbacks()
        
        # 기존 포지션 동기화
        await self._sync_positions()
    
    def _register_ws_callbacks(self):
        """웹소켓 콜백 등록"""
        
        # 체결 데이터
        async def on_trade(trade_data):
            symbol = trade_data['symbol']
            self.state.update_market_data(symbol, {
                'last_trade_price': trade_data['price'],
                'last_trade_qty': trade_data['quantity'],
                'last_trade_time': trade_data['time']
            })
        
        self.ws_stream.register_callback('trade', on_trade)
        
        # 호가 데이터
        async def on_depth(depth_data):
            symbol = depth_data['symbol']
            self.state.update_market_data(symbol, {
                'price': (depth_data['best_bid'] + depth_data['best_ask']) / 2,
                'spread_bps': depth_data['spread_bps'],
                'depth': depth_data
            })
        
        self.ws_stream.register_callback('depth', on_depth)
        
        # K라인 데이터
        async def on_kline(kline_data):
            if kline_data['is_closed']:
                symbol = kline_data['symbol']
                # 특징 업데이트 트리거
                await self._update_features(symbol)
        
        self.ws_stream.register_callback('kline', on_kline)
    
    async def _sync_positions(self):
        """포지션 동기화"""
        
        if self.dry_run:
            return
        
        positions = await self.exchange.get_positions()
        
        for pos in positions:
            symbol = pos['symbol']
            
            if pos['positionAmt'] != 0:
                # 포지션 등록
                self.position_manager.open_position(
                    symbol=symbol,
                    side='long' if pos['positionAmt'] > 0 else 'short',
                    size=abs(pos['positionAmt']),
                    entry_price=pos['entryPrice'],
                    metadata={'synced': True}
                )
                
                logger.info(f"포지션 동기화: {symbol} {pos['positionAmt']}@{pos['entryPrice']}")
    
    async def _websocket_task(self):
        """웹소켓 태스크"""
        
        while self.running:
            try:
                await self.ws_stream.start_stream()
            except Exception as e:
                logger.error(f"웹소켓 에러: {e}")
                await asyncio.sleep(5)
    
    async def _trading_task(self):
        """거래 태스크"""
        
        while self.running:
            try:
                # 거래 가능 체크
                if not self.state.system_state['status'] == 'running':
                    await asyncio.sleep(1)
                    continue
                
                # 각 심볼 처리
                for symbol in self.symbols:
                    await self._process_symbol(symbol)
                
                # 포트폴리오 리밸런싱
                await self._rebalance_portfolio()
                
                # 대기
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"거래 태스크 에러: {e}")
                await asyncio.sleep(5)
    
    async def _process_symbol(self, symbol: str):
        """심볼 처리
        
        Args:
            symbol: 심볼
        """
        
        # 거래 가능 체크
        if not self.state.should_trade(symbol):
            return
        
        # 시장 가격
        current_price = self.state.get_market_price(symbol)
        if not current_price:
            return
        
        # 포지션 체크
        position = self.position_manager.get_position(symbol)
        
        if position:
            # 포지션 관리
            await self._manage_position(symbol, position, current_price)
        else:
            # 신규 진입 체크
            await self._check_entry(symbol, current_price)
    
    async def _manage_position(self, symbol: str, position: Any, current_price: float):
        """포지션 관리
        
        Args:
            symbol: 심볼
            position: 포지션
            current_price: 현재 가격
        """
        
        # 포지션 업데이트
        self.position_manager.update_position(symbol, current_price)
        
        # 청산 조건 체크
        exit_reason = self.position_manager.check_exit_conditions(symbol, current_price)
        
        if exit_reason:
            # 청산 실행
            await self._execute_exit(symbol, position, current_price, exit_reason)
        else:
            # 동적 조정 체크
            await self._check_position_adjustment(symbol, position, current_price)
    
    async def _check_entry(self, symbol: str, current_price: float):
        """진입 체크
        
        Args:
            symbol: 심볼
            current_price: 현재 가격
        """
        
        # 특징 조회
        features = self.state.features.get(symbol)
        if not features:
            return
        
        # 메인 전략 신호
        main_signal = self.main_strategy.generate_signal(symbol, features, current_price)
        
        if main_signal and main_signal.get('action') in ['long', 'short']:
            # 메타 라벨 필터
            if self.ml_model and self.config.get('meta_label', {}).get('use', False):
                meta_prob = self.ml_model.predict_proba([features])[0, 1]
                
                # 적응형 임계값
                thresholds = self.threshold_learner.current_thresholds
                
                if meta_prob < thresholds.get('meta_label', 0.6):
                    logger.debug(f"메타 라벨 필터: {symbol} {meta_prob:.2f} < {thresholds['meta_label']:.2f}")
                    return
                
                main_signal['meta_prob'] = meta_prob
            
            # 리스크 체크
            size = self._calculate_position_size(symbol, main_signal, current_price)
            
            allowed, reason = self.risk_manager.check_pre_trade_risk(
                symbol=symbol,
                side=main_signal['action'],
                size=size,
                price=current_price,
                current_positions=self.position_manager.get_all_positions(),
                account_balance=self.state.get_total_equity()
            )
            
            if not allowed:
                logger.warning(f"리스크 체크 실패: {symbol} - {reason}")
                return
            
            # 진입 실행
            await self._execute_entry(symbol, main_signal, size, current_price)
        
        # 보조 전략 체크
        await self._check_auxiliary_strategies(symbol, current_price, features)
    
    async def _check_auxiliary_strategies(self, symbol: str, current_price: float, features: Dict):
        """보조 전략 체크
        
        Args:
            symbol: 심볼
            current_price: 현재 가격
            features: 특징
        """
        
        # 펀딩 차익거래
        if self.funding_strategy:
            funding_signal = self.funding_strategy.generate_signal(
                funding_rate=features.get('funding_rate', 0),
                funding_ma=features.get('funding_ma', 0),
                funding_std=features.get('funding_std', 0.001),
                momentum=features.get('momentum', 0),
                next_funding_time=datetime.now() + timedelta(hours=8),
                symbol=symbol,
                current_price=current_price
            )
            
            if funding_signal and funding_signal.confidence > 0.7:
                # 펀딩 포지션 실행
                await self._execute_funding_position(symbol, funding_signal, current_price)
        
        # 극초단 스캘핑
        if self.scalping_strategy:
            depth = self.state.market_state['depths'].get(symbol, {})
            recent_trades = self.ws_stream.get_recent_trades(20)
            
            if depth and recent_trades:
                scalp_signal = self.scalping_strategy.find_opportunity(
                    spread_bps=self.state.market_state['spreads'].get(symbol, 1),
                    depth_imbalance=depth.get('imbalance', 0.5),
                    recent_trades=recent_trades,
                    best_bid=depth.get('best_bid', current_price),
                    best_ask=depth.get('best_ask', current_price),
                    bid_depth=depth.get('bid_depth', 0),
                    ask_depth=depth.get('ask_depth', 0),
                    current_time=datetime.now()
                )
                
                if scalp_signal:
                    await self._execute_scalp(symbol, scalp_signal)
    
    async def _execute_entry(self, symbol: str, signal: Dict, size: float, current_price: float):
        """진입 실행
        
        Args:
            symbol: 심볼
            signal: 신호
            size: 크기
            current_price: 현재 가격
        """
        
        logger.info(f"진입 신호: {symbol} {signal['action']} {size:.4f}@{current_price:.2f}")
        
        if self.dry_run:
            # 드라이런: 포지션 기록만
            self.position_manager.open_position(
                symbol=symbol,
                side=signal['action'],
                size=size,
                entry_price=current_price,
                stop_loss=current_price * (1 - signal.get('sl_pct', 0.01)),
                take_profit=current_price * (1 + signal.get('tp_pct', 0.02)),
                strategy='main',
                metadata=signal
            )
            
            self.state.update_system_status('running', f"Dry run entry: {symbol}")
        else:
            # 실제 주문
            if self.pyramid_optimizer and signal.get('use_pyramid', False):
                # 피라미드 진입
                pyramid_plan = self.pyramid_optimizer.optimize_entries(
                    signal_strength=signal.get('confidence', 0.5),
                    market_depth=self.state.market_state['depths'].get(symbol, {}),
                    vwap=features.get('vwap', current_price),
                    atr=features.get('atr', current_price * 0.01),
                    current_price=current_price,
                    total_size=size,
                    side=signal['action']
                )
                
                # 첫 번째 레벨 실행
                await self.pyramid_optimizer.execute_pyramid(pyramid_plan, current_price, datetime.now())
            else:
                # 단일 진입
                result = await self.order_executor.execute_market_order(
                    symbol=symbol,
                    side='buy' if signal['action'] == 'long' else 'sell',
                    quantity=size,
                    urgency=signal.get('urgency', 0.5)
                )
                
                if result['success']:
                    # 포지션 등록
                    self.position_manager.open_position(
                        symbol=symbol,
                        side=signal['action'],
                        size=size,
                        entry_price=current_price,
                        stop_loss=current_price * (1 - signal.get('sl_pct', 0.01)),
                        take_profit=current_price * (1 + signal.get('tp_pct', 0.02)),
                        strategy='main',
                        metadata=signal
                    )
    
    async def _execute_exit(self, symbol: str, position: Any, current_price: float, reason: str):
        """청산 실행
        
        Args:
            symbol: 심볼
            position: 포지션
            current_price: 현재 가격
            reason: 청산 사유
        """
        
        logger.info(f"청산 신호: {symbol} @ {current_price:.2f} ({reason})")
        
        if self.dry_run:
            # 드라이런: 포지션 청산 기록
            result = self.position_manager.close_position(symbol, current_price, reason)
            
            if result:
                # 성과 업데이트
                self.state.update_performance(result)
                
                # 임계값 학습
                if self.threshold_learner:
                    self.threshold_learner.update([result])
        else:
            # 실제 주문
            result = await self.order_executor.execute_market_order(
                symbol=symbol,
                side='sell' if position.side.value == 'long' else 'buy',
                quantity=position.size,
                urgency=0.8
            )
            
            if result['success']:
                # 포지션 청산
                close_result = self.position_manager.close_position(
                    symbol, current_price, reason, 
                    fees=result.get('fees', 0)
                )
                
                if close_result:
                    self.state.update_performance(close_result)
    
    def _calculate_position_size(self, symbol: str, signal: Dict, current_price: float) -> float:
        """포지션 크기 계산
        
        Args:
            symbol: 심볼
            signal: 신호
            current_price: 현재 가격
            
        Returns:
            포지션 크기
        """
        
        # 기본 크기
        account_balance = self.state.get_total_equity()
        base_size = account_balance * self.config.get('position_ratio', 0.1) / current_price
        
        # 변동성 조정
        volatility = self.state.market_state['volatilities'].get(symbol, 0.01)
        target_vol = self.config.get('risk', {}).get('target_vol', 0.25)
        
        if volatility > 0:
            vol_adjusted_size = base_size * (target_vol / volatility)
        else:
            vol_adjusted_size = base_size
        
        # 신호 강도 조정
        confidence = signal.get('confidence', 0.5)
        signal_adjusted_size = vol_adjusted_size * confidence
        
        # 레짐 조정
        if self.state.regime_state['current_regime'] == 'high_vol':
            signal_adjusted_size *= 0.5
        elif self.state.regime_state['current_regime'] == 'trending':
            signal_adjusted_size *= 1.2
        
        # 제한
        max_size = account_balance * self.config.get('max_position_size', 0.3) / current_price
        
        return min(signal_adjusted_size, max_size)
    
    async def _update_features(self, symbol: str):
        """특징 업데이트
        
        Args:
            symbol: 심볼
        """
        
        # 최신 데이터로 특징 계산
        # TODO: 실제 구현
        pass
    
    async def _check_position_adjustment(self, symbol: str, position: Any, current_price: float):
        """포지션 조정 체크
        
        Args:
            symbol: 심볼
            position: 포지션
            current_price: 현재 가격
        """
        
        # 부분 익절/손절
        # TODO: 구현
        pass
    
    async def _rebalance_portfolio(self):
        """포트폴리오 리밸런싱"""
        
        # 리밸런싱 주기 체크
        # TODO: 구현
        pass
    
    async def _risk_monitoring_task(self):
        """리스크 모니터링 태스크"""
        
        while self.running:
            try:
                # 포지션 리스크 업데이트
                positions = self.position_manager.get_all_positions()
                prices = self.state.market_state['prices']
                equity = self.state.get_total_equity()
                
                self.risk_manager.update_position_risk(positions, prices, equity)
                
                # 중단 조건 체크
                stop_reason = self.risk_manager.check_stop_conditions(equity)
                
                if stop_reason:
                    logger.critical(f"거래 중단: {stop_reason}")
                    self.state.update_system_status('stopped', stop_reason)
                    
                    # 모든 포지션 청산
                    for symbol in list(positions.keys()):
                        price = prices.get(symbol)
                        if price:
                            await self._execute_exit(symbol, positions[symbol], price, stop_reason)
                
                # 리스크 상태 업데이트
                self.state.update_risk_state()
                
                await asyncio.sleep(5)
                
            except Exception as e:
                logger.error(f"리스크 모니터링 에러: {e}")
                await asyncio.sleep(10)
    
    async def _performance_monitoring_task(self):
        """성과 모니터링 태스크"""
        
        while self.running:
            try:
                # 성과 메트릭 계산
                metrics = self.state.get_performance_metrics()
                
                # 로깅
                logger.info(f"성과: PnL={metrics['realized_pnl']:.2f}, "
                          f"Win Rate={metrics['win_rate']:.2%}, "
                          f"Equity={self.state.get_total_equity():.2f}")
                
                # 스냅샷 생성
                if datetime.now().minute % 15 == 0:  # 15분마다
                    snapshot = self.state.create_snapshot()
                    
                    # 저장
                    snapshot_path = Path('snapshots') / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                    snapshot_path.parent.mkdir(exist_ok=True)
                    
                    with open(snapshot_path, 'w') as f:
                        json.dump(snapshot, f, indent=2, default=str)
                
                await asyncio.sleep(60)
                
            except Exception as e:
                logger.error(f"성과 모니터링 에러: {e}")
                await asyncio.sleep(60)
    
    async def _heartbeat_task(self):
        """하트비트 태스크"""
        
        while self.running:
            try:
                self.state.update_system_status('running', 'heartbeat')
                
                # 일일 리셋 체크
                if datetime.now().hour == 0 and datetime.now().minute == 0:
                    self.state.reset_daily_stats()
                    self.risk_manager.reset_daily_stats(self.state.get_total_equity())
                    logger.info("일일 통계 리셋")
                
                await asyncio.sleep(30)
                
            except Exception as e:
                logger.error(f"하트비트 에러: {e}")
                await asyncio.sleep(30)
    
    async def shutdown(self):
        """종료"""
        
        logger.info("거래 시스템 종료 중...")
        
        self.running = False
        
        # 태스크 취소
        for task in self.tasks:
            if not task.done():
                task.cancel()
        
        # 웹소켓 종료
        await self.ws_stream.disconnect()
        
        # 상태 저장
        final_snapshot = self.state.create_snapshot()
        
        snapshot_path = Path('snapshots') / f"final_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        snapshot_path.parent.mkdir(exist_ok=True)
        
        with open(snapshot_path, 'w') as f:
            json.dump(final_snapshot, f, indent=2, default=str)
        
        # 임계값 상태 저장
        if self.threshold_learner:
            self.threshold_learner.save_state('models/threshold_state.json')
        
        logger.info("거래 시스템 종료 완료")
        
        # 최종 성과 출력
        metrics = self.state.get_performance_metrics()
        
        print("\n" + "="*60)
        print("최종 성과")
        print("="*60)
        print(f"실현 손익: ${metrics['realized_pnl']:,.2f}")
        print(f"미실현 손익: ${metrics['unrealized_pnl']:,.2f}")
        print(f"총 손익: ${metrics['realized_pnl'] + metrics['unrealized_pnl']:,.2f}")
        print(f"승률: {metrics['win_rate']:.2%}")
        print(f"총 거래: {metrics['total_trades']}")
        print(f"최대 드로우다운: {metrics['max_drawdown']:.2%}")
        print("="*60)


class TrendFollowPullbackStrategy:
    """메인 전략: Trend Follow + Pullback"""
    
    def __init__(self, config: Dict):
        self.config = config
        
    def generate_signal(self, symbol: str, features: Dict, current_price: float) -> Optional[Dict]:
        """신호 생성"""
        
        # TODO: 실제 전략 로직 구현
        # 임시 구현
        
        signal = {
            'symbol': symbol,
            'action': None,
            'confidence': 0,
            'reason': []
        }
        
        # 추세 확인
        if features.get('trend_signal', 0) > 0:
            # 상승 추세
            if features.get('rsi', 50) < 30:
                # 과매도 되돌림
                signal['action'] = 'long'
                signal['confidence'] = 0.7
                signal['reason'].append('bullish_pullback')
        
        elif features.get('trend_signal', 0) < 0:
            # 하락 추세
            if features.get('rsi', 50) > 70:
                # 과매수 되돌림
                signal['action'] = 'short'
                signal['confidence'] = 0.7
                signal['reason'].append('bearish_pullback')
        
        # OFI 게이팅
        if signal['action']:
            ofi_z = features.get('ofi_z', 0)
            
            if signal['action'] == 'long' and ofi_z < 0.2:
                signal['action'] = None
            elif signal['action'] == 'short' and ofi_z > -0.2:
                signal['action'] = None
        
        return signal if signal['action'] else None


class PerformanceMonitor:
    """성능 모니터링"""
    
    def __init__(self):
        self.latencies = deque(maxlen=1000)
        self.feature_times = deque(maxlen=100)
        self.execution_times = deque(maxlen=100)
    
    def record_latency(self, operation: str, latency_ms: float):
        """레이턴시 기록"""
        
        self.latencies.append({
            'operation': operation,
            'latency_ms': latency_ms,
            'timestamp': datetime.now()
        })
        
        if latency_ms > 100:
            logger.warning(f"High latency: {operation} = {latency_ms:.1f}ms")


# CLI 실행용
def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='실시간 거래')
    parser.add_argument('--config', required=True, help='설정 파일')
    parser.add_argument('--dry-run', action='store_true', help='드라이런 모드')
    parser.add_argument('--symbols', nargs='+', help='거래 심볼')
    
    args = parser.parse_args()
    
    # 설정 로드 및 수정
    config = IOUtils.load_config(args.config)
    
    if args.symbols:
        config['symbols'] = args.symbols
    
    # 거래 루프 실행
    loop = LiveTradingLoop(args.config, dry_run=args.dry_run)
    
    # 비동기 실행
    asyncio.run(loop.start())


if __name__ == "__main__":
    main()