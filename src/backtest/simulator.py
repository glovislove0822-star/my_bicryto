"""백테스트 시뮬레이터"""

import pandas as pd
import numpy as np
import duckdb
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import logging
from tqdm import tqdm
import json

from ..utils.logging import Logger
from ..utils.io import IOUtils
from .cost_model import CostModel
from .metrics import PerformanceMetrics

logger = Logger.get_logger(__name__)

class BacktestSimulator:
    """백테스트 시뮬레이션 엔진
    
    바 단위 시뮬레이션, 정교한 비용 모델, 리스크 관리 포함
    """
    
    def __init__(self, 
                 config: Dict,
                 db_path: str = "data/trading.db"):
        """
        Args:
            config: 백테스트 설정
            db_path: 데이터베이스 경로
        """
        self.config = config
        self.db_path = Path(db_path)
        self.conn = duckdb.connect(str(self.db_path))
        
        # 비용 모델
        self.cost_model = CostModel(config.get('fees', {}))
        
        # 성과 측정
        self.metrics = PerformanceMetrics()
        
        # 포지션 상태
        self.positions = {}  # symbol -> position dict
        self.trades = []
        self.equity_curve = []
        
        # 리스크 관리
        self.risk_params = config.get('risk', {})
        self.daily_pnl = 0
        self.daily_trades = 0
        self.circuit_breaker_triggered = False
        
        # 초기 자본
        self.initial_capital = config.get('initial_capital', 100000)
        self.current_capital = self.initial_capital
    
    def run(self,
           strategy: Any,
           start_date: str,
           end_date: str,
           symbols: List[str],
           timeframe: str = '3m') -> Dict:
        """백테스트 실행
        
        Args:
            strategy: 전략 객체
            start_date: 시작 날짜
            end_date: 종료 날짜
            symbols: 심볼 리스트
            timeframe: 시간프레임
            
        Returns:
            백테스트 결과
        """
        
        logger.info(f"백테스트 시작: {start_date} ~ {end_date}")
        logger.info(f"심볼: {symbols}")
        logger.info(f"초기 자본: ${self.initial_capital:,.0f}")
        
        # 데이터 로드
        data = self._load_data(symbols, timeframe, start_date, end_date)
        
        if data.empty:
            logger.error("데이터가 없습니다")
            return {}
        
        # 시뮬레이션 루프
        for timestamp in tqdm(data.index.unique(), desc="백테스트"):
            # 현재 바 데이터
            current_bar = data.loc[timestamp]
            
            # 일일 리셋 체크
            self._check_daily_reset(timestamp)
            
            # 서킷 브레이커 체크
            if self.circuit_breaker_triggered:
                continue
            
            # 각 심볼 처리
            for symbol in symbols:
                if symbol not in current_bar.index:
                    continue
                
                bar_data = current_bar.loc[symbol]
                
                # 1. 기존 포지션 업데이트
                self._update_positions(symbol, bar_data, timestamp)
                
                # 2. 전략 신호 생성
                signal = strategy.generate_signal(symbol, bar_data, timestamp)
                
                # 3. 리스크 체크
                if signal and self._check_risk_limits(signal, bar_data):
                    # 4. 주문 실행
                    self._execute_signal(signal, bar_data, timestamp)
            
            # 5. 자산 곡선 업데이트
            self._update_equity_curve(timestamp)
        
        # 남은 포지션 청산
        self._close_all_positions(data.index[-1])
        
        # 결과 계산
        results = self._calculate_results()
        
        # 결과 저장
        self._save_results(results)
        
        return results
    
    def _load_data(self,
                  symbols: List[str],
                  timeframe: str,
                  start_date: str,
                  end_date: str) -> pd.DataFrame:
        """데이터 로드"""
        
        symbols_str = "','".join(symbols)
        
        query = f"""
            SELECT 
                f.*,
                b.open,
                b.high,
                b.low,
                b.close,
                b.volume,
                l.label,
                l.meta_label,
                l.pnl_pct as label_pnl
            FROM trading.features_{timeframe} f
            JOIN trading.bars_{timeframe} b
                ON f.symbol = b.symbol AND f.ts = b.open_time
            LEFT JOIN trading.labels_{timeframe} l
                ON f.symbol = l.symbol AND f.ts = l.ts
            WHERE f.symbol IN ('{symbols_str}')
                AND f.ts >= '{start_date}'
                AND f.ts <= '{end_date}'
            ORDER BY f.ts, f.symbol
        """
        
        df = self.conn.execute(query).df()
        
        if not df.empty:
            df['ts'] = pd.to_datetime(df['ts'])
            df.set_index(['ts', 'symbol'], inplace=True)
        
        logger.info(f"데이터 로드 완료: {len(df)} rows")
        
        return df
    
    def _update_positions(self, symbol: str, bar_data: pd.Series, timestamp: datetime):
        """포지션 업데이트"""
        
        if symbol not in self.positions:
            return
        
        position = self.positions[symbol]
        current_price = bar_data['close']
        
        # PnL 계산
        if position['side'] == 'long':
            pnl = (current_price - position['entry_price']) / position['entry_price']
        else:
            pnl = (position['entry_price'] - current_price) / position['entry_price']
        
        position['current_pnl'] = pnl
        position['current_price'] = current_price
        
        # Exit 조건 체크
        should_exit = False
        exit_reason = ''
        
        # 1. Stop Loss
        if pnl <= -position['sl_pct']:
            should_exit = True
            exit_reason = 'stop_loss'
        
        # 2. Take Profit
        elif pnl >= position['tp_pct']:
            should_exit = True
            exit_reason = 'take_profit'
        
        # 3. Time Exit
        elif (timestamp - position['entry_time']).total_seconds() / 60 >= position.get('max_bars', 40) * 3:
            should_exit = True
            exit_reason = 'time_exit'
        
        # 4. Trailing Stop
        if position.get('trail_pct', 0) > 0:
            position['max_pnl'] = max(position.get('max_pnl', 0), pnl)
            
            if pnl < position['max_pnl'] - position['trail_pct']:
                should_exit = True
                exit_reason = 'trailing_stop'
        
        # Exit 실행
        if should_exit:
            self._close_position(symbol, current_price, timestamp, exit_reason)
    
    def _execute_signal(self, signal: Dict, bar_data: pd.Series, timestamp: datetime):
        """신호 실행"""
        
        symbol = signal['symbol']
        
        # 이미 포지션이 있으면 스킵
        if symbol in self.positions:
            return
        
        # 포지션 크기 계산
        position_size = self._calculate_position_size(
            signal, bar_data, self.current_capital
        )
        
        if position_size <= 0:
            return
        
        # 진입 가격 (슬리피지 포함)
        entry_price = bar_data['close']
        
        # 비용 계산
        costs = self.cost_model.calculate_trade_cost(
            symbol=symbol,
            side=signal['side'],
            price=entry_price,
            quantity=position_size,
            is_maker=False,  # 테이커 가정
            spread=bar_data.get('spread_bps', 1) / 10000,
            depth=bar_data.get('depth_total', 10000),
            funding_rate=bar_data.get('funding_rate', 0),
            holding_hours=0,
            volatility=bar_data.get('realized_vol', 0.01)
        )
        
        # 조정된 진입 가격
        if signal['side'] == 'long':
            adjusted_entry = entry_price * (1 + costs['total_bps'] / 10000)
        else:
            adjusted_entry = entry_price * (1 - costs['total_bps'] / 10000)
        
        # 포지션 생성
        position = {
            'symbol': symbol,
            'side': signal['side'],
            'entry_price': adjusted_entry,
            'entry_time': timestamp,
            'size': position_size,
            'notional': position_size * adjusted_entry,
            'tp_pct': signal.get('tp_pct', self.risk_params.get('tp_atr', 1.2) * bar_data.get('atr', 0.01)),
            'sl_pct': signal.get('sl_pct', self.risk_params.get('sl_atr', 0.6) * bar_data.get('atr', 0.01)),
            'trail_pct': signal.get('trail_pct', self.risk_params.get('trail_atr', 0)),
            'max_bars': signal.get('max_bars', self.risk_params.get('tmax_bars', 40)),
            'entry_costs': costs,
            'current_pnl': 0,
            'max_pnl': 0
        }
        
        self.positions[symbol] = position
        
        # 자본 차감
        self.current_capital -= costs['total']
        
        logger.debug(f"포지션 진입: {symbol} {signal['side']} @ {adjusted_entry:.4f}")
    
    def _close_position(self, 
                       symbol: str, 
                       exit_price: float,
                       timestamp: datetime,
                       reason: str):
        """포지션 청산"""
        
        if symbol not in self.positions:
            return
        
        position = self.positions[symbol]
        
        # 비용 계산
        costs = self.cost_model.calculate_trade_cost(
            symbol=symbol,
            side='sell' if position['side'] == 'long' else 'buy',
            price=exit_price,
            quantity=position['size'],
            is_maker=False,
            spread=0.0001,
            depth=10000,
            funding_rate=0,
            holding_hours=(timestamp - position['entry_time']).total_seconds() / 3600,
            volatility=0.01
        )
        
        # 조정된 청산 가격
        if position['side'] == 'long':
            adjusted_exit = exit_price * (1 - costs['total_bps'] / 10000)
        else:
            adjusted_exit = exit_price * (1 + costs['total_bps'] / 10000)
        
        # PnL 계산
        if position['side'] == 'long':
            gross_pnl = (adjusted_exit - position['entry_price']) * position['size']
        else:
            gross_pnl = (position['entry_price'] - adjusted_exit) * position['size']
        
        net_pnl = gross_pnl - position['entry_costs']['total'] - costs['total']
        
        # 거래 기록
        trade = {
            'symbol': symbol,
            'side': position['side'],
            'entry_time': position['entry_time'],
            'entry_price': position['entry_price'],
            'exit_time': timestamp,
            'exit_price': adjusted_exit,
            'size': position['size'],
            'gross_pnl': gross_pnl,
            'net_pnl': net_pnl,
            'return_pct': net_pnl / position['notional'],
            'entry_costs': position['entry_costs']['total'],
            'exit_costs': costs['total'],
            'total_costs': position['entry_costs']['total'] + costs['total'],
            'reason': reason,
            'holding_time': (timestamp - position['entry_time']).total_seconds() / 3600
        }
        
        self.trades.append(trade)
        
        # 자본 업데이트
        self.current_capital += net_pnl
        self.daily_pnl += net_pnl
        self.daily_trades += 1
        
        # 포지션 제거
        del self.positions[symbol]
        
        logger.debug(f"포지션 청산: {symbol} @ {adjusted_exit:.4f}, PnL: {net_pnl:.2f} ({reason})")
    
    def _calculate_position_size(self,
                                signal: Dict,
                                bar_data: pd.Series,
                                capital: float) -> float:
        """포지션 크기 계산"""
        
        # 변동성 타겟팅
        target_vol = self.risk_params.get('target_vol', 0.25)
        realized_vol = bar_data.get('realized_vol', 0.01)
        
        if realized_vol > 0:
            vol_adjustment = target_vol / realized_vol
        else:
            vol_adjustment = 1.0
        
        # 기본 크기
        base_size = capital * self.config.get('position_ratio', 0.1)
        
        # 신호 강도 조정
        signal_strength = signal.get('confidence', 0.5)
        
        # 최종 크기
        position_size = base_size * vol_adjustment * signal_strength
        
        # 제한
        max_size = capital * self.config.get('max_position_ratio', 0.3)
        min_size = capital * 0.01
        
        position_size = np.clip(position_size, min_size, max_size)
        
        # 레버리지 체크
        max_leverage = self.config.get('max_leverage', 3)
        total_exposure = sum(p['notional'] for p in self.positions.values()) + position_size
        
        if total_exposure > capital * max_leverage:
            position_size = max(0, capital * max_leverage - sum(p['notional'] for p in self.positions.values()))
        
        return position_size / bar_data['close']  # 수량으로 변환
    
    def _check_risk_limits(self, signal: Dict, bar_data: pd.Series) -> bool:
        """리스크 한도 체크"""
        
        # 1. 일일 손실 한도
        daily_loss_limit = self.initial_capital * self.risk_params.get('daily_stop_pct', 0.02)
        
        if self.daily_pnl < -daily_loss_limit:
            logger.warning(f"일일 손실 한도 도달: {self.daily_pnl:.2f}")
            self.circuit_breaker_triggered = True
            return False
        
        # 2. 일일 거래 수 제한
        max_daily_trades = self.config.get('max_daily_trades', 50)
        
        if self.daily_trades >= max_daily_trades:
            return False
        
        # 3. 포지션 수 제한
        max_positions = self.config.get('max_positions', 10)
        
        if len(self.positions) >= max_positions:
            return False
        
        # 4. 상관관계 체크
        if signal['symbol'] in self.positions:
            return False
        
        # 5. 최소 자본 체크
        min_capital = self.initial_capital * 0.2
        
        if self.current_capital < min_capital:
            logger.warning(f"최소 자본 미달: {self.current_capital:.2f}")
            return False
        
        return True
    
    def _check_daily_reset(self, timestamp: datetime):
        """일일 리셋 체크"""
        
        if hasattr(self, 'last_date'):
            if timestamp.date() != self.last_date:
                # 새로운 날
                self.daily_pnl = 0
                self.daily_trades = 0
                self.circuit_breaker_triggered = False
                self.last_date = timestamp.date()
        else:
            self.last_date = timestamp.date()
    
    def _update_equity_curve(self, timestamp: datetime):
        """자산 곡선 업데이트"""
        
        # 미실현 손익
        unrealized_pnl = sum(
            p['current_pnl'] * p['notional'] 
            for p in self.positions.values()
        )
        
        # 총 자산
        total_equity = self.current_capital + unrealized_pnl
        
        self.equity_curve.append({
            'timestamp': timestamp,
            'equity': total_equity,
            'cash': self.current_capital,
            'unrealized_pnl': unrealized_pnl,
            'n_positions': len(self.positions),
            'daily_pnl': self.daily_pnl
        })
    
    def _close_all_positions(self, timestamp: datetime):
        """모든 포지션 청산"""
        
        for symbol in list(self.positions.keys()):
            position = self.positions[symbol]
            exit_price = position['current_price']
            self._close_position(symbol, exit_price, timestamp, 'end_of_backtest')
    
    def _calculate_results(self) -> Dict:
        """백테스트 결과 계산"""
        
        if not self.trades:
            return {
                'error': 'No trades executed'
            }
        
        # 거래 데이터프레임
        trades_df = pd.DataFrame(self.trades)
        
        # 자산 곡선 데이터프레임
        equity_df = pd.DataFrame(self.equity_curve)
        equity_df.set_index('timestamp', inplace=True)
        
        # 성과 메트릭 계산
        metrics = self.metrics.calculate_all_metrics(
            trades_df,
            equity_df,
            self.initial_capital
        )
        
        # 비용 요약
        cost_summary = self.cost_model.get_cost_summary()
        
        return {
            'metrics': metrics,
            'trades': trades_df.to_dict('records'),
            'equity_curve': equity_df.to_dict('records'),
            'cost_summary': cost_summary,
            'config': self.config
        }
    
    def _save_results(self, results: Dict):
        """결과 저장"""
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 결과 디렉토리
        results_dir = Path('results') / timestamp
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # 메트릭 저장
        with open(results_dir / 'metrics.json', 'w') as f:
            json.dump(results['metrics'], f, indent=2, default=str)
        
        # 거래 내역 저장
        if results.get('trades'):
            pd.DataFrame(results['trades']).to_csv(
                results_dir / 'trades.csv', index=False
            )
        
        # 자산 곡선 저장
        if results.get('equity_curve'):
            pd.DataFrame(results['equity_curve']).to_csv(
                results_dir / 'equity_curve.csv', index=False
            )
        
        logger.info(f"결과 저장: {results_dir}")

# CLI 실행용
def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='백테스트 시뮬레이터')
    parser.add_argument('--config', required=True, help='설정 파일')
    parser.add_argument('--strategy', required=True, help='전략 클래스')
    parser.add_argument('--symbols', nargs='+', required=True, help='심볼 리스트')
    parser.add_argument('--start', required=True, help='시작 날짜')
    parser.add_argument('--end', required=True, help='종료 날짜')
    parser.add_argument('--timeframe', default='3m', help='시간프레임')
    parser.add_argument('--db', default='data/trading.db', help='DB 경로')
    
    args = parser.parse_args()
    
    # 설정 로드
    config = IOUtils.load_config(args.config)
    
    # 시뮬레이터 생성
    simulator = BacktestSimulator(config, db_path=args.db)
    
    # 전략 로드 (동적 임포트)
    module_name, class_name = args.strategy.rsplit('.', 1)
    module = __import__(module_name, fromlist=[class_name])
    strategy_class = getattr(module, class_name)
    strategy = strategy_class(config)
    
    # 백테스트 실행
    results = simulator.run(
        strategy=strategy,
        start_date=args.start,
        end_date=args.end,
        symbols=args.symbols,
        timeframe=args.timeframe
    )
    
    # 결과 출력
    if 'metrics' in results:
        print("\n=== 백테스트 결과 ===")
        for key, value in results['metrics'].items():
            if isinstance(value, float):
                print(f"{key}: {value:.4f}")
            else:
                print(f"{key}: {value}")

if __name__ == "__main__":
    main()