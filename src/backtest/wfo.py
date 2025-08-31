"""Walk Forward Optimization (WFO) 모듈"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import logging
from tqdm import tqdm
import json
import optuna
from concurrent.futures import ProcessPoolExecutor, as_completed

from ..utils.logging import Logger
from ..utils.io import IOUtils
from .simulator import BacktestSimulator
from .metrics import PerformanceMetrics

logger = Logger.get_logger(__name__)

class WalkForwardOptimizer:
    """Walk Forward 최적화
    
    롤링 윈도우 방식으로 파라미터 최적화 및 검증
    """
    
    def __init__(self,
                 config: Dict,
                 db_path: str = "data/trading.db"):
        """
        Args:
            config: WFO 설정
            db_path: 데이터베이스 경로
        """
        self.config = config
        self.db_path = Path(db_path)
        
        # WFO 파라미터
        self.wfo_params = {
            'train_days': config.get('wfo', {}).get('train_days', 60),
            'test_days': config.get('wfo', {}).get('test_days', 15),
            'step_days': config.get('wfo', {}).get('step_days', 15),
            'regime_specific': config.get('wfo', {}).get('regime_specific', True),
            'min_trades': config.get('optuna', {}).get('min_trades', 100),
            'min_sharpe': config.get('optuna', {}).get('accept_gate', {}).get('min_wfo_sharpe', 0.8)
        }
        
        # Optuna 설정
        self.optuna_params = config.get('optuna', {})
        
        # 결과 저장
        self.wfo_results = []
        self.best_params_history = []
        self.performance_history = []
    
    def run(self,
           strategy_class: Any,
           symbols: List[str],
           start_date: str,
           end_date: str,
           timeframe: str = '3m',
           n_jobs: int = 1) -> Dict:
        """WFO 실행
        
        Args:
            strategy_class: 전략 클래스
            symbols: 심볼 리스트
            start_date: 시작 날짜
            end_date: 종료 날짜
            timeframe: 시간프레임
            n_jobs: 병렬 작업 수
            
        Returns:
            WFO 결과
        """
        
        logger.info("Walk Forward Optimization 시작")
        logger.info(f"기간: {start_date} ~ {end_date}")
        logger.info(f"학습: {self.wfo_params['train_days']}일, "
                   f"테스트: {self.wfo_params['test_days']}일, "
                   f"스텝: {self.wfo_params['step_days']}일")
        
        # WFO 윈도우 생성
        windows = self._generate_windows(start_date, end_date)
        
        logger.info(f"총 {len(windows)} 윈도우 생성")
        
        # 각 윈도우 처리
        if n_jobs > 1:
            # 병렬 처리
            results = self._parallel_optimization(
                windows, strategy_class, symbols, timeframe, n_jobs
            )
        else:
            # 순차 처리
            results = []
            for window in tqdm(windows, desc="WFO Windows"):
                result = self._optimize_window(
                    window, strategy_class, symbols, timeframe
                )
                results.append(result)
        
        # 결과 분석
        final_results = self._analyze_results(results)
        
        # 결과 저장
        self._save_results(final_results)
        
        return final_results
    
    def _generate_windows(self, start_date: str, end_date: str) -> List[Dict]:
        """WFO 윈도우 생성"""
        
        windows = []
        
        current_start = pd.to_datetime(start_date)
        final_end = pd.to_datetime(end_date)
        
        window_id = 0
        
        while current_start + timedelta(days=self.wfo_params['train_days']) < final_end:
            # 학습 기간
            train_start = current_start
            train_end = train_start + timedelta(days=self.wfo_params['train_days'])
            
            # 테스트 기간
            test_start = train_end
            test_end = test_start + timedelta(days=self.wfo_params['test_days'])
            
            if test_end > final_end:
                test_end = final_end
            
            windows.append({
                'window_id': window_id,
                'train_start': train_start.strftime('%Y-%m-%d'),
                'train_end': train_end.strftime('%Y-%m-%d'),
                'test_start': test_start.strftime('%Y-%m-%d'),
                'test_end': test_end.strftime('%Y-%m-%d')
            })
            
            # 다음 윈도우
            current_start += timedelta(days=self.wfo_params['step_days'])
            window_id += 1
        
        return windows
    
    def _optimize_window(self,
                        window: Dict,
                        strategy_class: Any,
                        symbols: List[str],
                        timeframe: str) -> Dict:
        """단일 윈도우 최적화"""
        
        logger.info(f"\n윈도우 {window['window_id']} 최적화 시작")
        logger.info(f"학습: {window['train_start']} ~ {window['train_end']}")
        logger.info(f"테스트: {window['test_start']} ~ {window['test_end']}")
        
        # 1. 학습 기간 최적화
        best_params = self._optimize_parameters(
            strategy_class,
            symbols,
            window['train_start'],
            window['train_end'],
            timeframe
        )
        
        if not best_params:
            logger.warning(f"윈도우 {window['window_id']} 최적화 실패")
            return {
                'window': window,
                'best_params': None,
                'train_metrics': None,
                'test_metrics': None,
                'status': 'failed'
            }
        
        # 2. 학습 기간 성과 평가
        train_metrics = self._evaluate_parameters(
            best_params,
            strategy_class,
            symbols,
            window['train_start'],
            window['train_end'],
            timeframe
        )
        
        # 3. 테스트 기간 성과 평가
        test_metrics = self._evaluate_parameters(
            best_params,
            strategy_class,
            symbols,
            window['test_start'],
            window['test_end'],
            timeframe
        )
        
        # 4. 결과 검증
        status = self._validate_window_result(train_metrics, test_metrics)
        
        result = {
            'window': window,
            'best_params': best_params,
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'status': status
        }
        
        logger.info(f"윈도우 {window['window_id']} 완료: {status}")
        if train_metrics and test_metrics:
            logger.info(f"  학습 Sharpe: {train_metrics.get('sharpe_ratio', 0):.2f}")
            logger.info(f"  테스트 Sharpe: {test_metrics.get('sharpe_ratio', 0):.2f}")
        
        return result
    
    def _optimize_parameters(self,
                            strategy_class: Any,
                            symbols: List[str],
                            start_date: str,
                            end_date: str,
                            timeframe: str) -> Optional[Dict]:
        """파라미터 최적화 (Optuna)"""
        
        def objective(trial):
            # 파라미터 샘플링
            params = self._sample_parameters(trial)
            
            # 백테스트 실행
            config = self.config.copy()
            config.update(params)
            
            simulator = BacktestSimulator(config, self.db_path)
            strategy = strategy_class(config)
            
            try:
                results = simulator.run(
                    strategy=strategy,
                    start_date=start_date,
                    end_date=end_date,
                    symbols=symbols,
                    timeframe=timeframe
                )
                
                # 목적 함수 계산
                if 'metrics' in results:
                    score = self._calculate_objective(results['metrics'])
                else:
                    score = -1e10
                    
            except Exception as e:
                logger.error(f"백테스트 에러: {e}")
                score = -1e10
            
            return score
        
        # Optuna 스터디 생성
        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=10)
        )
        
        # 최적화 실행
        study.optimize(
            objective,
            n_trials=self.optuna_params.get('trials', 100),
            show_progress_bar=False
        )
        
        # 최적 파라미터
        if study.best_trial:
            best_params = study.best_params
            best_params['_score'] = study.best_value
            return best_params
        
        return None
    
    def _sample_parameters(self, trial: optuna.Trial) -> Dict:
        """파라미터 샘플링"""
        
        params = {}
        
        # 추세 파라미터
        params['donchian_n'] = trial.suggest_int('donchian_n', 20, 120)
        params['mom_k'] = trial.suggest_int('mom_k', 30, 180)
        params['ema_fast'] = trial.suggest_int('ema_fast', 5, 30)
        params['ema_slow'] = trial.suggest_int('ema_slow', 20, 120)
        
        # 엔트리 파라미터
        params['rsi_len'] = trial.suggest_int('rsi_len', 2, 6)
        params['rsi_buy'] = trial.suggest_int('rsi_buy', 5, 30)
        params['rsi_sell'] = trial.suggest_int('rsi_sell', 70, 95)
        params['vwap_z_entry'] = trial.suggest_float('vwap_z_entry', 0.2, 1.5)
        
        # 게이팅 파라미터
        params['ofi_z_th_long'] = trial.suggest_float('ofi_z_th_long', 0.0, 1.0)
        params['ofi_z_th_short'] = trial.suggest_float('ofi_z_th_short', -1.0, 0.0)
        params['spread_bp_max'] = trial.suggest_float('spread_bp_max', 0.5, 3.0)
        params['depth_min'] = trial.suggest_int('depth_min', 1000, 20000, step=1000)
        
        # 리스크 파라미터
        params['target_vol'] = trial.suggest_float('target_vol', 0.15, 0.50)
        params['tp_atr'] = trial.suggest_float('tp_atr', 0.8, 2.0)
        params['sl_atr'] = trial.suggest_float('sl_atr', 0.4, 1.2)
        params['trail_atr'] = trial.suggest_float('trail_atr', 0.0, 1.5)
        params['tmax_bars'] = trial.suggest_int('tmax_bars', 20, 80)
        
        # 메타라벨 파라미터
        params['p_threshold'] = trial.suggest_float('p_threshold', 0.50, 0.75)
        
        # 펀딩 파라미터 (v2.0)
        if self.config.get('funding', {}).get('enabled', False):
            params['funding_z_threshold'] = trial.suggest_float('funding_z_threshold', 1.5, 3.0)
            params['funding_harvest_min'] = trial.suggest_float('funding_harvest_min', 0.005, 0.02)
        
        return params
    
    def _calculate_objective(self, metrics: Dict) -> float:
        """목적 함수 계산"""
        
        # 기본 메트릭
        sharpe = metrics.get('sharpe_ratio', 0)
        mdd = metrics.get('max_drawdown', 0)
        turnover = metrics.get('turnover', 0)
        n_trades = metrics.get('total_trades', 0)
        
        # 제약 조건 체크
        if n_trades < self.optuna_params.get('min_trades', 100):
            return -1e10
        
        if abs(mdd) > self.optuna_params.get('mdd_cap', 0.35):
            return -1e10
        
        if turnover > self.optuna_params.get('turnover_cap', 25):
            return -1e10
        
        # 목적 함수
        lambda_mdd = self.optuna_params.get('lambda_mdd', 0.25)
        mu_turnover = self.optuna_params.get('mu_turnover', 0.02)
        
        score = sharpe - lambda_mdd * abs(mdd) - mu_turnover * turnover
        
        # 펀딩 수익 보너스 (v2.0)
        if 'funding_income' in metrics:
            nu_funding = self.optuna_params.get('nu_funding', 0.1)
            score += nu_funding * metrics['funding_income']
        
        return score
    
    def _evaluate_parameters(self,
                            params: Dict,
                            strategy_class: Any,
                            symbols: List[str],
                            start_date: str,
                            end_date: str,
                            timeframe: str) -> Optional[Dict]:
        """파라미터 평가"""
        
        if not params:
            return None
        
        # 설정 업데이트
        config = self.config.copy()
        config.update(params)
        
        # 백테스트 실행
        simulator = BacktestSimulator(config, self.db_path)
        strategy = strategy_class(config)
        
        try:
            results = simulator.run(
                strategy=strategy,
                start_date=start_date,
                end_date=end_date,
                symbols=symbols,
                timeframe=timeframe
            )
            
            if 'metrics' in results:
                return results['metrics']
                
        except Exception as e:
            logger.error(f"평가 에러: {e}")
        
        return None
    
    def _validate_window_result(self,
                               train_metrics: Optional[Dict],
                               test_metrics: Optional[Dict]) -> str:
        """윈도우 결과 검증"""
        
        if not train_metrics or not test_metrics:
            return 'failed'
        
        # 최소 성과 체크
        min_sharpe = self.optuna_params.get('accept_gate', {}).get('min_wfo_sharpe', 0.8)
        max_mdd = self.optuna_params.get('accept_gate', {}).get('max_wfo_mdd', 0.3)
        
        if test_metrics.get('sharpe_ratio', 0) < min_sharpe:
            return 'rejected_sharpe'
        
        if abs(test_metrics.get('max_drawdown', 0)) > max_mdd:
            return 'rejected_mdd'
        
        # 과적합 체크
        train_sharpe = train_metrics.get('sharpe_ratio', 0)
        test_sharpe = test_metrics.get('sharpe_ratio', 0)
        
        if train_sharpe > 0 and test_sharpe < train_sharpe * 0.3:
            return 'overfitted'
        
        return 'accepted'
    
    def _parallel_optimization(self,
                              windows: List[Dict],
                              strategy_class: Any,
                              symbols: List[str],
                              timeframe: str,
                              n_jobs: int) -> List[Dict]:
        """병렬 최적화"""
        
        results = []
        
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            futures = {
                executor.submit(
                    self._optimize_window,
                    window,
                    strategy_class,
                    symbols,
                    timeframe
                ): window
                for window in windows
            }
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="WFO Windows"):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"윈도우 처리 에러: {e}")
                    window = futures[future]
                    results.append({
                        'window': window,
                        'status': 'error',
                        'error': str(e)
                    })
        
        # 윈도우 ID로 정렬
        results.sort(key=lambda x: x['window']['window_id'])
        
        return results
    
    def _analyze_results(self, results: List[Dict]) -> Dict:
        """WFO 결과 분석"""
        
        # 통계 계산
        total_windows = len(results)
        accepted_windows = sum(1 for r in results if r.get('status') == 'accepted')
        
        # 테스트 기간 성과 집계
        test_sharpes = []
        test_returns = []
        test_mdds = []
        
        for result in results:
            if result.get('status') == 'accepted' and result.get('test_metrics'):
                test_sharpes.append(result['test_metrics'].get('sharpe_ratio', 0))
                test_returns.append(result['test_metrics'].get('total_return', 0))
                test_mdds.append(result['test_metrics'].get('max_drawdown', 0))
        
        # 안정적인 파라미터 찾기
        stable_params = self._find_stable_parameters(results)
        
        # 최종 결과
        final_results = {
            'total_windows': total_windows,
            'accepted_windows': accepted_windows,
            'acceptance_rate': accepted_windows / total_windows if total_windows > 0 else 0,
            'avg_test_sharpe': np.mean(test_sharpes) if test_sharpes else 0,
            'std_test_sharpe': np.std(test_sharpes) if test_sharpes else 0,
            'avg_test_return': np.mean(test_returns) if test_returns else 0,
            'avg_test_mdd': np.mean(test_mdds) if test_mdds else 0,
            'stable_parameters': stable_params,
            'window_results': results
        }
        
        # 요약 출력
        self._print_summary(final_results)
        
        return final_results
    
    def _find_stable_parameters(self, results: List[Dict]) -> Dict:
        """안정적인 파라미터 찾기"""
        
        # 수락된 윈도우의 파라미터 수집
        accepted_params = []
        
        for result in results:
            if result.get('status') == 'accepted' and result.get('best_params'):
                accepted_params.append(result['best_params'])
        
        if not accepted_params:
            return {}
        
        # 파라미터별 통계
        stable_params = {}
        
        param_df = pd.DataFrame(accepted_params)
        
        for col in param_df.columns:
            if col.startswith('_'):
                continue
            
            values = param_df[col]
            
            # 수치형 파라미터
            if pd.api.types.is_numeric_dtype(values):
                stable_params[col] = {
                    'mean': values.mean(),
                    'median': values.median(),
                    'std': values.std(),
                    'min': values.min(),
                    'max': values.max(),
                    'recommended': values.median()  # 중앙값 사용
                }
        
        return stable_params
    
    def _save_results(self, results: Dict):
        """결과 저장"""
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 결과 디렉토리
        results_dir = Path('results') / 'wfo' / timestamp
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # 전체 결과 저장
        with open(results_dir / 'wfo_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # 안정적 파라미터 저장
        if results.get('stable_parameters'):
            stable_params = {
                param: values['recommended']
                for param, values in results['stable_parameters'].items()
                if 'recommended' in values
            }
            
            with open(results_dir / 'stable_parameters.json', 'w') as f:
                json.dump(stable_params, f, indent=2)
        
        # 윈도우별 결과 CSV
        window_data = []
        for result in results.get('window_results', []):
            row = {
                'window_id': result['window']['window_id'],
                'train_start': result['window']['train_start'],
                'train_end': result['window']['train_end'],
                'test_start': result['window']['test_start'],
                'test_end': result['window']['test_end'],
                'status': result.get('status', 'unknown')
            }
            
            if result.get('train_metrics'):
                row['train_sharpe'] = result['train_metrics'].get('sharpe_ratio', 0)
                row['train_return'] = result['train_metrics'].get('total_return', 0)
            
            if result.get('test_metrics'):
                row['test_sharpe'] = result['test_metrics'].get('sharpe_ratio', 0)
                row['test_return'] = result['test_metrics'].get('total_return', 0)
                row['test_mdd'] = result['test_metrics'].get('max_drawdown', 0)
            
            window_data.append(row)
        
        pd.DataFrame(window_data).to_csv(results_dir / 'window_results.csv', index=False)
        
        logger.info(f"WFO 결과 저장: {results_dir}")
    
    def _print_summary(self, results: Dict):
        """결과 요약 출력"""
        
        print("\n" + "="*60)
        print("Walk Forward Optimization 결과")
        print("="*60)
        
        print(f"\n총 윈도우: {results['total_windows']}")
        print(f"수락 윈도우: {results['accepted_windows']} ({results['acceptance_rate']:.1%})")
        
        print(f"\n평균 테스트 Sharpe: {results['avg_test_sharpe']:.2f} ± {results['std_test_sharpe']:.2f}")
        print(f"평균 테스트 수익률: {results['avg_test_return']:.2%}")
        print(f"평균 테스트 MDD: {results['avg_test_mdd']:.2%}")
        
        if results.get('stable_parameters'):
            print("\n안정적 파라미터 (권장값):")
            for param, values in results['stable_parameters'].items():
                if 'recommended' in values:
                    print(f"  {param}: {values['recommended']:.4f}")
        
        print("="*60)

# CLI 실행용
def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Walk Forward Optimization')
    parser.add_argument('--config', required=True, help='설정 파일')
    parser.add_argument('--strategy', required=True, help='전략 클래스')
    parser.add_argument('--symbols', nargs='+', required=True, help='심볼 리스트')
    parser.add_argument('--start', required=True, help='시작 날짜')
    parser.add_argument('--end', required=True, help='종료 날짜')
    parser.add_argument('--timeframe', default='3m', help='시간프레임')
    parser.add_argument('--jobs', type=int, default=1, help='병렬 작업 수')
    parser.add_argument('--db', default='data/trading.db', help='DB 경로')
    
    args = parser.parse_args()
    
    # 설정 로드
    config = IOUtils.load_config(args.config)
    
    # WFO 실행
    wfo = WalkForwardOptimizer(config, db_path=args.db)
    
    # 전략 로드
    module_name, class_name = args.strategy.rsplit('.', 1)
    module = __import__(module_name, fromlist=[class_name])
    strategy_class = getattr(module, class_name)
    
    # 최적화 실행
    results = wfo.run(
        strategy_class=strategy_class,
        symbols=args.symbols,
        start_date=args.start,
        end_date=args.end,
        timeframe=args.timeframe,
        n_jobs=args.jobs
    )

if __name__ == "__main__":
    main()