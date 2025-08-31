"""Optuna 목적 함수"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Callable
import logging
import traceback
from datetime import datetime

from ..utils.logging import Logger
from ..backtest.simulator import BacktestSimulator
from ..backtest.metrics import PerformanceMetrics

logger = Logger.get_logger(__name__)

class ObjectiveFunction:
    """Optuna 최적화 목적 함수
    
    백테스트를 실행하고 성과 메트릭을 기반으로 목적 값 계산
    """
    
    def __init__(self,
                 strategy_class: Any,
                 symbols: list,
                 start_date: str,
                 end_date: str,
                 config: Dict,
                 db_path: str = "data/trading.db"):
        """
        Args:
            strategy_class: 전략 클래스
            symbols: 심볼 리스트
            start_date: 시작 날짜
            end_date: 종료 날짜
            config: 기본 설정
            db_path: 데이터베이스 경로
        """
        self.strategy_class = strategy_class
        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date
        self.base_config = config
        self.db_path = db_path
        
        # 목적 함수 파라미터
        self.objective_params = config.get('optuna', {})
        
        # 가중치
        self.weights = {
            'sharpe': 1.0,
            'mdd': self.objective_params.get('lambda_mdd', 0.25),
            'turnover': self.objective_params.get('mu_turnover', 0.02),
            'funding': self.objective_params.get('nu_funding', 0.1),
            'stability': self.objective_params.get('xi_stability', 0.05),
            'consistency': self.objective_params.get('psi_consistency', 0.05)
        }
        
        # 제약 조건
        self.constraints = {
            'min_trades': self.objective_params.get('min_trades', 100),
            'min_sharpe': self.objective_params.get('min_sharpe', 0),
            'max_mdd': self.objective_params.get('mdd_cap', 0.35),
            'max_turnover': self.objective_params.get('turnover_cap', 25),
            'min_win_rate': self.objective_params.get('min_win_rate', 0.4),
            'skew_floor': self.objective_params.get('skew_floor', -0.5)
        }
        
        # 페널티
        self.penalties = {
            'neg_sharpe': self.objective_params.get('penalty_neg_sharpe', 1.0),
            'constraint_violation': self.objective_params.get('penalty_mid', 0.5),
            'hard_constraint': self.objective_params.get('penalty_big', 2.0)
        }
        
        # 메트릭 계산기
        self.metrics_calculator = PerformanceMetrics()
        
        # 캐시 (동일 파라미터 재계산 방지)
        self.cache = {}
    
    def __call__(self, trial) -> float:
        """목적 함수 실행
        
        Args:
            trial: Optuna trial 객체
            
        Returns:
            목적 값 (최대화)
        """
        
        try:
            # 파라미터 캐시 키
            cache_key = str(sorted(trial.params.items()))
            
            if cache_key in self.cache:
                logger.debug(f"Trial {trial.number}: 캐시 히트")
                return self.cache[cache_key]
            
            # 백테스트 실행
            metrics = self._run_backtest(trial)
            
            if metrics is None:
                score = -1e10
            else:
                # 목적 값 계산
                score = self._calculate_objective(metrics, trial)
            
            # 캐시 저장
            self.cache[cache_key] = score
            
            # Trial 정보 저장
            self._save_trial_info(trial, metrics, score)
            
            logger.info(f"Trial {trial.number}: Score = {score:.4f}")
            
            return score
            
        except Exception as e:
            logger.error(f"Trial {trial.number} 에러: {e}")
            logger.error(traceback.format_exc())
            return -1e10
    
    def _run_backtest(self, trial) -> Optional[Dict]:
        """백테스트 실행
        
        Args:
            trial: Optuna trial 객체
            
        Returns:
            성과 메트릭 또는 None
        """
        
        # 파라미터를 설정으로 변환
        config = self._params_to_config(trial.params)
        
        # 시뮬레이터 생성
        simulator = BacktestSimulator(config, self.db_path)
        
        # 전략 인스턴스 생성
        strategy = self.strategy_class(config)
        
        try:
            # 백테스트 실행
            results = simulator.run(
                strategy=strategy,
                start_date=self.start_date,
                end_date=self.end_date,
                symbols=self.symbols,
                timeframe=config.get('timeframe', '3m')
            )
            
            if 'metrics' in results:
                return results['metrics']
            else:
                return None
                
        except Exception as e:
            logger.error(f"백테스트 실행 에러: {e}")
            return None
    
    def _params_to_config(self, params: Dict) -> Dict:
        """Optuna 파라미터를 설정으로 변환
        
        Args:
            params: Optuna 파라미터
            
        Returns:
            설정 딕셔너리
        """
        
        config = self.base_config.copy()
        
        # 파라미터를 카테고리별로 정리
        for param_name, value in params.items():
            if '_' in param_name:
                category = param_name.split('_')[0]
                param = '_'.join(param_name.split('_')[1:])
                
                if category not in config:
                    config[category] = {}
                
                config[category][param] = value
            else:
                config[param_name] = value
        
        return config
    
    def _calculate_objective(self, metrics: Dict, trial) -> float:
        """목적 값 계산
        
        Args:
            metrics: 성과 메트릭
            trial: Optuna trial 객체
            
        Returns:
            목적 값
        """
        
        # 기본 메트릭 추출
        sharpe = metrics.get('sharpe_ratio', 0)
        mdd = abs(metrics.get('max_drawdown', 0))
        turnover = metrics.get('turnover', 0)
        n_trades = metrics.get('total_trades', 0)
        win_rate = metrics.get('win_rate', 0)
        skewness = metrics.get('skewness', 0)
        stability = metrics.get('stability', 0)
        
        # 제약 조건 체크
        constraint_violations = []
        
        if n_trades < self.constraints['min_trades']:
            constraint_violations.append('min_trades')
            
        if sharpe < self.constraints['min_sharpe']:
            constraint_violations.append('min_sharpe')
            
        if mdd > self.constraints['max_mdd']:
            constraint_violations.append('max_mdd')
            
        if turnover > self.constraints['max_turnover']:
            constraint_violations.append('max_turnover')
            
        if win_rate < self.constraints['min_win_rate']:
            constraint_violations.append('min_win_rate')
            
        if skewness < self.constraints['skew_floor']:
            constraint_violations.append('skew_floor')
        
        # 하드 제약 위반 시 즉시 반환
        if 'min_trades' in constraint_violations or n_trades == 0:
            return -1e10
        
        # 기본 목적 값
        base_score = sharpe
        
        # MDD 페널티
        mdd_penalty = self.weights['mdd'] * mdd
        
        # 회전율 페널티
        turnover_penalty = self.weights['turnover'] * turnover
        
        # 펀딩 수익 보너스 (v2.0)
        funding_income = metrics.get('funding_income', 0)
        funding_bonus = self.weights['funding'] * funding_income
        
        # 안정성 보너스
        stability_bonus = self.weights['stability'] * stability
        
        # 일관성 보너스 (연속 승/패 고려)
        max_consecutive_losses = metrics.get('max_consecutive_losses', 0)
        consistency_penalty = self.weights['consistency'] * max_consecutive_losses / 10
        
        # 종합 점수
        score = (
            base_score
            - mdd_penalty
            - turnover_penalty
            + funding_bonus
            + stability_bonus
            - consistency_penalty
        )
        
        # 제약 위반 페널티
        for violation in constraint_violations:
            if violation in ['max_mdd', 'max_turnover']:
                score -= self.penalties['hard_constraint']
            else:
                score -= self.penalties['constraint_violation']
        
        # Sharpe가 음수면 추가 페널티
        if sharpe < 0:
            score -= self.penalties['neg_sharpe']
        
        # 레짐 일관성 체크 (v2.0)
        if trial.params.get('regime_enabled', False):
            regime_consistency = metrics.get('regime_consistency', 0)
            min_consistency = self.objective_params.get('regime_consistency_min', 0.6)
            
            if regime_consistency < min_consistency:
                score -= self.penalties['constraint_violation']
        
        # Trial 사용자 속성 저장
        trial.set_user_attr('sharpe', sharpe)
        trial.set_user_attr('mdd', mdd)
        trial.set_user_attr('turnover', turnover)
        trial.set_user_attr('n_trades', n_trades)
        trial.set_user_attr('win_rate', win_rate)
        trial.set_user_attr('violations', constraint_violations)
        
        return score
    
    def _save_trial_info(self, trial, metrics: Optional[Dict], score: float):
        """Trial 정보 저장
        
        Args:
            trial: Optuna trial 객체
            metrics: 성과 메트릭
            score: 목적 값
        """
        
        # 기본 정보
        trial.set_user_attr('score', score)
        trial.set_user_attr('timestamp', datetime.now().isoformat())
        
        if metrics:
            # 주요 메트릭 저장
            key_metrics = [
                'total_return', 'annual_return', 'volatility',
                'sharpe_ratio', 'sortino_ratio', 'calmar_ratio',
                'max_drawdown', 'win_rate', 'profit_factor',
                'avg_holding_time', 'turnover', 'total_trades'
            ]
            
            for metric in key_metrics:
                if metric in metrics:
                    trial.set_user_attr(metric, metrics[metric])
            
            # v2.0 추가 메트릭
            if 'funding_income' in metrics:
                trial.set_user_attr('funding_income', metrics['funding_income'])
            
            if 'scalping_pnl' in metrics:
                trial.set_user_attr('scalping_pnl', metrics['scalping_pnl'])
            
            if 'regime_consistency' in metrics:
                trial.set_user_attr('regime_consistency', metrics['regime_consistency'])
    
    def get_feature_importance(self, study) -> Dict:
        """특징 중요도 계산
        
        Args:
            study: 완료된 Optuna study
            
        Returns:
            파라미터별 중요도
        """
        
        if len(study.trials) < 10:
            logger.warning("충분한 trials이 없어 중요도 계산 불가")
            return {}
        
        try:
            # Optuna 내장 중요도 계산
            importance = optuna.importance.get_param_importances(study)
            
            # 상위 파라미터 로깅
            sorted_importance = sorted(
                importance.items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            logger.info("파라미터 중요도 (상위 10개):")
            for param, score in sorted_importance[:10]:
                logger.info(f"  {param}: {score:.4f}")
            
            return importance
            
        except Exception as e:
            logger.error(f"중요도 계산 에러: {e}")
            return {}
    
    def analyze_convergence(self, study) -> Dict:
        """수렴 분석
        
        Args:
            study: 진행 중인 Optuna study
            
        Returns:
            수렴 분석 결과
        """
        
        if len(study.trials) < 20:
            return {'converged': False, 'reason': 'insufficient_trials'}
        
        # 최근 N개 trials의 최고 값
        n_recent = min(20, len(study.trials) // 4)
        recent_best_values = []
        
        for i in range(len(study.trials) - n_recent, len(study.trials)):
            best_so_far = max(
                t.value for t in study.trials[:i+1]
                if t.value is not None
            )
            recent_best_values.append(best_so_far)
        
        if not recent_best_values:
            return {'converged': False, 'reason': 'no_valid_trials'}
        
        # 개선율 계산
        improvement_rate = (recent_best_values[-1] - recent_best_values[0]) / \
                          (abs(recent_best_values[0]) + 1e-10)
        
        # 표준편차
        std_recent = np.std(recent_best_values)
        
        # 수렴 판단
        converged = improvement_rate < 0.01 and std_recent < 0.05
        
        return {
            'converged': converged,
            'improvement_rate': improvement_rate,
            'std_recent': std_recent,
            'best_value': study.best_value,
            'n_trials': len(study.trials),
            'recent_best_values': recent_best_values
        }