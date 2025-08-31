"""Optuna 최적화 실행기"""

import optuna
from optuna.visualization import plot_optimization_history, plot_param_importances
from optuna.visualization import plot_parallel_coordinate, plot_slice
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging
import json
import pickle
import joblib
from concurrent.futures import ProcessPoolExecutor
import warnings
warnings.filterwarnings('ignore')

from ..utils.logging import Logger
from ..utils.io import IOUtils
from .search_space import SearchSpace
from .objective import ObjectiveFunction

logger = Logger.get_logger(__name__)

class OptunaRunner:
    """Optuna 최적화 실행 및 관리
    
    Study 생성, 실행, 분석, 저장 등 전체 최적화 프로세스 관리
    """
    
    def __init__(self,
                 config: Dict,
                 strategy_class: Any,
                 symbols: List[str],
                 start_date: str,
                 end_date: str,
                 db_path: str = "data/trading.db",
                 study_name: Optional[str] = None):
        """
        Args:
            config: 최적화 설정
            strategy_class: 전략 클래스
            symbols: 심볼 리스트
            start_date: 백테스트 시작 날짜
            end_date: 백테스트 종료 날짜
            db_path: 데이터베이스 경로
            study_name: Study 이름
        """
        self.config = config
        self.strategy_class = strategy_class
        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date
        self.db_path = db_path
        
        # Study 이름
        if study_name:
            self.study_name = study_name
        else:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.study_name = f"study_{timestamp}"
        
        # 검색 공간
        self.search_space = SearchSpace(config.get('search_space', {}))
        
        # 목적 함수
        self.objective = ObjectiveFunction(
            strategy_class=strategy_class,
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            config=config,
            db_path=db_path
        )
        
        # Optuna 설정
        self.optuna_config = config.get('optuna', {})
        
        # Study
        self.study = None
        
        # 결과 저장 경로
        self.results_dir = Path('results') / 'optuna' / self.study_name
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # 실행 히스토리
        self.execution_history = []
    
    def create_study(self,
                    storage: Optional[str] = None,
                    load_if_exists: bool = False) -> optuna.Study:
        """Study 생성
        
        Args:
            storage: 저장소 URL (SQLite, PostgreSQL 등)
            load_if_exists: 기존 Study 로드 여부
            
        Returns:
            Optuna Study 객체
        """
        
        # 샘플러 생성
        sampler_type = self.optuna_config.get('sampler', 'tpe')
        
        if sampler_type == 'tpe':
            sampler = optuna.samplers.TPESampler(
                n_startup_trials=self.optuna_config.get('n_startup_trials', 10),
                n_ei_candidates=self.optuna_config.get('n_ei_candidates', 24),
                seed=42
            )
        elif sampler_type == 'cmaes':
            sampler = optuna.samplers.CmaEsSampler(
                n_startup_trials=self.optuna_config.get('n_startup_trials', 10),
                seed=42
            )
        elif sampler_type == 'random':
            sampler = optuna.samplers.RandomSampler(seed=42)
        elif sampler_type == 'grid':
            # 그리드 서치용 파라미터 그리드 필요
            search_space_grid = self._create_grid_search_space()
            sampler = optuna.samplers.GridSampler(search_space_grid)
        else:
            sampler = optuna.samplers.TPESampler(seed=42)
        
        # 프루너 생성
        pruner_type = self.optuna_config.get('pruner', 'median')
        
        if pruner_type == 'median':
            pruner = optuna.pruners.MedianPruner(
                n_startup_trials=5,
                n_warmup_steps=10,
                interval_steps=1
            )
        elif pruner_type == 'hyperband':
            pruner = optuna.pruners.HyperbandPruner(
                min_resource=1,
                max_resource=100,
                reduction_factor=3
            )
        elif pruner_type == 'successive_halving':
            pruner = optuna.pruners.SuccessiveHalvingPruner(
                min_resource=1,
                min_early_stopping_rate=0
            )
        elif pruner_type == 'none':
            pruner = optuna.pruners.NopPruner()
        else:
            pruner = optuna.pruners.MedianPruner()
        
        # Study 생성
        self.study = optuna.create_study(
            study_name=self.study_name,
            direction='maximize',
            sampler=sampler,
            pruner=pruner,
            storage=storage,
            load_if_exists=load_if_exists
        )
        
        logger.info(f"Study 생성: {self.study_name}")
        logger.info(f"Sampler: {sampler_type}, Pruner: {pruner_type}")
        
        return self.study
    
    def optimize(self,
                n_trials: Optional[int] = None,
                timeout: Optional[float] = None,
                n_jobs: int = 1,
                show_progress_bar: bool = True,
                callbacks: Optional[List] = None) -> optuna.Study:
        """최적화 실행
        
        Args:
            n_trials: Trial 수
            timeout: 타임아웃 (초)
            n_jobs: 병렬 작업 수
            show_progress_bar: 진행 바 표시
            callbacks: 콜백 함수 리스트
            
        Returns:
            완료된 Study
        """
        
        if self.study is None:
            self.create_study()
        
        # 기본 trial 수
        if n_trials is None:
            n_trials = self.optuna_config.get('trials', 100)
        
        logger.info(f"최적화 시작: {n_trials} trials")
        
        # 콜백 설정
        if callbacks is None:
            callbacks = []
        
        # 수렴 체크 콜백
        callbacks.append(self._convergence_callback)
        
        # 중간 저장 콜백
        callbacks.append(self._checkpoint_callback)
        
        # 최적화 실행
        try:
            # 파라미터 샘플링 래퍼
            def wrapped_objective(trial):
                # 파라미터 샘플링
                params = self.search_space.sample_parameters(trial)
                
                # Trial에 파라미터 설정
                for param_name, value in params.items():
                    trial.set_user_attr(f'param_{param_name}', value)
                
                # 목적 함수 실행
                return self.objective(trial)
            
            self.study.optimize(
                wrapped_objective,
                n_trials=n_trials,
                timeout=timeout,
                n_jobs=n_jobs,
                show_progress_bar=show_progress_bar,
                callbacks=callbacks,
                gc_after_trial=True
            )
            
            logger.info(f"최적화 완료: {len(self.study.trials)} trials")
            logger.info(f"최고 점수: {self.study.best_value:.4f}")
            
        except KeyboardInterrupt:
            logger.info("최적화 중단됨")
        except Exception as e:
            logger.error(f"최적화 에러: {e}")
            raise
        
        # 결과 분석 및 저장
        self._analyze_results()
        self._save_results()
        
        return self.study
    
    def _convergence_callback(self, study: optuna.Study, trial: optuna.FrozenTrial):
        """수렴 체크 콜백"""
        
        # 매 N trials마다 수렴 체크
        check_interval = 20
        
        if len(study.trials) % check_interval == 0:
            convergence = self.objective.analyze_convergence(study)
            
            if convergence.get('converged', False):
                logger.info(f"수렴 감지: Trial {len(study.trials)}")
                logger.info(f"개선율: {convergence['improvement_rate']:.4f}")
                
                # 조기 종료 옵션
                if self.optuna_config.get('early_stopping', False):
                    study.stop()
    
    def _checkpoint_callback(self, study: optuna.Study, trial: optuna.FrozenTrial):
        """중간 저장 콜백"""
        
        # 매 N trials마다 저장
        save_interval = 10
        
        if len(study.trials) % save_interval == 0:
            checkpoint_path = self.results_dir / f'checkpoint_trial_{len(study.trials)}.pkl'
            
            with open(checkpoint_path, 'wb') as f:
                pickle.dump(study, f)
            
            logger.debug(f"체크포인트 저장: {checkpoint_path}")
    
    def _create_grid_search_space(self) -> Dict:
        """그리드 서치용 검색 공간 생성"""
        
        grid_space = {}
        
        # 간단한 그리드 (주요 파라미터만)
        grid_space['trend_donchian_n'] = [40, 60, 80]
        grid_space['risk_target_vol'] = [0.2, 0.3, 0.4]
        grid_space['risk_tp_atr'] = [1.0, 1.5, 2.0]
        grid_space['risk_sl_atr'] = [0.5, 0.75, 1.0]
        grid_space['meta_label_p_threshold'] = [0.5, 0.6, 0.7]
        
        return grid_space
    
    def _analyze_results(self):
        """결과 분석"""
        
        if not self.study or len(self.study.trials) == 0:
            logger.warning("분석할 trials이 없습니다")
            return
        
        logger.info("\n" + "="*60)
        logger.info("최적화 결과 분석")
        logger.info("="*60)
        
        # 1. 기본 통계
        completed_trials = [t for t in self.study.trials if t.value is not None]
        
        logger.info(f"\n총 Trials: {len(self.study.trials)}")
        logger.info(f"완료 Trials: {len(completed_trials)}")
        logger.info(f"실패 Trials: {len(self.study.trials) - len(completed_trials)}")
        
        # 2. 최고 성능
        logger.info(f"\n최고 점수: {self.study.best_value:.4f}")
        logger.info(f"최고 Trial: #{self.study.best_trial.number}")
        
        # 최고 파라미터
        logger.info("\n최고 파라미터:")
        for param, value in self.study.best_params.items():
            if isinstance(value, float):
                logger.info(f"  {param}: {value:.4f}")
            else:
                logger.info(f"  {param}: {value}")
        
        # 3. 최고 Trial 메트릭
        logger.info("\n최고 Trial 메트릭:")
        for attr, value in self.study.best_trial.user_attrs.items():
            if not attr.startswith('param_'):
                if isinstance(value, float):
                    logger.info(f"  {attr}: {value:.4f}")
                else:
                    logger.info(f"  {attr}: {value}")
        
        # 4. 파라미터 중요도
        importance = self.objective.get_feature_importance(self.study)
        
        # 5. 상위 5개 Trials
        top_trials = sorted(
            completed_trials,
            key=lambda t: t.value,
            reverse=True
        )[:5]
        
        logger.info("\n상위 5개 Trials:")
        for i, trial in enumerate(top_trials, 1):
            logger.info(f"  {i}. Trial #{trial.number}: {trial.value:.4f}")
            
            # 주요 메트릭
            sharpe = trial.user_attrs.get('sharpe_ratio', 0)
            mdd = trial.user_attrs.get('max_drawdown', 0)
            win_rate = trial.user_attrs.get('win_rate', 0)
            
            logger.info(f"     Sharpe: {sharpe:.2f}, MDD: {mdd:.2%}, WR: {win_rate:.2%}")
        
        logger.info("="*60)
    
    def _save_results(self):
        """결과 저장"""
        
        if not self.study:
            return
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 1. Study 객체 저장
        study_path = self.results_dir / f'study_{timestamp}.pkl'
        with open(study_path, 'wb') as f:
            pickle.dump(self.study, f)
        
        logger.info(f"Study 저장: {study_path}")
        
        # 2. 최고 파라미터 저장
        best_params = self.search_space.export_best_params(self.study, top_n=5)
        
        best_params_path = self.results_dir / f'best_params_{timestamp}.json'
        with open(best_params_path, 'w') as f:
            json.dump(best_params, f, indent=2, default=str)
        
        # 3. Trials DataFrame 저장
        trials_df = self._create_trials_dataframe()
        
        if not trials_df.empty:
            trials_df.to_csv(
                self.results_dir / f'trials_{timestamp}.csv',
                index=False
            )
        
        # 4. 시각화 저장
        self._save_visualizations()
        
        logger.info(f"결과 저장 완료: {self.results_dir}")
    
    def _create_trials_dataframe(self) -> pd.DataFrame:
        """Trials를 DataFrame으로 변환"""
        
        if not self.study or len(self.study.trials) == 0:
            return pd.DataFrame()
        
        data = []
        
        for trial in self.study.trials:
            row = {
                'trial_number': trial.number,
                'value': trial.value,
                'state': trial.state.name,
                'datetime_start': trial.datetime_start,
                'datetime_complete': trial.datetime_complete
            }
            
            # 파라미터 추가
            for param, value in trial.params.items():
                row[f'param_{param}'] = value
            
            # 사용자 속성 추가
            for attr, value in trial.user_attrs.items():
                if not attr.startswith('param_'):
                    row[f'attr_{attr}'] = value
            
            data.append(row)
        
        return pd.DataFrame(data)
    
    def _save_visualizations(self):
        """시각화 저장"""
        
        if not self.study or len(self.study.trials) < 2:
            return
        
        try:
            # 1. 최적화 히스토리
            fig = plot_optimization_history(self.study)
            fig.write_html(self.results_dir / 'optimization_history.html')
            
            # 2. 파라미터 중요도
            if len(self.study.trials) >= 10:
                fig = plot_param_importances(self.study)
                fig.write_html(self.results_dir / 'param_importances.html')
            
            # 3. 병렬 좌표
            fig = plot_parallel_coordinate(self.study)
            fig.write_html(self.results_dir / 'parallel_coordinate.html')
            
            # 4. 슬라이스 플롯
            fig = plot_slice(self.study)
            fig.write_html(self.results_dir / 'slice_plot.html')
            
            logger.info("시각화 저장 완료")
            
        except Exception as e:
            logger.error(f"시각화 저장 에러: {e}")
    
    def resume_optimization(self,
                          study_path: str,
                          additional_trials: int) -> optuna.Study:
        """최적화 재개
        
        Args:
            study_path: 저장된 Study 경로
            additional_trials: 추가 trial 수
            
        Returns:
            재개된 Study
        """
        
        # Study 로드
        with open(study_path, 'rb') as f:
            self.study = pickle.load(f)
        
        logger.info(f"Study 로드: {study_path}")
        logger.info(f"기존 trials: {len(self.study.trials)}")
        
        # 추가 최적화
        return self.optimize(n_trials=additional_trials)
    
    def hyperparameter_tuning_cv(self,
                                n_folds: int = 5,
                                n_trials_per_fold: int = 20) -> Dict:
        """교차 검증을 통한 하이퍼파라미터 튜닝
        
        Args:
            n_folds: 폴드 수
            n_trials_per_fold: 폴드당 trial 수
            
        Returns:
            CV 결과
        """
        
        # 데이터 기간 분할
        start = pd.to_datetime(self.start_date)
        end = pd.to_datetime(self.end_date)
        total_days = (end - start).days
        fold_days = total_days // n_folds
        
        cv_results = []
        
        for fold in range(n_folds):
            logger.info(f"\nFold {fold+1}/{n_folds}")
            
            # 학습/검증 기간 설정
            fold_start = start + pd.Timedelta(days=fold * fold_days)
            fold_end = fold_start + pd.Timedelta(days=fold_days)
            
            if fold == n_folds - 1:
                fold_end = end
            
            # 폴드별 Study 생성
            fold_study_name = f"{self.study_name}_fold_{fold}"
            
            # 폴드별 목적 함수
            fold_objective = ObjectiveFunction(
                strategy_class=self.strategy_class,
                symbols=self.symbols,
                start_date=fold_start.strftime('%Y-%m-%d'),
                end_date=fold_end.strftime('%Y-%m-%d'),
                config=self.config,
                db_path=self.db_path
            )
            
            # 폴드 최적화
            fold_study = optuna.create_study(
                study_name=fold_study_name,
                direction='maximize'
            )
            
            fold_study.optimize(
                lambda trial: fold_objective(trial),
                n_trials=n_trials_per_fold,
                show_progress_bar=False
            )
            
            cv_results.append({
                'fold': fold,
                'best_value': fold_study.best_value,
                'best_params': fold_study.best_params,
                'n_trials': len(fold_study.trials)
            })
        
        # CV 결과 분석
        cv_scores = [r['best_value'] for r in cv_results]
        
        cv_summary = {
            'mean_score': np.mean(cv_scores),
            'std_score': np.std(cv_scores),
            'fold_results': cv_results,
            'consistent_params': self._find_consistent_params(cv_results)
        }
        
        # 결과 저장
        cv_path = self.results_dir / 'cv_results.json'
        with open(cv_path, 'w') as f:
            json.dump(cv_summary, f, indent=2, default=str)
        
        logger.info(f"\nCV 평균 점수: {cv_summary['mean_score']:.4f} ± {cv_summary['std_score']:.4f}")
        
        return cv_summary
    
    def _find_consistent_params(self, cv_results: List[Dict]) -> Dict:
        """CV에서 일관된 파라미터 찾기"""
        
        if not cv_results:
            return {}
        
        # 모든 폴드의 최고 파라미터 수집
        all_params = {}
        
        for result in cv_results:
            for param, value in result['best_params'].items():
                if param not in all_params:
                    all_params[param] = []
                all_params[param].append(value)
        
        # 일관된 파라미터 (중앙값)
        consistent_params = {}
        
        for param, values in all_params.items():
            if all(isinstance(v, (int, float)) for v in values):
                # 수치형: 중앙값
                consistent_params[param] = np.median(values)
            else:
                # 범주형: 최빈값
                from collections import Counter
                most_common = Counter(values).most_common(1)[0][0]
                consistent_params[param] = most_common
        
        return consistent_params

# CLI 실행용
def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Optuna 최적화 실행')
    parser.add_argument('--config', required=True, help='설정 파일')
    parser.add_argument('--strategy', required=True, help='전략 클래스')
    parser.add_argument('--symbols', nargs='+', required=True, help='심볼 리스트')
    parser.add_argument('--start', required=True, help='시작 날짜')
    parser.add_argument('--end', required=True, help='종료 날짜')
    parser.add_argument('--trials', type=int, default=100, help='Trial 수')
    parser.add_argument('--jobs', type=int, default=1, help='병렬 작업 수')
    parser.add_argument('--cv', action='store_true', help='교차 검증')
    parser.add_argument('--db', default='data/trading.db', help='DB 경로')
    
    args = parser.parse_args()
    
    # 설정 로드
    config = IOUtils.load_config(args.config)
    
    # 전략 로드
    module_name, class_name = args.strategy.rsplit('.', 1)
    module = __import__(module_name, fromlist=[class_name])
    strategy_class = getattr(module, class_name)
    
    # Runner 생성
    runner = OptunaRunner(
        config=config,
        strategy_class=strategy_class,
        symbols=args.symbols,
        start_date=args.start,
        end_date=args.end,
        db_path=args.db
    )
    
    # 최적화 실행
    if args.cv:
        # 교차 검증
        results = runner.hyperparameter_tuning_cv()
    else:
        # 일반 최적화
        study = runner.optimize(
            n_trials=args.trials,
            n_jobs=args.jobs
        )

if __name__ == "__main__":
    main()