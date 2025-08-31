"""임계값 최적화 모듈"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any
from datetime import datetime
import logging
from scipy.optimize import minimize_scalar, differential_evolution
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from ..utils.logging import Logger
from ..utils.io import IOUtils

logger = Logger.get_logger(__name__)

class ThresholdOptimizer:
    """최적 임계값 탐색 및 관리"""
    
    def __init__(self):
        """초기화"""
        self.optimal_thresholds = {}
        self.threshold_metrics = {}
        self.optimization_history = []
    
    def optimize_threshold(self,
                          y_true: np.ndarray,
                          y_pred_proba: np.ndarray,
                          metric: str = 'f1',
                          constraints: Optional[Dict] = None,
                          method: str = 'grid_search') -> float:
        """최적 임계값 탐색
        
        Args:
            y_true: 실제 라벨
            y_pred_proba: 예측 확률
            metric: 최적화 메트릭 ('f1', 'precision', 'recall', 'profit')
            constraints: 제약 조건 (예: {'min_precision': 0.6})
            method: 최적화 방법 ('grid_search', 'binary_search', 'bayesian')
            
        Returns:
            최적 임계값
        """
        
        logger.info(f"임계값 최적화 시작 (metric={metric}, method={method})")
        
        if method == 'grid_search':
            optimal_threshold = self._grid_search(
                y_true, y_pred_proba, metric, constraints
            )
        elif method == 'binary_search':
            optimal_threshold = self._binary_search(
                y_true, y_pred_proba, metric, constraints
            )
        elif method == 'bayesian':
            optimal_threshold = self._bayesian_optimization(
                y_true, y_pred_proba, metric, constraints
            )
        else:
            raise ValueError(f"Unknown optimization method: {method}")
        
        # 최적 임계값에서의 성능 평가
        self._evaluate_threshold(y_true, y_pred_proba, optimal_threshold, metric)
        
        # 시각화
        self._plot_threshold_analysis(y_true, y_pred_proba, optimal_threshold, metric)
        
        return optimal_threshold
    
    def _grid_search(self,
                    y_true: np.ndarray,
                    y_pred_proba: np.ndarray,
                    metric: str,
                    constraints: Optional[Dict]) -> float:
        """그리드 서치 최적화"""
        
        thresholds = np.linspace(0.1, 0.9, 81)
        best_score = -np.inf
        best_threshold = 0.5
        
        for threshold in thresholds:
            y_pred = (y_pred_proba >= threshold).astype(int)
            
            # 제약 조건 체크
            if constraints:
                if not self._check_constraints(y_true, y_pred, constraints):
                    continue
            
            # 메트릭 계산
            score = self._calculate_metric(y_true, y_pred, y_pred_proba, metric, threshold)
            
            if score > best_score:
                best_score = score
                best_threshold = threshold
            
            # 기록
            self.optimization_history.append({
                'threshold': threshold,
                'score': score,
                'metric': metric
            })
        
        logger.info(f"Grid Search 완료: threshold={best_threshold:.3f}, {metric}={best_score:.4f}")
        
        return best_threshold
    
    def _binary_search(self,
                      y_true: np.ndarray,
                      y_pred_proba: np.ndarray,
                      metric: str,
                      constraints: Optional[Dict]) -> float:
        """이진 탐색 최적화"""
        
        def objective(threshold):
            y_pred = (y_pred_proba >= threshold).astype(int)
            
            # 제약 조건 체크
            if constraints:
                if not self._check_constraints(y_true, y_pred, constraints):
                    return -1e10  # 페널티
            
            score = self._calculate_metric(y_true, y_pred, y_pred_proba, metric, threshold)
            return -score  # 최소화 문제로 변환
        
        # 최적화
        result = minimize_scalar(
            objective,
            bounds=(0.1, 0.9),
            method='bounded',
            options={'xatol': 0.001}
        )
        
        optimal_threshold = result.x
        
        logger.info(f"Binary Search 완료: threshold={optimal_threshold:.3f}")
        
        return optimal_threshold
    
    def _bayesian_optimization(self,
                              y_true: np.ndarray,
                              y_pred_proba: np.ndarray,
                              metric: str,
                              constraints: Optional[Dict]) -> float:
        """베이지안 최적화"""
        
        from skopt import gp_minimize
        from skopt.space import Real
        
        def objective(params):
            threshold = params[0]
            y_pred = (y_pred_proba >= threshold).astype(int)
            
            # 제약 조건 체크
            if constraints:
                if not self._check_constraints(y_true, y_pred, constraints):
                    return 1e10  # 페널티
            
            score = self._calculate_metric(y_true, y_pred, y_pred_proba, metric, threshold)
            return -score  # 최소화
        
        # 베이지안 최적화
        space = [Real(0.1, 0.9, name='threshold')]
        
        result = gp_minimize(
            func=objective,
            dimensions=space,
            n_calls=50,
            n_initial_points=10,
            acq_func='EI',
            random_state=42
        )
        
        optimal_threshold = result.x[0]
        
        logger.info(f"Bayesian Optimization 완료: threshold={optimal_threshold:.3f}")
        
        return optimal_threshold
    
    def _calculate_metric(self,
                        y_true: np.ndarray,
                        y_pred: np.ndarray,
                        y_pred_proba: np.ndarray,
                        metric: str,
                        threshold: float) -> float:
        """메트릭 계산"""
        
        if metric == 'f1':
            return f1_score(y_true, y_pred)
        
        elif metric == 'precision':
            return precision_score(y_true, y_pred, zero_division=0)
        
        elif metric == 'recall':
            return recall_score(y_true, y_pred, zero_division=0)
        
        elif metric == 'profit':
            # 수익 기반 메트릭 (커스텀)
            tp = np.sum((y_true == 1) & (y_pred == 1))
            fp = np.sum((y_true == 0) & (y_pred == 1))
            fn = np.sum((y_true == 1) & (y_pred == 0))
            
            # 수익/손실 가정
            profit_per_tp = 1.0  # True Positive 수익
            loss_per_fp = 0.5   # False Positive 손실
            opportunity_cost_per_fn = 0.3  # False Negative 기회비용
            
            total_profit = (profit_per_tp * tp - 
                          loss_per_fp * fp - 
                          opportunity_cost_per_fn * fn)
            
            return total_profit
        
        elif metric == 'sharpe':
            # Sharpe Ratio 기반
            returns = np.where(y_pred == 1, 
                             np.where(y_true == 1, 0.01, -0.005),  # 수익률 가정
                             0)
            
            if returns.std() > 0:
                sharpe = returns.mean() / returns.std() * np.sqrt(252)  # 연율화
            else:
                sharpe = 0
            
            return sharpe
        
        elif metric == 'custom':
            # 복합 메트릭
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            
            # 가중 평균
            if precision + recall > 0:
                # Precision을 더 중요시
                score = (2 * precision + recall) / 3
            else:
                score = 0
            
            return score
        
        else:
            raise ValueError(f"Unknown metric: {metric}")
    
    def _check_constraints(self,
                         y_true: np.ndarray,
                         y_pred: np.ndarray,
                         constraints: Dict) -> bool:
        """제약 조건 체크"""
        
        # 최소 정밀도
        if 'min_precision' in constraints:
            precision = precision_score(y_true, y_pred, zero_division=0)
            if precision < constraints['min_precision']:
                return False
        
        # 최소 재현율
        if 'min_recall' in constraints:
            recall = recall_score(y_true, y_pred, zero_division=0)
            if recall < constraints['min_recall']:
                return False
        
        # 최소 거래 수
        if 'min_trades' in constraints:
            n_trades = np.sum(y_pred == 1)
            if n_trades < constraints['min_trades']:
                return False
        
        # 최대 거래 수
        if 'max_trades' in constraints:
            n_trades = np.sum(y_pred == 1)
            if n_trades > constraints['max_trades']:
                return False
        
        return True
    
    def _evaluate_threshold(self,
                          y_true: np.ndarray,
                          y_pred_proba: np.ndarray,
                          threshold: float,
                          metric: str):
        """임계값 평가"""
        
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        # 기본 메트릭
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred)
        
        # 거래 통계
        n_trades = np.sum(y_pred == 1)
        n_positive = np.sum(y_true == 1)
        trade_ratio = n_trades / len(y_true)
        
        # Confusion Matrix
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        
        # 메트릭 저장
        self.threshold_metrics[threshold] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'n_trades': n_trades,
            'trade_ratio': trade_ratio,
            'tp': tp,
            'fp': fp,
            'tn': tn,
            'fn': fn,
            'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
            'npv': tn / (tn + fn) if (tn + fn) > 0 else 0
        }
        
        logger.info(f"\n임계값 {threshold:.3f} 평가:")
        logger.info(f"  Precision: {precision:.3f}")
        logger.info(f"  Recall: {recall:.3f}")
        logger.info(f"  F1 Score: {f1:.3f}")
        logger.info(f"  거래 수: {n_trades} ({trade_ratio:.1%})")
        logger.info(f"  Confusion Matrix: TP={tp}, FP={fp}, TN={tn}, FN={fn}")
    
    def _plot_threshold_analysis(self,
                                y_true: np.ndarray,
                                y_pred_proba: np.ndarray,
                                optimal_threshold: float,
                                metric: str):
        """임계값 분석 플롯"""
        
        thresholds = np.linspace(0.05, 0.95, 91)
        metrics = {
            'precision': [],
            'recall': [],
            'f1': [],
            'trade_ratio': []
        }
        
        for threshold in thresholds:
            y_pred = (y_pred_proba >= threshold).astype(int)
            
            metrics['precision'].append(precision_score(y_true, y_pred, zero_division=0))
            metrics['recall'].append(recall_score(y_true, y_pred, zero_division=0))
            metrics['f1'].append(f1_score(y_true, y_pred, zero_division=0))
            metrics['trade_ratio'].append(np.sum(y_pred == 1) / len(y_true))
        
        # 플롯
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. Precision-Recall vs Threshold
        axes[0, 0].plot(thresholds, metrics['precision'], label='Precision', alpha=0.7)
        axes[0, 0].plot(thresholds, metrics['recall'], label='Recall', alpha=0.7)
        axes[0, 0].plot(thresholds, metrics['f1'], label='F1', alpha=0.7, linewidth=2)
        axes[0, 0].axvline(optimal_threshold, color='red', linestyle='--', 
                          label=f'Optimal ({optimal_threshold:.3f})')
        axes[0, 0].set_xlabel('Threshold')
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].set_title('Metrics vs Threshold')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Trade Ratio vs Threshold
        axes[0, 1].plot(thresholds, metrics['trade_ratio'], color='green', alpha=0.7)
        axes[0, 1].axvline(optimal_threshold, color='red', linestyle='--')
        axes[0, 1].set_xlabel('Threshold')
        axes[0, 1].set_ylabel('Trade Ratio')
        axes[0, 1].set_title('Trade Frequency vs Threshold')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Precision-Recall Curve
        from sklearn.metrics import precision_recall_curve
        precisions, recalls, pr_thresholds = precision_recall_curve(y_true, y_pred_proba)
        
        axes[1, 0].plot(recalls, precisions, alpha=0.7)
        
        # 최적 임계값 위치 표시
        y_pred_opt = (y_pred_proba >= optimal_threshold).astype(int)
        opt_precision = precision_score(y_true, y_pred_opt, zero_division=0)
        opt_recall = recall_score(y_true, y_pred_opt, zero_division=0)
        axes[1, 0].scatter([opt_recall], [opt_precision], color='red', s=100, zorder=5)
        
        axes[1, 0].set_xlabel('Recall')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].set_title('Precision-Recall Curve')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Probability Distribution
        axes[1, 1].hist(y_pred_proba[y_true == 0], bins=50, alpha=0.5, 
                       label='Negative', density=True, color='blue')
        axes[1, 1].hist(y_pred_proba[y_true == 1], bins=50, alpha=0.5, 
                       label='Positive', density=True, color='red')
        axes[1, 1].axvline(optimal_threshold, color='black', linestyle='--', 
                          label=f'Threshold ({optimal_threshold:.3f})')
        axes[1, 1].set_xlabel('Predicted Probability')
        axes[1, 1].set_ylabel('Density')
        axes[1, 1].set_title('Probability Distribution by Class')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('threshold_analysis.png', dpi=100, bbox_inches='tight')
        plt.close()
        
        logger.info("임계값 분석 플롯 저장: threshold_analysis.png")
    
    def optimize_multi_threshold(self,
                                y_true: np.ndarray,
                                y_pred_proba: np.ndarray,
                                n_classes: int = 3) -> List[float]:
        """다중 임계값 최적화 (multi-class 또는 multi-level)
        
        Args:
            y_true: 실제 라벨
            y_pred_proba: 예측 확률
            n_classes: 클래스 수
            
        Returns:
            임계값 리스트
        """
        
        if n_classes == 3:
            # 3클래스: Sell, Hold, Buy
            thresholds = self._optimize_ternary_thresholds(y_true, y_pred_proba)
        else:
            # 일반적인 다중 클래스
            thresholds = self._optimize_multiclass_thresholds(
                y_true, y_pred_proba, n_classes
            )
        
        return thresholds
    
    def _optimize_ternary_thresholds(self,
                                    y_true: np.ndarray,
                                    y_pred_proba: np.ndarray) -> List[float]:
        """3진 분류 임계값 최적화 (Sell/Hold/Buy)"""
        
        def objective(thresholds):
            lower_threshold, upper_threshold = thresholds
            
            if lower_threshold >= upper_threshold:
                return -1e10  # 잘못된 순서
            
            # 분류
            y_pred = np.zeros_like(y_true)
            y_pred[y_pred_proba < lower_threshold] = -1  # Sell
            y_pred[(y_pred_proba >= lower_threshold) & 
                  (y_pred_proba < upper_threshold)] = 0  # Hold
            y_pred[y_pred_proba >= upper_threshold] = 1  # Buy
            
            # 수익 계산 (예시)
            returns = np.where(y_pred == 1, 
                             np.where(y_true == 1, 0.01, -0.005),
                             np.where(y_pred == -1,
                                    np.where(y_true == -1, 0.01, -0.005),
                                    0))
            
            # Sharpe Ratio
            if returns.std() > 0:
                sharpe = returns.mean() / returns.std() * np.sqrt(252)
            else:
                sharpe = 0
            
            return -sharpe  # 최소화
        
        # 최적화
        result = differential_evolution(
            objective,
            bounds=[(0.2, 0.5), (0.5, 0.8)],
            maxiter=100,
            seed=42
        )
        
        thresholds = list(result.x)
        
        logger.info(f"Ternary thresholds: {thresholds}")
        
        return thresholds
    
    def _optimize_multiclass_thresholds(self,
                                       y_true: np.ndarray,
                                       y_pred_proba: np.ndarray,
                                       n_classes: int) -> List[float]:
        """다중 클래스 임계값 최적화"""
        
        # 균등 분할로 초기화
        thresholds = np.linspace(0, 1, n_classes + 1)[1:-1]
        
        # TODO: 더 정교한 최적화 구현
        
        return thresholds.tolist()
    
    def adaptive_threshold(self,
                         y_pred_proba: np.ndarray,
                         market_regime: str,
                         base_threshold: float = 0.5) -> float:
        """시장 상황에 따른 적응형 임계값
        
        Args:
            y_pred_proba: 예측 확률
            market_regime: 시장 레짐
            base_threshold: 기본 임계값
            
        Returns:
            조정된 임계값
        """
        
        # 레짐별 조정
        regime_adjustments = {
            'bull': -0.05,  # 상승장: 더 공격적
            'bear': +0.10,  # 하락장: 더 보수적
            'high_vol': +0.05,  # 고변동성: 신중
            'low_vol': -0.03,  # 저변동성: 약간 공격적
            'trending': -0.02,  # 트렌드: 추종
            'ranging': +0.03  # 횡보: 신중
        }
        
        adjustment = regime_adjustments.get(market_regime, 0)
        
        # 확률 분포 기반 추가 조정
        prob_std = np.std(y_pred_proba)
        if prob_std < 0.1:  # 확신도가 낮음
            adjustment += 0.02
        elif prob_std > 0.3:  # 확신도가 높음
            adjustment -= 0.02
        
        # 최종 임계값
        adapted_threshold = np.clip(base_threshold + adjustment, 0.1, 0.9)
        
        return adapted_threshold

# CLI 실행용
def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='임계값 최적화')
    parser.add_argument('--predictions', required=True, help='예측 결과 파일')
    parser.add_argument('--metric', default='f1', 
                       choices=['f1', 'precision', 'recall', 'profit', 'sharpe'],
                       help='최적화 메트릭')
    parser.add_argument('--method', default='grid_search',
                       choices=['grid_search', 'binary_search', 'bayesian'],
                       help='최적화 방법')
    parser.add_argument('--min-precision', type=float, help='최소 정밀도 제약')
    parser.add_argument('--min-recall', type=float, help='최소 재현율 제약')
    
    args = parser.parse_args()
    
    # 데이터 로드
    data = pd.read_csv(args.predictions)
    y_true = data['y_true'].values
    y_pred_proba = data['y_pred_proba'].values
    
    # 제약 조건
    constraints = {}
    if args.min_precision:
        constraints['min_precision'] = args.min_precision
    if args.min_recall:
        constraints['min_recall'] = args.min_recall
    
    # 최적화
    optimizer = ThresholdOptimizer()
    
    optimal_threshold = optimizer.optimize_threshold(
        y_true=y_true,
        y_pred_proba=y_pred_proba,
        metric=args.metric,
        constraints=constraints,
        method=args.method
    )
    
    print(f"\n최적 임계값: {optimal_threshold:.4f}")

if __name__ == "__main__":
    main()