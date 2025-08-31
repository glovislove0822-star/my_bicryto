"""모델 캘리브레이션 모듈"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any
from datetime import datetime
import logging
import pickle
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from ..utils.logging import Logger
from ..utils.io import IOUtils

logger = Logger.get_logger(__name__)

class ModelCalibrator:
    """모델 확률 캘리브레이션
    
    예측 확률을 실제 확률에 맞게 보정
    """
    
    def __init__(self, model_dir: str = "models"):
        """
        Args:
            model_dir: 모델 디렉토리
        """
        self.model_dir = Path(model_dir)
        self.calibrators = {}
        self.calibration_metrics = {}
    
    def calibrate(self,
                 model: Any,
                 X_cal: pd.DataFrame,
                 y_cal: pd.Series,
                 method: str = 'isotonic',
                 cv: int = 3) -> Any:
        """모델 캘리브레이션
        
        Args:
            model: 원본 모델
            X_cal: 캘리브레이션 데이터
            y_cal: 캘리브레이션 타겟
            method: 캘리브레이션 방법 ('isotonic', 'sigmoid', 'beta')
            cv: Cross-validation 폴드 수
            
        Returns:
            캘리브레이션된 모델
        """
        
        logger.info(f"모델 캘리브레이션 시작 (method={method})")
        
        if method == 'isotonic':
            # Isotonic Regression
            calibrated_model = self._isotonic_calibration(
                model, X_cal, y_cal, cv
            )
        elif method == 'sigmoid':
            # Platt Scaling
            calibrated_model = self._sigmoid_calibration(
                model, X_cal, y_cal, cv
            )
        elif method == 'beta':
            # Beta Calibration
            calibrated_model = self._beta_calibration(
                model, X_cal, y_cal
            )
        else:
            raise ValueError(f"Unknown calibration method: {method}")
        
        # 캘리브레이션 평가
        self._evaluate_calibration(model, calibrated_model, X_cal, y_cal)
        
        # 캘리브레이션 플롯
        self._plot_calibration_curve(model, calibrated_model, X_cal, y_cal)
        
        return calibrated_model
    
    def _isotonic_calibration(self,
                             model: Any,
                             X_cal: pd.DataFrame,
                             y_cal: pd.Series,
                             cv: int) -> Any:
        """Isotonic Regression 캘리브레이션"""
        
        # sklearn의 CalibratedClassifierCV 사용
        calibrated = CalibratedClassifierCV(
            model,
            method='isotonic',
            cv=cv
        )
        
        # 학습
        calibrated.fit(X_cal, y_cal)
        
        # 대체 구현 (직접 Isotonic Regression)
        y_pred = self._get_predictions(model, X_cal)
        
        iso_reg = IsotonicRegression(out_of_bounds='clip')
        iso_reg.fit(y_pred, y_cal)
        
        # 캘리브레이터 저장
        self.calibrators['isotonic'] = iso_reg
        
        return calibrated
    
    def _sigmoid_calibration(self,
                            model: Any,
                            X_cal: pd.DataFrame,
                            y_cal: pd.Series,
                            cv: int) -> Any:
        """Platt Scaling (Sigmoid) 캘리브레이션"""
        
        # sklearn의 CalibratedClassifierCV 사용
        calibrated = CalibratedClassifierCV(
            model,
            method='sigmoid',
            cv=cv
        )
        
        # 학습
        calibrated.fit(X_cal, y_cal)
        
        # 대체 구현 (Logistic Regression)
        y_pred = self._get_predictions(model, X_cal).reshape(-1, 1)
        
        lr = LogisticRegression()
        lr.fit(y_pred, y_cal)
        
        # 캘리브레이터 저장
        self.calibrators['sigmoid'] = lr
        
        return calibrated
    
    def _beta_calibration(self,
                         model: Any,
                         X_cal: pd.DataFrame,
                         y_cal: pd.Series) -> Any:
        """Beta Calibration
        
        확률을 Beta 분포로 모델링
        """
        
        from scipy.optimize import minimize
        from scipy.special import betaln
        
        y_pred = self._get_predictions(model, X_cal)
        
        # Beta 분포 파라미터 최적화
        def beta_nll(params, y_true, y_pred):
            """Beta 분포 Negative Log Likelihood"""
            a, b = params
            
            # 안정성을 위한 클리핑
            y_pred = np.clip(y_pred, 1e-6, 1-1e-6)
            
            # Log likelihood
            ll = (a - 1) * np.log(y_pred) + (b - 1) * np.log(1 - y_pred)
            ll -= betaln(a, b)
            
            # 가중치 (실제 라벨 기반)
            weights = np.where(y_true == 1, y_pred, 1 - y_pred)
            
            return -np.sum(weights * ll)
        
        # 초기값
        init_params = [1.0, 1.0]
        
        # 최적화
        result = minimize(
            beta_nll,
            init_params,
            args=(y_cal, y_pred),
            method='L-BFGS-B',
            bounds=[(0.1, 10), (0.1, 10)]
        )
        
        a_opt, b_opt = result.x
        
        # Beta 캘리브레이션 함수
        class BetaCalibrator:
            def __init__(self, a, b):
                self.a = a
                self.b = b
            
            def transform(self, p):
                """Beta 변환"""
                from scipy.stats import beta
                p = np.clip(p, 1e-6, 1-1e-6)
                return beta.cdf(p, self.a, self.b)
        
        beta_cal = BetaCalibrator(a_opt, b_opt)
        self.calibrators['beta'] = beta_cal
        
        # 캘리브레이션된 모델 래퍼
        class CalibratedModel:
            def __init__(self, base_model, calibrator):
                self.base_model = base_model
                self.calibrator = calibrator
            
            def predict_proba(self, X):
                base_pred = self._get_base_predictions(X)
                cal_pred = self.calibrator.transform(base_pred)
                return np.column_stack([1 - cal_pred, cal_pred])
            
            def _get_base_predictions(self, X):
                # 모델 타입에 따른 예측
                if hasattr(self.base_model, 'predict_proba'):
                    return self.base_model.predict_proba(X)[:, 1]
                elif hasattr(self.base_model, 'predict'):
                    return self.base_model.predict(X)
                else:
                    raise ValueError("Model must have predict or predict_proba method")
        
        return CalibratedModel(model, beta_cal)
    
    def _get_predictions(self, model: Any, X: pd.DataFrame) -> np.ndarray:
        """모델 예측 획득"""
        
        if hasattr(model, 'predict_proba'):
            return model.predict_proba(X)[:, 1]
        elif hasattr(model, 'predict'):
            import xgboost as xgb
            if isinstance(model, xgb.Booster):
                dmatrix = xgb.DMatrix(X)
                return model.predict(dmatrix)
            else:
                return model.predict(X)
        else:
            raise ValueError("Model must have predict or predict_proba method")
    
    def _evaluate_calibration(self,
                            original_model: Any,
                            calibrated_model: Any,
                            X_test: pd.DataFrame,
                            y_test: pd.Series):
        """캘리브레이션 평가"""
        
        # 원본 모델 예측
        y_pred_orig = self._get_predictions(original_model, X_test)
        
        # 캘리브레이션 모델 예측
        if hasattr(calibrated_model, 'predict_proba'):
            y_pred_cal = calibrated_model.predict_proba(X_test)[:, 1]
        else:
            y_pred_cal = self._get_predictions(calibrated_model, X_test)
        
        # Brier Score
        brier_orig = brier_score_loss(y_test, y_pred_orig)
        brier_cal = brier_score_loss(y_test, y_pred_cal)
        
        # Log Loss
        logloss_orig = log_loss(y_test, y_pred_orig)
        logloss_cal = log_loss(y_test, y_pred_cal)
        
        # ECE (Expected Calibration Error)
        ece_orig = self._calculate_ece(y_test, y_pred_orig)
        ece_cal = self._calculate_ece(y_test, y_pred_cal)
        
        # 메트릭 저장
        self.calibration_metrics = {
            'original': {
                'brier_score': brier_orig,
                'log_loss': logloss_orig,
                'ece': ece_orig
            },
            'calibrated': {
                'brier_score': brier_cal,
                'log_loss': logloss_cal,
                'ece': ece_cal
            },
            'improvement': {
                'brier_score': (brier_orig - brier_cal) / brier_orig * 100,
                'log_loss': (logloss_orig - logloss_cal) / logloss_orig * 100,
                'ece': (ece_orig - ece_cal) / ece_orig * 100
            }
        }
        
        logger.info("\n캘리브레이션 평가:")
        logger.info(f"  Brier Score: {brier_orig:.4f} → {brier_cal:.4f} "
                   f"({self.calibration_metrics['improvement']['brier_score']:+.1f}%)")
        logger.info(f"  Log Loss: {logloss_orig:.4f} → {logloss_cal:.4f} "
                   f"({self.calibration_metrics['improvement']['log_loss']:+.1f}%)")
        logger.info(f"  ECE: {ece_orig:.4f} → {ece_cal:.4f} "
                   f"({self.calibration_metrics['improvement']['ece']:+.1f}%)")
    
    def _calculate_ece(self,
                      y_true: np.ndarray,
                      y_prob: np.ndarray,
                      n_bins: int = 10) -> float:
        """Expected Calibration Error 계산"""
        
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = y_true[in_bin].mean()
                avg_confidence_in_bin = y_prob[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return ece
    
    def _plot_calibration_curve(self,
                               original_model: Any,
                               calibrated_model: Any,
                               X_test: pd.DataFrame,
                               y_test: pd.Series):
        """캘리브레이션 곡선 플롯"""
        
        from sklearn.calibration import calibration_curve
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # 원본 모델 예측
        y_pred_orig = self._get_predictions(original_model, X_test)
        
        # 캘리브레이션 모델 예측
        if hasattr(calibrated_model, 'predict_proba'):
            y_pred_cal = calibrated_model.predict_proba(X_test)[:, 1]
        else:
            y_pred_cal = self._get_predictions(calibrated_model, X_test)
        
        # Calibration Plot
        n_bins = 10
        
        # 원본 모델
        fraction_pos_orig, mean_pred_orig = calibration_curve(
            y_test, y_pred_orig, n_bins=n_bins
        )
        
        # 캘리브레이션 모델
        fraction_pos_cal, mean_pred_cal = calibration_curve(
            y_test, y_pred_cal, n_bins=n_bins
        )
        
        # Plot 1: Calibration Curve
        axes[0].plot([0, 1], [0, 1], 'k--', label='Perfect')
        axes[0].plot(mean_pred_orig, fraction_pos_orig, 
                    'o-', label='Original', alpha=0.7)
        axes[0].plot(mean_pred_cal, fraction_pos_cal, 
                    's-', label='Calibrated', alpha=0.7)
        axes[0].set_xlabel('Mean Predicted Probability')
        axes[0].set_ylabel('Fraction of Positives')
        axes[0].set_title('Calibration Plot')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Histogram
        axes[1].hist(y_pred_orig, bins=30, alpha=0.5, 
                    label='Original', density=True)
        axes[1].hist(y_pred_cal, bins=30, alpha=0.5, 
                    label='Calibrated', density=True)
        axes[1].set_xlabel('Predicted Probability')
        axes[1].set_ylabel('Density')
        axes[1].set_title('Prediction Distribution')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.model_dir / 'calibration_plot.png', dpi=100, bbox_inches='tight')
        plt.close()
        
        logger.info(f"캘리브레이션 플롯 저장: {self.model_dir / 'calibration_plot.png'}")
    
    def save_calibrator(self, name: str):
        """캘리브레이터 저장"""
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        for cal_name, calibrator in self.calibrators.items():
            cal_path = self.model_dir / f'calibrator_{cal_name}_{name}_{timestamp}.pkl'
            
            with open(cal_path, 'wb') as f:
                pickle.dump(calibrator, f)
            
            logger.info(f"캘리브레이터 저장: {cal_path}")
        
        # 메트릭 저장
        if self.calibration_metrics:
            metrics_path = self.model_dir / f'calibration_metrics_{name}_{timestamp}.json'
            IOUtils.save_json(self.calibration_metrics, metrics_path)

# CLI 실행용
def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='모델 캘리브레이션')
    parser.add_argument('--model', required=True, help='모델 파일 경로')
    parser.add_argument('--data', required=True, help='캘리브레이션 데이터 경로')
    parser.add_argument('--method', default='isotonic', 
                       choices=['isotonic', 'sigmoid', 'beta'],
                       help='캘리브레이션 방법')
    parser.add_argument('--cv', type=int, default=3, help='CV 폴드 수')
    parser.add_argument('--output', default='models', help='출력 디렉토리')
    
    args = parser.parse_args()
    
    # 모델 로드
    with open(args.model, 'rb') as f:
        model = pickle.load(f)
    
    # 데이터 로드
    data_path = Path(args.data)
    X_cal = pd.read_parquet(data_path / 'X_cal.parquet')
    y_cal = pd.read_parquet(data_path / 'y_cal.parquet')['target']
    
    # 캘리브레이션
    calibrator = ModelCalibrator(model_dir=args.output)
    
    calibrated_model = calibrator.calibrate(
        model=model,
        X_cal=X_cal,
        y_cal=y_cal,
        method=args.method,
        cv=args.cv
    )
    
    # 저장
    calibrator.save_calibrator('final')

if __name__ == "__main__":
    main()