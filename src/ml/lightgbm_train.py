"""LightGBM 모델 학습 모듈"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any
from datetime import datetime
import logging
import pickle
import json
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, log_loss, confusion_matrix, classification_report
)
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from ..utils.logging import Logger
from ..utils.io import IOUtils
from .dataset import DatasetBuilder

logger = Logger.get_logger(__name__)

class LightGBMTrainer:
    """LightGBM 메타 라벨 분류기 학습"""
    
    def __init__(self, 
                 model_dir: str = "models",
                 use_ensemble: bool = True):
        """
        Args:
            model_dir: 모델 저장 디렉토리
            use_ensemble: 앙상블 사용 여부
        """
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.use_ensemble = use_ensemble
        
        # 기본 하이퍼파라미터
        self.lgb_params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'max_depth': 6,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'min_data_in_leaf': 100,
            'lambda_l1': 0.1,
            'lambda_l2': 0.1,
            'min_gain_to_split': 0.01,
            'is_unbalance': True,
            'verbosity': -1,
            'seed': 42,
            'n_jobs': -1
        }
        
        self.xgb_params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'max_depth': 6,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 100,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'scale_pos_weight': 1,
            'seed': 42,
            'n_jobs': -1
        }
        
        self.cat_params = {
            'loss_function': 'Logloss',
            'eval_metric': 'AUC',
            'learning_rate': 0.05,
            'depth': 6,
            'l2_leaf_reg': 3,
            'min_data_in_leaf': 100,
            'random_seed': 42,
            'verbose': False
        }
        
        self.models = {}
        self.feature_importance = {}
        self.training_history = {}
    
    def train(self,
             X_train: pd.DataFrame,
             y_train: pd.Series,
             X_val: Optional[pd.DataFrame] = None,
             y_val: Optional[pd.Series] = None,
             params: Optional[Dict] = None,
             cv_folds: int = 5,
             use_purged_cv: bool = True,
             embargo_bars: int = 2) -> Dict[str, Any]:
        """모델 학습
        
        Args:
            X_train: 학습 피처
            y_train: 학습 타겟
            X_val: 검증 피처
            y_val: 검증 타겟
            params: 커스텀 파라미터
            cv_folds: CV 폴드 수
            use_purged_cv: Purged CV 사용 여부
            embargo_bars: 엠바고 기간
            
        Returns:
            학습 결과 딕셔너리
        """
        
        logger.info("모델 학습 시작")
        
        # 파라미터 업데이트
        if params:
            self.lgb_params.update(params.get('lgb', {}))
            self.xgb_params.update(params.get('xgb', {}))
            self.cat_params.update(params.get('cat', {}))
        
        results = {}
        
        # LightGBM 학습
        logger.info("LightGBM 학습 중...")
        lgb_result = self._train_lightgbm(
            X_train, y_train, X_val, y_val,
            cv_folds, use_purged_cv, embargo_bars
        )
        results['lightgbm'] = lgb_result
        
        if self.use_ensemble:
            # XGBoost 학습
            logger.info("XGBoost 학습 중...")
            xgb_result = self._train_xgboost(
                X_train, y_train, X_val, y_val,
                cv_folds, use_purged_cv, embargo_bars
            )
            results['xgboost'] = xgb_result
            
            # CatBoost 학습
            logger.info("CatBoost 학습 중...")
            cat_result = self._train_catboost(
                X_train, y_train, X_val, y_val,
                cv_folds, use_purged_cv, embargo_bars
            )
            results['catboost'] = cat_result
            
            # 앙상블 생성
            logger.info("앙상블 모델 생성 중...")
            ensemble_result = self._create_ensemble(
                X_train, y_train, X_val, y_val
            )
            results['ensemble'] = ensemble_result
        
        # 피처 중요도 분석
        self._analyze_feature_importance()
        
        # 모델 저장
        self._save_models()
        
        # 결과 요약
        self._print_training_summary(results)
        
        return results
    
    def _train_lightgbm(self,
                       X_train: pd.DataFrame,
                       y_train: pd.Series,
                       X_val: Optional[pd.DataFrame],
                       y_val: Optional[pd.Series],
                       cv_folds: int,
                       use_purged_cv: bool,
                       embargo_bars: int) -> Dict:
        """LightGBM 학습"""
        
        # 데이터셋 생성
        train_data = lgb.Dataset(X_train, label=y_train)
        
        if X_val is not None and y_val is not None:
            valid_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            valid_sets = [train_data, valid_data]
            valid_names = ['train', 'valid']
        else:
            valid_sets = [train_data]
            valid_names = ['train']
        
        # Callbacks
        callbacks = [
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=100)
        ]
        
        # Cross Validation
        if use_purged_cv:
            # Purged K-Fold
            dataset_builder = DatasetBuilder()
            cv_splits = dataset_builder.create_purged_kfold(
                X_train, y_train, cv_folds, embargo_bars
            )
            
            cv_scores = []
            cv_models = []
            
            for fold, (train_idx, val_idx) in enumerate(cv_splits):
                X_fold_train = X_train.iloc[train_idx]
                y_fold_train = y_train.iloc[train_idx]
                X_fold_val = X_train.iloc[val_idx]
                y_fold_val = y_train.iloc[val_idx]
                
                fold_train_data = lgb.Dataset(X_fold_train, label=y_fold_train)
                fold_valid_data = lgb.Dataset(X_fold_val, label=y_fold_val)
                
                model = lgb.train(
                    self.lgb_params,
                    fold_train_data,
                    num_boost_round=1000,
                    valid_sets=[fold_valid_data],
                    valid_names=['valid'],
                    callbacks=callbacks
                )
                
                # 검증 점수
                y_pred = model.predict(X_fold_val)
                score = roc_auc_score(y_fold_val, y_pred)
                cv_scores.append(score)
                cv_models.append(model)
                
                logger.info(f"  Fold {fold+1}: AUC = {score:.4f}")
            
            # 최고 모델 선택
            best_fold = np.argmax(cv_scores)
            self.models['lightgbm'] = cv_models[best_fold]
            
            cv_mean = np.mean(cv_scores)
            cv_std = np.std(cv_scores)
            logger.info(f"  CV AUC: {cv_mean:.4f} ± {cv_std:.4f}")
            
        else:
            # 일반 학습
            self.models['lightgbm'] = lgb.train(
                self.lgb_params,
                train_data,
                num_boost_round=1000,
                valid_sets=valid_sets,
                valid_names=valid_names,
                callbacks=callbacks
            )
        
        # 피처 중요도
        self.feature_importance['lightgbm'] = pd.DataFrame({
            'feature': X_train.columns,
            'importance': self.models['lightgbm'].feature_importance(importance_type='gain'),
            'split': self.models['lightgbm'].feature_importance(importance_type='split')
        }).sort_values('importance', ascending=False)
        
        # 검증 성능
        if X_val is not None and y_val is not None:
            y_pred = self.models['lightgbm'].predict(X_val)
            y_pred_binary = (y_pred > 0.5).astype(int)
            
            result = {
                'auc': roc_auc_score(y_val, y_pred),
                'accuracy': accuracy_score(y_val, y_pred_binary),
                'precision': precision_score(y_val, y_pred_binary),
                'recall': recall_score(y_val, y_pred_binary),
                'f1': f1_score(y_val, y_pred_binary),
                'log_loss': log_loss(y_val, y_pred)
            }
        else:
            result = {'cv_auc': cv_mean if use_purged_cv else None}
        
        return result
    
    def _train_xgboost(self,
                      X_train: pd.DataFrame,
                      y_train: pd.Series,
                      X_val: Optional[pd.DataFrame],
                      y_val: Optional[pd.Series],
                      cv_folds: int,
                      use_purged_cv: bool,
                      embargo_bars: int) -> Dict:
        """XGBoost 학습"""
        
        # DMatrix 생성
        dtrain = xgb.DMatrix(X_train, label=y_train)
        
        if X_val is not None and y_val is not None:
            dval = xgb.DMatrix(X_val, label=y_val)
            evallist = [(dtrain, 'train'), (dval, 'eval')]
        else:
            evallist = [(dtrain, 'train')]
        
        # 학습
        self.models['xgboost'] = xgb.train(
            self.xgb_params,
            dtrain,
            num_boost_round=1000,
            evals=evallist,
            early_stopping_rounds=50,
            verbose_eval=100
        )
        
        # 피처 중요도
        importance = self.models['xgboost'].get_score(importance_type='gain')
        self.feature_importance['xgboost'] = pd.DataFrame({
            'feature': list(importance.keys()),
            'importance': list(importance.values())
        }).sort_values('importance', ascending=False)
        
        # 검증 성능
        if X_val is not None and y_val is not None:
            y_pred = self.models['xgboost'].predict(dval)
            y_pred_binary = (y_pred > 0.5).astype(int)
            
            result = {
                'auc': roc_auc_score(y_val, y_pred),
                'accuracy': accuracy_score(y_val, y_pred_binary),
                'precision': precision_score(y_val, y_pred_binary),
                'recall': recall_score(y_val, y_pred_binary),
                'f1': f1_score(y_val, y_pred_binary)
            }
        else:
            result = {}
        
        return result
    
    def _train_catboost(self,
                       X_train: pd.DataFrame,
                       y_train: pd.Series,
                       X_val: Optional[pd.DataFrame],
                       y_val: Optional[pd.Series],
                       cv_folds: int,
                       use_purged_cv: bool,
                       embargo_bars: int) -> Dict:
        """CatBoost 학습"""
        
        # Pool 생성
        train_pool = cb.Pool(X_train, y_train)
        
        if X_val is not None and y_val is not None:
            val_pool = cb.Pool(X_val, y_val)
        else:
            val_pool = None
        
        # 학습
        self.models['catboost'] = cb.CatBoostClassifier(**self.cat_params)
        self.models['catboost'].fit(
            train_pool,
            eval_set=val_pool,
            early_stopping_rounds=50,
            verbose=100
        )
        
        # 피처 중요도
        self.feature_importance['catboost'] = pd.DataFrame({
            'feature': X_train.columns,
            'importance': self.models['catboost'].get_feature_importance()
        }).sort_values('importance', ascending=False)
        
        # 검증 성능
        if X_val is not None and y_val is not None:
            y_pred = self.models['catboost'].predict_proba(X_val)[:, 1]
            y_pred_binary = (y_pred > 0.5).astype(int)
            
            result = {
                'auc': roc_auc_score(y_val, y_pred),
                'accuracy': accuracy_score(y_val, y_pred_binary),
                'precision': precision_score(y_val, y_pred_binary),
                'recall': recall_score(y_val, y_pred_binary),
                'f1': f1_score(y_val, y_pred_binary)
            }
        else:
            result = {}
        
        return result
    
    def _create_ensemble(self,
                        X_train: pd.DataFrame,
                        y_train: pd.Series,
                        X_val: Optional[pd.DataFrame],
                        y_val: Optional[pd.Series]) -> Dict:
        """앙상블 모델 생성"""
        
        # 가중 평균 앙상블
        self.ensemble_weights = {
            'lightgbm': 0.4,
            'xgboost': 0.3,
            'catboost': 0.3
        }
        
        if X_val is not None and y_val is not None:
            # 각 모델 예측
            predictions = {}
            
            if 'lightgbm' in self.models:
                predictions['lightgbm'] = self.models['lightgbm'].predict(X_val)
            
            if 'xgboost' in self.models:
                dval = xgb.DMatrix(X_val)
                predictions['xgboost'] = self.models['xgboost'].predict(dval)
            
            if 'catboost' in self.models:
                predictions['catboost'] = self.models['catboost'].predict_proba(X_val)[:, 1]
            
            # 가중 평균
            ensemble_pred = np.zeros(len(y_val))
            for model_name, weight in self.ensemble_weights.items():
                if model_name in predictions:
                    ensemble_pred += weight * predictions[model_name]
            
            # 정규화
            total_weight = sum(self.ensemble_weights[m] for m in predictions.keys())
            ensemble_pred /= total_weight
            
            # 성능 평가
            y_pred_binary = (ensemble_pred > 0.5).astype(int)
            
            result = {
                'auc': roc_auc_score(y_val, ensemble_pred),
                'accuracy': accuracy_score(y_val, y_pred_binary),
                'precision': precision_score(y_val, y_pred_binary),
                'recall': recall_score(y_val, y_pred_binary),
                'f1': f1_score(y_val, y_pred_binary)
            }
            
            # 개별 모델 vs 앙상블 비교
            logger.info("\n앙상블 성능 비교:")
            for model_name in predictions.keys():
                model_auc = roc_auc_score(y_val, predictions[model_name])
                logger.info(f"  {model_name}: AUC = {model_auc:.4f}")
            logger.info(f"  Ensemble: AUC = {result['auc']:.4f}")
            
        else:
            result = {}
        
        return result
    
    def predict(self,
               X: pd.DataFrame,
               model_type: str = 'ensemble',
               return_proba: bool = True) -> np.ndarray:
        """예측 수행
        
        Args:
            X: 입력 피처
            model_type: 모델 타입 ('lightgbm', 'xgboost', 'catboost', 'ensemble')
            return_proba: 확률 반환 여부
            
        Returns:
            예측 결과
        """
        
        if model_type == 'ensemble':
            predictions = {}
            
            if 'lightgbm' in self.models:
                predictions['lightgbm'] = self.models['lightgbm'].predict(X)
            
            if 'xgboost' in self.models:
                dmatrix = xgb.DMatrix(X)
                predictions['xgboost'] = self.models['xgboost'].predict(dmatrix)
            
            if 'catboost' in self.models:
                predictions['catboost'] = self.models['catboost'].predict_proba(X)[:, 1]
            
            # 가중 평균
            y_pred = np.zeros(len(X))
            total_weight = 0
            
            for model_name, pred in predictions.items():
                weight = self.ensemble_weights.get(model_name, 1/len(predictions))
                y_pred += weight * pred
                total_weight += weight
            
            y_pred /= total_weight
            
        else:
            if model_type not in self.models:
                raise ValueError(f"모델 '{model_type}'를 찾을 수 없습니다")
            
            if model_type == 'lightgbm':
                y_pred = self.models['lightgbm'].predict(X)
            elif model_type == 'xgboost':
                dmatrix = xgb.DMatrix(X)
                y_pred = self.models['xgboost'].predict(dmatrix)
            elif model_type == 'catboost':
                y_pred = self.models['catboost'].predict_proba(X)[:, 1]
        
        if not return_proba:
            y_pred = (y_pred > 0.5).astype(int)
        
        return y_pred
    
    def _analyze_feature_importance(self):
        """피처 중요도 분석 및 시각화"""
        
        fig, axes = plt.subplots(1, len(self.feature_importance), 
                                figsize=(6*len(self.feature_importance), 8))
        
        if len(self.feature_importance) == 1:
            axes = [axes]
        
        for idx, (model_name, importance_df) in enumerate(self.feature_importance.items()):
            # 상위 20개 피처
            top_features = importance_df.head(20)
            
            axes[idx].barh(range(len(top_features)), 
                          top_features['importance'].values)
            axes[idx].set_yticks(range(len(top_features)))
            axes[idx].set_yticklabels(top_features['feature'].values)
            axes[idx].set_xlabel('Importance')
            axes[idx].set_title(f'{model_name.upper()} Feature Importance')
            axes[idx].invert_yaxis()
        
        plt.tight_layout()
        plt.savefig(self.model_dir / 'feature_importance.png', dpi=100, bbox_inches='tight')
        plt.close()
        
        # 통합 중요도 (앙상블)
        if len(self.feature_importance) > 1:
            combined_importance = pd.DataFrame()
            
            for model_name, importance_df in self.feature_importance.items():
                if combined_importance.empty:
                    combined_importance = importance_df[['feature', 'importance']].copy()
                    combined_importance.columns = ['feature', model_name]
                else:
                    combined_importance = combined_importance.merge(
                        importance_df[['feature', 'importance']],
                        on='feature',
                        how='outer',
                        suffixes=('', f'_{model_name}')
                    ).rename(columns={'importance': model_name})
            
            # 평균 중요도
            model_cols = [col for col in combined_importance.columns if col != 'feature']
            combined_importance['avg_importance'] = combined_importance[model_cols].mean(axis=1)
            combined_importance = combined_importance.sort_values('avg_importance', ascending=False)
            
            # 저장
            combined_importance.to_csv(self.model_dir / 'feature_importance.csv', index=False)
            
            logger.info("\n상위 10개 중요 피처 (평균):")
            for _, row in combined_importance.head(10).iterrows():
                logger.info(f"  {row['feature']}: {row['avg_importance']:.4f}")
    
    def _save_models(self):
        """모델 저장"""
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        for model_name, model in self.models.items():
            model_path = self.model_dir / f'{model_name}_{timestamp}.pkl'
            
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            
            logger.info(f"{model_name} 모델 저장: {model_path}")
        
        # 앙상블 가중치 저장
        if hasattr(self, 'ensemble_weights'):
            weights_path = self.model_dir / f'ensemble_weights_{timestamp}.json'
            with open(weights_path, 'w') as f:
                json.dump(self.ensemble_weights, f)
        
        # 학습 설정 저장
        config = {
            'lgb_params': self.lgb_params,
            'xgb_params': self.xgb_params,
            'cat_params': self.cat_params,
            'timestamp': timestamp
        }
        
        config_path = self.model_dir / f'training_config_{timestamp}.json'
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
    
    def load_models(self, timestamp: str):
        """저장된 모델 로드"""
        
        model_files = {
            'lightgbm': f'lightgbm_{timestamp}.pkl',
            'xgboost': f'xgboost_{timestamp}.pkl',
            'catboost': f'catboost_{timestamp}.pkl'
        }
        
        for model_name, filename in model_files.items():
            model_path = self.model_dir / filename
            if model_path.exists():
                with open(model_path, 'rb') as f:
                    self.models[model_name] = pickle.load(f)
                logger.info(f"{model_name} 모델 로드: {model_path}")
        
        # 앙상블 가중치 로드
        weights_path = self.model_dir / f'ensemble_weights_{timestamp}.json'
        if weights_path.exists():
            with open(weights_path, 'r') as f:
                self.ensemble_weights = json.load(f)
    
    def _print_training_summary(self, results: Dict):
        """학습 결과 요약 출력"""
        
        logger.info("\n" + "="*50)
        logger.info("모델 학습 결과 요약")
        logger.info("="*50)
        
        for model_name, metrics in results.items():
            logger.info(f"\n{model_name.upper()}:")
            for metric_name, value in metrics.items():
                if isinstance(value, float):
                    logger.info(f"  {metric_name}: {value:.4f}")

# CLI 실행용
def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='LightGBM 모델 학습')
    parser.add_argument('--data-dir', default='data/processed', help='데이터 디렉토리')
    parser.add_argument('--model-dir', default='models', help='모델 저장 디렉토리')
    parser.add_argument('--ensemble', action='store_true', help='앙상블 사용')
    parser.add_argument('--cv-folds', type=int, default=5, help='CV 폴드 수')
    parser.add_argument('--embargo', type=int, default=2, help='엠바고 기간')
    
    args = parser.parse_args()
    
    # 데이터 로드
    data_path = Path(args.data_dir)
    X_train = pd.read_parquet(data_path / 'X_train.parquet')
    X_test = pd.read_parquet(data_path / 'X_test.parquet')
    y_train = pd.read_parquet(data_path / 'y_train.parquet')['target']
    y_test = pd.read_parquet(data_path / 'y_test.parquet')['target']
    
    # 학습
    trainer = LightGBMTrainer(
        model_dir=args.model_dir,
        use_ensemble=args.ensemble
    )
    
    results = trainer.train(
        X_train=X_train,
        y_train=y_train,
        X_val=X_test,
        y_val=y_test,
        cv_folds=args.cv_folds,
        embargo_bars=args.embargo
    )

if __name__ == "__main__":
    main()