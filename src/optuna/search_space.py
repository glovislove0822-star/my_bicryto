"""Optuna 검색 공간 정의"""

import optuna
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import logging

from ..utils.logging import Logger

logger = Logger.get_logger(__name__)

class SearchSpace:
    """하이퍼파라미터 검색 공간
    
    전략 파라미터의 검색 범위 정의 및 샘플링
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Args:
            config: 검색 공간 설정
        """
        self.config = config or {}
        
        # 기본 검색 공간
        self.default_space = {
            # 추세 파라미터
            'trend': {
                'donchian_n': {'type': 'int', 'low': 20, 'high': 120, 'step': 5},
                'mom_k': {'type': 'int', 'low': 30, 'high': 180, 'step': 10},
                'ema_fast': {'type': 'int', 'low': 5, 'high': 30, 'step': 1},
                'ema_slow': {'type': 'int', 'low': 20, 'high': 120, 'step': 5},
                'adx_period': {'type': 'int', 'low': 10, 'high': 30, 'step': 2},
                'adx_threshold': {'type': 'float', 'low': 20, 'high': 40, 'step': 2}
            },
            
            # 엔트리 파라미터
            'entry': {
                'rsi_len': {'type': 'int', 'low': 2, 'high': 6, 'step': 1},
                'rsi_buy': {'type': 'int', 'low': 5, 'high': 30, 'step': 5},
                'rsi_sell': {'type': 'int', 'low': 70, 'high': 95, 'step': 5},
                'vwap_z_entry': {'type': 'float', 'low': 0.2, 'high': 1.5, 'step': 0.1},
                'bb_period': {'type': 'int', 'low': 10, 'high': 30, 'step': 2},
                'bb_std': {'type': 'float', 'low': 1.5, 'high': 3.0, 'step': 0.25}
            },
            
            # 게이팅 파라미터
            'gating': {
                'ofi_z_th_long': {'type': 'float', 'low': 0.0, 'high': 1.0, 'step': 0.05},
                'ofi_z_th_short': {'type': 'float', 'low': -1.0, 'high': 0.0, 'step': 0.05},
                'spread_bp_max': {'type': 'float', 'low': 0.5, 'high': 3.0, 'step': 0.25},
                'depth_min': {'type': 'int', 'low': 1000, 'high': 20000, 'step': 1000},
                'qi_threshold': {'type': 'float', 'low': 0.4, 'high': 0.6, 'step': 0.02},
                'volume_spike': {'type': 'float', 'low': 1.5, 'high': 3.0, 'step': 0.25}
            },
            
            # 리스크 파라미터
            'risk': {
                'target_vol': {'type': 'float', 'low': 0.15, 'high': 0.50, 'step': 0.05},
                'tp_atr': {'type': 'float', 'low': 0.8, 'high': 2.0, 'step': 0.1},
                'sl_atr': {'type': 'float', 'low': 0.4, 'high': 1.2, 'step': 0.1},
                'trail_atr': {'type': 'float', 'low': 0.0, 'high': 1.5, 'step': 0.1},
                'tmax_bars': {'type': 'int', 'low': 20, 'high': 80, 'step': 5},
                'position_ratio': {'type': 'float', 'low': 0.05, 'high': 0.3, 'step': 0.05},
                'max_positions': {'type': 'int', 'low': 1, 'high': 10, 'step': 1}
            },
            
            # 집행 파라미터
            'execution': {
                'n_splits': {'type': 'categorical', 'choices': [1, 2, 3]},
                'split_ratio_pattern': {'type': 'categorical', 
                                       'choices': ['equal', '60_40', '50_30_20', '40_30_30']},
                'pyramid_interval': {'type': 'int', 'low': 5, 'high': 60, 'step': 5},
                'use_iceberg': {'type': 'categorical', 'choices': [True, False]},
                'post_only': {'type': 'categorical', 'choices': [True, False]}
            },
            
            # 메타라벨 파라미터
            'meta_label': {
                'use': {'type': 'categorical', 'choices': [True, False]},
                'p_threshold': {'type': 'float', 'low': 0.50, 'high': 0.75, 'step': 0.05},
                'min_confidence': {'type': 'float', 'low': 0.4, 'high': 0.7, 'step': 0.05},
                'calibration': {'type': 'categorical', 'choices': ['isotonic', 'sigmoid', None]}
            },
            
            # v2.0 Enhanced 파라미터
            'regime': {
                'enabled': {'type': 'categorical', 'choices': [True, False]},
                'vol_multiplier': {'type': 'float', 'low': 0.5, 'high': 2.0, 'step': 0.1},
                'trend_multiplier': {'type': 'float', 'low': 0.8, 'high': 1.5, 'step': 0.1},
                'liquidity_multiplier': {'type': 'float', 'low': 0.7, 'high': 1.3, 'step': 0.1}
            },
            
            'funding': {
                'enabled': {'type': 'categorical', 'choices': [True, False]},
                'funding_z_threshold': {'type': 'float', 'low': 1.5, 'high': 3.0, 'step': 0.25},
                'funding_harvest_min': {'type': 'float', 'low': 0.005, 'high': 0.02, 'step': 0.0025},
                'funding_momentum_window': {'type': 'int', 'low': 10, 'high': 40, 'step': 5}
            },
            
            'scalping': {
                'enabled': {'type': 'categorical', 'choices': [True, False]},
                'max_spread_bps': {'type': 'float', 'low': 0.3, 'high': 1.0, 'step': 0.1},
                'min_depth_imbalance': {'type': 'float', 'low': 0.6, 'high': 0.8, 'step': 0.05},
                'tick_momentum_threshold': {'type': 'float', 'low': 0.0005, 'high': 0.002, 'step': 0.0002}
            },
            
            'portfolio': {
                'kelly_fraction': {'type': 'float', 'low': 0.1, 'high': 0.5, 'step': 0.05},
                'max_correlation': {'type': 'float', 'low': 0.5, 'high': 0.9, 'step': 0.1},
                'rebalance_hours': {'type': 'int', 'low': 1, 'high': 12, 'step': 1},
                'risk_parity': {'type': 'categorical', 'choices': [True, False]}
            }
        }
        
        # 사용자 정의 공간으로 업데이트
        if config:
            self._update_space(config)
        
        # 조건부 파라미터
        self.conditional_params = self._define_conditional_params()
        
        # 파라미터 그룹
        self.param_groups = self._define_param_groups()
    
    def _update_space(self, custom_space: Dict):
        """사용자 정의 검색 공간으로 업데이트"""
        
        for category, params in custom_space.items():
            if category not in self.default_space:
                self.default_space[category] = {}
            
            for param_name, param_config in params.items():
                self.default_space[category][param_name] = param_config
    
    def _define_conditional_params(self) -> Dict:
        """조건부 파라미터 정의"""
        
        return {
            # 메타라벨 사용 시에만
            'meta_label_conditional': {
                'condition': lambda trial: trial.params.get('meta_label_use', False),
                'params': ['p_threshold', 'min_confidence', 'calibration']
            },
            
            # 펀딩 전략 사용 시에만
            'funding_conditional': {
                'condition': lambda trial: trial.params.get('funding_enabled', False),
                'params': ['funding_z_threshold', 'funding_harvest_min', 'funding_momentum_window']
            },
            
            # 스캘핑 사용 시에만
            'scalping_conditional': {
                'condition': lambda trial: trial.params.get('scalping_enabled', False),
                'params': ['max_spread_bps', 'min_depth_imbalance', 'tick_momentum_threshold']
            },
            
            # 트레일링 스탑 사용 시에만
            'trailing_conditional': {
                'condition': lambda trial: trial.params.get('trail_atr', 0) > 0,
                'params': ['trail_activation', 'trail_step']
            }
        }
    
    def _define_param_groups(self) -> Dict:
        """파라미터 그룹 정의 (상호 의존성)"""
        
        return {
            'trend_consistency': {
                'params': ['ema_fast', 'ema_slow'],
                'constraint': lambda p: p['ema_fast'] < p['ema_slow']
            },
            
            'risk_consistency': {
                'params': ['tp_atr', 'sl_atr'],
                'constraint': lambda p: p['tp_atr'] > p['sl_atr']
            },
            
            'ofi_consistency': {
                'params': ['ofi_z_th_long', 'ofi_z_th_short'],
                'constraint': lambda p: p['ofi_z_th_long'] > p['ofi_z_th_short']
            },
            
            'rsi_consistency': {
                'params': ['rsi_buy', 'rsi_sell'],
                'constraint': lambda p: p['rsi_buy'] < p['rsi_sell']
            }
        }
    
    def sample_parameters(self, trial: optuna.Trial) -> Dict:
        """파라미터 샘플링
        
        Args:
            trial: Optuna trial 객체
            
        Returns:
            샘플링된 파라미터
        """
        
        params = {}
        
        # 각 카테고리별 샘플링
        for category, category_params in self.default_space.items():
            for param_name, param_config in category_params.items():
                full_param_name = f"{category}_{param_name}"
                
                # 조건부 파라미터 체크
                if self._is_conditional_param(full_param_name, trial):
                    continue
                
                # 파라미터 타입별 샘플링
                if param_config['type'] == 'int':
                    value = trial.suggest_int(
                        full_param_name,
                        param_config['low'],
                        param_config['high'],
                        step=param_config.get('step', 1)
                    )
                    
                elif param_config['type'] == 'float':
                    if 'step' in param_config:
                        value = trial.suggest_float(
                            full_param_name,
                            param_config['low'],
                            param_config['high'],
                            step=param_config['step']
                        )
                    else:
                        value = trial.suggest_float(
                            full_param_name,
                            param_config['low'],
                            param_config['high']
                        )
                    
                elif param_config['type'] == 'categorical':
                    value = trial.suggest_categorical(
                        full_param_name,
                        param_config['choices']
                    )
                    
                elif param_config['type'] == 'loguniform':
                    value = trial.suggest_loguniform(
                        full_param_name,
                        param_config['low'],
                        param_config['high']
                    )
                
                else:
                    continue
                
                params[full_param_name] = value
        
        # 파라미터 그룹 제약 적용
        params = self._apply_group_constraints(params, trial)
        
        # 동적 파라미터 추가
        params = self._add_dynamic_params(params, trial)
        
        return params
    
    def _is_conditional_param(self, param_name: str, trial: optuna.Trial) -> bool:
        """조건부 파라미터 여부 확인"""
        
        for cond_name, cond_config in self.conditional_params.items():
            if param_name in cond_config['params']:
                if not cond_config['condition'](trial):
                    return True
        
        return False
    
    def _apply_group_constraints(self, params: Dict, trial: optuna.Trial) -> Dict:
        """파라미터 그룹 제약 적용"""
        
        for group_name, group_config in self.param_groups.items():
            group_params = {
                p: params.get(p) 
                for p in group_config['params']
                if p in params
            }
            
            if all(v is not None for v in group_params.values()):
                # 제약 조건 체크
                if not group_config['constraint'](group_params):
                    # 제약 위반 시 조정
                    params = self._adjust_for_constraint(
                        params, group_name, group_params, trial
                    )
        
        return params
    
    def _adjust_for_constraint(self,
                              params: Dict,
                              group_name: str,
                              group_params: Dict,
                              trial: optuna.Trial) -> Dict:
        """제약 조건 위반 시 파라미터 조정"""
        
        if group_name == 'trend_consistency':
            # ema_fast < ema_slow 보장
            if params.get('trend_ema_fast', 0) >= params.get('trend_ema_slow', 0):
                params['trend_ema_slow'] = params['trend_ema_fast'] + 10
                
        elif group_name == 'risk_consistency':
            # tp_atr > sl_atr 보장
            if params.get('risk_tp_atr', 0) <= params.get('risk_sl_atr', 0):
                params['risk_tp_atr'] = params['risk_sl_atr'] + 0.2
                
        elif group_name == 'ofi_consistency':
            # ofi_z_th_long > ofi_z_th_short 보장
            if params.get('gating_ofi_z_th_long', 0) <= params.get('gating_ofi_z_th_short', 0):
                params['gating_ofi_z_th_long'] = abs(params['gating_ofi_z_th_short']) + 0.1
                
        elif group_name == 'rsi_consistency':
            # rsi_buy < rsi_sell 보장
            if params.get('entry_rsi_buy', 0) >= params.get('entry_rsi_sell', 0):
                params['entry_rsi_sell'] = params['entry_rsi_buy'] + 40
        
        return params
    
    def _add_dynamic_params(self, params: Dict, trial: optuna.Trial) -> Dict:
        """동적 파라미터 추가"""
        
        # 시장 상황에 따른 동적 파라미터
        if params.get('regime_enabled', False):
            # 레짐별 세부 파라미터
            params['regime_vol_low_mult'] = trial.suggest_float(
                'regime_vol_low_mult', 0.5, 1.0
            )
            params['regime_vol_high_mult'] = trial.suggest_float(
                'regime_vol_high_mult', 1.0, 2.0
            )
            params['regime_trend_strong_mult'] = trial.suggest_float(
                'regime_trend_strong_mult', 1.0, 1.5
            )
        
        # 적응형 임계값
        if trial.suggest_categorical('use_adaptive_thresholds', [True, False]):
            params['adaptive_alpha'] = trial.suggest_float(
                'adaptive_alpha', 0.05, 0.2
            )
            params['adaptive_lookback'] = trial.suggest_int(
                'adaptive_lookback', 100, 1000, step=100
            )
        
        return params
    
    def get_parameter_importance(self, study: optuna.Study) -> Dict:
        """파라미터 중요도 계산
        
        Args:
            study: 완료된 Optuna study
            
        Returns:
            파라미터별 중요도
        """
        
        if len(study.trials) < 10:
            return {}
        
        # Optuna의 파라미터 중요도 계산
        importance = optuna.importance.get_param_importances(study)
        
        # 카테고리별 그룹화
        categorized_importance = {}
        
        for param_name, score in importance.items():
            # 카테고리 추출
            if '_' in param_name:
                category = param_name.split('_')[0]
                param = '_'.join(param_name.split('_')[1:])
            else:
                category = 'other'
                param = param_name
            
            if category not in categorized_importance:
                categorized_importance[category] = {}
            
            categorized_importance[category][param] = score
        
        return categorized_importance
    
    def suggest_next_params(self,
                           study: optuna.Study,
                           n_startup_trials: int = 10) -> Optional[Dict]:
        """다음 파라미터 제안 (가이드 샘플링)
        
        Args:
            study: 진행 중인 Optuna study
            n_startup_trials: 초기 랜덤 샘플링 수
            
        Returns:
            제안된 파라미터 또는 None
        """
        
        if len(study.trials) < n_startup_trials:
            # 초기에는 랜덤 샘플링
            return None
        
        # 최고 성능 trials 분석
        best_trials = sorted(
            study.trials,
            key=lambda t: t.value if t.value is not None else float('-inf'),
            reverse=True
        )[:5]
        
        if not best_trials:
            return None
        
        # 최고 성능 파라미터의 평균/중앙값
        suggested_params = {}
        
        for param_name in best_trials[0].params.keys():
            values = [t.params.get(param_name) for t in best_trials 
                     if param_name in t.params]
            
            if not values:
                continue
            
            if isinstance(values[0], (int, float)):
                # 수치형: 중앙값 사용
                suggested_params[param_name] = np.median(values)
            else:
                # 범주형: 최빈값 사용
                from collections import Counter
                most_common = Counter(values).most_common(1)[0][0]
                suggested_params[param_name] = most_common
        
        return suggested_params
    
    def export_best_params(self, study: optuna.Study, top_n: int = 5) -> List[Dict]:
        """최고 성능 파라미터 추출
        
        Args:
            study: 완료된 Optuna study
            top_n: 상위 N개 trials
            
        Returns:
            최고 성능 파라미터 리스트
        """
        
        best_trials = sorted(
            [t for t in study.trials if t.value is not None],
            key=lambda t: t.value,
            reverse=True
        )[:top_n]
        
        best_params = []
        
        for trial in best_trials:
            # 카테고리별로 정리
            organized_params = {}
            
            for param_name, value in trial.params.items():
                if '_' in param_name:
                    category = param_name.split('_')[0]
                    param = '_'.join(param_name.split('_')[1:])
                else:
                    category = 'other'
                    param = param_name
                
                if category not in organized_params:
                    organized_params[category] = {}
                
                organized_params[category][param] = value
            
            best_params.append({
                'trial_number': trial.number,
                'value': trial.value,
                'params': organized_params,
                'user_attrs': trial.user_attrs
            })
        
        return best_params
    
    def validate_params(self, params: Dict) -> Tuple[bool, List[str]]:
        """파라미터 유효성 검증
        
        Args:
            params: 검증할 파라미터
            
        Returns:
            (유효 여부, 오류 메시지 리스트)
        """
        
        errors = []
        
        # 필수 파라미터 체크
        required_params = [
            'trend_donchian_n',
            'risk_target_vol',
            'risk_tp_atr',
            'risk_sl_atr'
        ]
        
        for param in required_params:
            if param not in params:
                errors.append(f"필수 파라미터 누락: {param}")
        
        # 범위 체크
        for param_name, value in params.items():
            # 카테고리와 파라미터 분리
            if '_' in param_name:
                category = param_name.split('_')[0]
                param = '_'.join(param_name.split('_')[1:])
                
                if category in self.default_space and param in self.default_space[category]:
                    config = self.default_space[category][param]
                    
                    if config['type'] in ['int', 'float']:
                        if value < config['low'] or value > config['high']:
                            errors.append(
                                f"{param_name}: 범위 벗어남 "
                                f"({value} not in [{config['low']}, {config['high']}])"
                            )
                    
                    elif config['type'] == 'categorical':
                        if value not in config['choices']:
                            errors.append(
                                f"{param_name}: 유효하지 않은 선택 "
                                f"({value} not in {config['choices']})"
                            )
        
        # 그룹 제약 체크
        for group_name, group_config in self.param_groups.items():
            group_params = {
                p: params.get(p)
                for p in group_config['params']
            }
            
            if all(v is not None for v in group_params.values()):
                if not group_config['constraint'](group_params):
                    errors.append(f"그룹 제약 위반: {group_name}")
        
        return len(errors) == 0, errors