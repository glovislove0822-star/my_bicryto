"""자기학습 임계값 모듈"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any
from datetime import datetime, timedelta
import logging
from collections import deque
from scipy import stats
import json
import pickle
import warnings
warnings.filterwarnings('ignore')

from ..utils.logging import Logger
from ..utils.io import IOUtils

logger = Logger.get_logger(__name__)

class SelfLearningThreshold:
    """자기학습 적응형 임계값 시스템
    
    베이지안 업데이트와 Thompson Sampling을 사용한 온라인 학습
    """
    
    def __init__(self, 
                 alpha: float = 0.1,
                 exploration_rate: float = 0.1,
                 memory_size: int = 1000):
        """
        Args:
            alpha: 학습률 (스무딩 계수)
            exploration_rate: 탐색 비율
            memory_size: 메모리 버퍼 크기
        """
        self.alpha = alpha
        self.exploration_rate = exploration_rate
        self.memory_size = memory_size
        
        # 베이지안 파라미터 (Beta 분포)
        self.posterior_distributions = {}
        
        # 성과 메모리
        self.performance_memory = deque(maxlen=memory_size)
        
        # 현재 임계값
        self.current_thresholds = {
            'meta_label': 0.6,
            'ofi_long': 0.2,
            'ofi_short': -0.2,
            'spread_max': 1.5,
            'depth_min': 5000
        }
        
        # 임계값 범위
        self.threshold_bounds = {
            'meta_label': (0.3, 0.9),
            'ofi_long': (0.0, 1.0),
            'ofi_short': (-1.0, 0.0),
            'spread_max': (0.5, 3.0),
            'depth_min': (1000, 20000)
        }
        
        # 학습 히스토리
        self.learning_history = []
        
        # Multi-Armed Bandit 설정
        self.mab_arms = {}  # 각 임계값의 암(arm) 정보
        self.mab_rewards = {}  # 보상 히스토리
        
        # 초기화
        self._initialize_distributions()
    
    def _initialize_distributions(self):
        """베이지안 분포 초기화"""
        
        for param_name in self.current_thresholds.keys():
            # Beta 분포 초기화 (균등 prior)
            self.posterior_distributions[param_name] = {
                'alpha': 1.0,  # 성공 수 + 1
                'beta': 1.0,   # 실패 수 + 1
                'n_trials': 0
            }
            
            # MAB 암 초기화
            self.mab_arms[param_name] = {
                'n_pulls': 0,
                'total_reward': 0,
                'avg_reward': 0
            }
            
            self.mab_rewards[param_name] = deque(maxlen=100)
    
    def update(self, 
              recent_trades: List[Dict],
              current_thresholds: Optional[Dict] = None) -> Dict:
        """임계값 업데이트
        
        Args:
            recent_trades: 최근 거래 결과
            current_thresholds: 현재 임계값
            
        Returns:
            업데이트된 임계값
        """
        
        if current_thresholds:
            self.current_thresholds.update(current_thresholds)
        
        # 거래 결과 분석
        trade_analysis = self._analyze_trades(recent_trades)
        
        # 베이지안 업데이트
        self._bayesian_update(trade_analysis)
        
        # Thompson Sampling으로 새 임계값 샘플링
        new_thresholds = self._thompson_sampling()
        
        # 탐색 vs 활용 결정
        if np.random.random() < self.exploration_rate:
            # 탐색: 랜덤 perturbation
            new_thresholds = self._exploration_step(new_thresholds)
        
        # 스무딩 적용
        smoothed_thresholds = self._smooth_thresholds(new_thresholds)
        
        # 제약 조건 체크
        validated_thresholds = self._validate_thresholds(smoothed_thresholds)
        
        # 업데이트 기록
        self._record_update(validated_thresholds, trade_analysis)
        
        self.current_thresholds = validated_thresholds
        
        return validated_thresholds
    
    def _analyze_trades(self, recent_trades: List[Dict]) -> Dict:
        """거래 결과 분석"""
        
        if not recent_trades:
            return {}
        
        analysis = {
            'total_trades': len(recent_trades),
            'profitable_trades': 0,
            'avg_pnl': 0,
            'sharpe_ratio': 0,
            'win_rate': 0,
            'threshold_effectiveness': {}
        }
        
        pnls = []
        
        for trade in recent_trades:
            pnl = trade.get('pnl', 0)
            pnls.append(pnl)
            
            if pnl > 0:
                analysis['profitable_trades'] += 1
            
            # 임계값별 효과성 분석
            for param in self.current_thresholds.keys():
                if param not in analysis['threshold_effectiveness']:
                    analysis['threshold_effectiveness'][param] = {
                        'hits': 0,
                        'misses': 0,
                        'total_pnl': 0
                    }
                
                # 메타 라벨 관련
                if param == 'meta_label':
                    meta_prob = trade.get('meta_label_prob', 0.5)
                    if meta_prob >= self.current_thresholds['meta_label']:
                        if pnl > 0:
                            analysis['threshold_effectiveness'][param]['hits'] += 1
                        else:
                            analysis['threshold_effectiveness'][param]['misses'] += 1
                        analysis['threshold_effectiveness'][param]['total_pnl'] += pnl
                
                # OFI 관련
                elif param in ['ofi_long', 'ofi_short']:
                    ofi = trade.get('ofi_z', 0)
                    if param == 'ofi_long' and ofi >= self.current_thresholds['ofi_long']:
                        if trade.get('side') == 'long' and pnl > 0:
                            analysis['threshold_effectiveness'][param]['hits'] += 1
                        else:
                            analysis['threshold_effectiveness'][param]['misses'] += 1
                    elif param == 'ofi_short' and ofi <= self.current_thresholds['ofi_short']:
                        if trade.get('side') == 'short' and pnl > 0:
                            analysis['threshold_effectiveness'][param]['hits'] += 1
                        else:
                            analysis['threshold_effectiveness'][param]['misses'] += 1
        
        # 통계 계산
        if pnls:
            analysis['avg_pnl'] = np.mean(pnls)
            analysis['win_rate'] = analysis['profitable_trades'] / len(pnls)
            
            if np.std(pnls) > 0:
                analysis['sharpe_ratio'] = np.mean(pnls) / np.std(pnls) * np.sqrt(252)
        
        return analysis
    
    def _bayesian_update(self, trade_analysis: Dict):
        """베이지안 파라미터 업데이트"""
        
        if not trade_analysis or 'threshold_effectiveness' not in trade_analysis:
            return
        
        for param, effectiveness in trade_analysis['threshold_effectiveness'].items():
            if param not in self.posterior_distributions:
                continue
            
            hits = effectiveness.get('hits', 0)
            misses = effectiveness.get('misses', 0)
            
            if hits + misses > 0:
                # Beta 분포 업데이트
                self.posterior_distributions[param]['alpha'] += hits
                self.posterior_distributions[param]['beta'] += misses
                self.posterior_distributions[param]['n_trials'] += hits + misses
                
                # MAB 업데이트
                reward = hits / (hits + misses)
                self.mab_arms[param]['n_pulls'] += 1
                self.mab_arms[param]['total_reward'] += reward
                self.mab_arms[param]['avg_reward'] = (
                    self.mab_arms[param]['total_reward'] / 
                    self.mab_arms[param]['n_pulls']
                )
                self.mab_rewards[param].append(reward)
                
                logger.debug(f"{param} 베이지안 업데이트: "
                           f"α={self.posterior_distributions[param]['alpha']:.2f}, "
                           f"β={self.posterior_distributions[param]['beta']:.2f}")
    
    def _thompson_sampling(self) -> Dict:
        """Thompson Sampling으로 새 임계값 샘플링"""
        
        new_thresholds = {}
        
        for param in self.current_thresholds.keys():
            dist = self.posterior_distributions[param]
            
            # Beta 분포에서 샘플링
            if dist['n_trials'] > 0:
                # Thompson Sampling
                sample = np.random.beta(dist['alpha'], dist['beta'])
                
                # 임계값 범위로 변환
                bounds = self.threshold_bounds[param]
                new_value = bounds[0] + sample * (bounds[1] - bounds[0])
            else:
                # 경험이 없으면 현재 값 유지
                new_value = self.current_thresholds[param]
            
            new_thresholds[param] = new_value
        
        return new_thresholds
    
    def _exploration_step(self, thresholds: Dict) -> Dict:
        """탐색 단계 (랜덤 perturbation)"""
        
        explored_thresholds = {}
        
        for param, value in thresholds.items():
            bounds = self.threshold_bounds[param]
            
            # 가우시안 노이즈 추가
            noise_scale = (bounds[1] - bounds[0]) * 0.1  # 범위의 10%
            noise = np.random.normal(0, noise_scale)
            
            new_value = value + noise
            
            # 범위 제한
            new_value = np.clip(new_value, bounds[0], bounds[1])
            
            explored_thresholds[param] = new_value
        
        return explored_thresholds
    
    def _smooth_thresholds(self, new_thresholds: Dict) -> Dict:
        """임계값 스무딩 (급격한 변화 방지)"""
        
        smoothed = {}
        
        for param, new_value in new_thresholds.items():
            old_value = self.current_thresholds[param]
            
            # 지수 이동 평균
            smoothed_value = self.alpha * new_value + (1 - self.alpha) * old_value
            
            smoothed[param] = smoothed_value
        
        return smoothed
    
    def _validate_thresholds(self, thresholds: Dict) -> Dict:
        """임계값 검증 및 제약 조건 적용"""
        
        validated = {}
        
        for param, value in thresholds.items():
            bounds = self.threshold_bounds[param]
            
            # 범위 제한
            validated_value = np.clip(value, bounds[0], bounds[1])
            
            # 특수 제약 조건
            if param == 'ofi_long' and 'ofi_short' in validated:
                # ofi_long은 ofi_short보다 커야 함
                if validated_value <= validated['ofi_short']:
                    validated_value = validated['ofi_short'] + 0.1
            
            validated[param] = validated_value
        
        return validated
    
    def _record_update(self, new_thresholds: Dict, trade_analysis: Dict):
        """업데이트 기록"""
        
        record = {
            'timestamp': datetime.now(),
            'old_thresholds': self.current_thresholds.copy(),
            'new_thresholds': new_thresholds.copy(),
            'trade_analysis': trade_analysis,
            'posterior_params': {
                param: {
                    'alpha': dist['alpha'],
                    'beta': dist['beta'],
                    'n_trials': dist['n_trials']
                }
                for param, dist in self.posterior_distributions.items()
            }
        }
        
        self.learning_history.append(record)
        
        # 메모리에 추가
        if trade_analysis:
            self.performance_memory.append({
                'timestamp': datetime.now(),
                'win_rate': trade_analysis.get('win_rate', 0),
                'avg_pnl': trade_analysis.get('avg_pnl', 0),
                'sharpe_ratio': trade_analysis.get('sharpe_ratio', 0)
            })
    
    def get_confidence_intervals(self, confidence: float = 0.95) -> Dict:
        """임계값 신뢰 구간 계산"""
        
        intervals = {}
        
        for param in self.current_thresholds.keys():
            dist = self.posterior_distributions[param]
            
            if dist['n_trials'] > 0:
                # Beta 분포의 신뢰 구간
                alpha = dist['alpha']
                beta = dist['beta']
                
                # 분위수 계산
                lower = stats.beta.ppf((1 - confidence) / 2, alpha, beta)
                upper = stats.beta.ppf((1 + confidence) / 2, alpha, beta)
                
                # 임계값 범위로 변환
                bounds = self.threshold_bounds[param]
                lower_threshold = bounds[0] + lower * (bounds[1] - bounds[0])
                upper_threshold = bounds[0] + upper * (bounds[1] - bounds[0])
                
                intervals[param] = {
                    'current': self.current_thresholds[param],
                    'lower': lower_threshold,
                    'upper': upper_threshold,
                    'confidence': confidence
                }
            else:
                intervals[param] = {
                    'current': self.current_thresholds[param],
                    'lower': self.threshold_bounds[param][0],
                    'upper': self.threshold_bounds[param][1],
                    'confidence': 0
                }
        
        return intervals
    
    def get_learning_statistics(self) -> Dict:
        """학습 통계"""
        
        stats = {
            'total_updates': len(self.learning_history),
            'current_thresholds': self.current_thresholds.copy(),
            'performance_trend': None,
            'convergence_status': {},
            'best_performing_params': {}
        }
        
        # 성과 트렌드
        if len(self.performance_memory) >= 10:
            recent_performance = list(self.performance_memory)[-10:]
            
            win_rates = [p['win_rate'] for p in recent_performance]
            pnls = [p['avg_pnl'] for p in recent_performance]
            
            # 선형 회귀로 트렌드 계산
            x = np.arange(len(win_rates))
            
            win_rate_trend = np.polyfit(x, win_rates, 1)[0]
            pnl_trend = np.polyfit(x, pnls, 1)[0]
            
            stats['performance_trend'] = {
                'win_rate_trend': win_rate_trend,
                'pnl_trend': pnl_trend,
                'improving': win_rate_trend > 0 and pnl_trend > 0
            }
        
        # 수렴 상태
        for param in self.current_thresholds.keys():
            if len(self.learning_history) >= 20:
                # 최근 20개 업데이트의 변화량
                recent_values = [
                    h['new_thresholds'][param] 
                    for h in self.learning_history[-20:]
                ]
                
                std = np.std(recent_values)
                mean_change = np.mean(np.abs(np.diff(recent_values)))
                
                stats['convergence_status'][param] = {
                    'std': std,
                    'mean_change': mean_change,
                    'converged': std < 0.01 and mean_change < 0.005
                }
        
        # 최고 성과 파라미터
        for param, arm in self.mab_arms.items():
            if arm['n_pulls'] > 0:
                stats['best_performing_params'][param] = {
                    'avg_reward': arm['avg_reward'],
                    'n_pulls': arm['n_pulls'],
                    'recent_rewards': list(self.mab_rewards[param])[-10:]
                }
        
        return stats
    
    def save_state(self, filepath: str):
        """상태 저장"""
        
        state = {
            'current_thresholds': self.current_thresholds,
            'posterior_distributions': self.posterior_distributions,
            'mab_arms': self.mab_arms,
            'learning_history': self.learning_history[-100:],  # 최근 100개만
            'performance_memory': list(self.performance_memory),
            'timestamp': datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2, default=str)
        
        logger.info(f"학습 상태 저장: {filepath}")
    
    def load_state(self, filepath: str):
        """상태 로드"""
        
        with open(filepath, 'r') as f:
            state = json.load(f)
        
        self.current_thresholds = state['current_thresholds']
        self.posterior_distributions = state['posterior_distributions']
        self.mab_arms = state['mab_arms']
        self.learning_history = state['learning_history']
        self.performance_memory = deque(state['performance_memory'], 
                                       maxlen=self.memory_size)
        
        logger.info(f"학습 상태 로드: {filepath}")
    
    def reset_param(self, param: str):
        """특정 파라미터 리셋"""
        
        if param in self.current_thresholds:
            # 기본값으로 리셋
            bounds = self.threshold_bounds[param]
            self.current_thresholds[param] = (bounds[0] + bounds[1]) / 2
            
            # 베이지안 분포 리셋
            self.posterior_distributions[param] = {
                'alpha': 1.0,
                'beta': 1.0,
                'n_trials': 0
            }
            
            # MAB 리셋
            self.mab_arms[param] = {
                'n_pulls': 0,
                'total_reward': 0,
                'avg_reward': 0
            }
            
            self.mab_rewards[param].clear()
            
            logger.info(f"파라미터 {param} 리셋 완료")

# CLI 실행용
def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='자기학습 임계값')
    parser.add_argument('--trades', help='거래 결과 파일')
    parser.add_argument('--alpha', type=float, default=0.1, help='학습률')
    parser.add_argument('--exploration', type=float, default=0.1, help='탐색 비율')
    parser.add_argument('--save', help='상태 저장 경로')
    parser.add_argument('--load', help='상태 로드 경로')
    
    args = parser.parse_args()
    
    # 학습기 생성
    learner = SelfLearningThreshold(
        alpha=args.alpha,
        exploration_rate=args.exploration
    )
    
    # 상태 로드
    if args.load:
        learner.load_state(args.load)
    
    # 거래 데이터 로드
    if args.trades:
        trades_df = pd.read_csv(args.trades)
        recent_trades = trades_df.to_dict('records')
        
        # 임계값 업데이트
        new_thresholds = learner.update(recent_trades)
        
        print("\n업데이트된 임계값:")
        for param, value in new_thresholds.items():
            print(f"  {param}: {value:.4f}")
        
        # 신뢰 구간
        intervals = learner.get_confidence_intervals()
        print("\n신뢰 구간 (95%):")
        for param, interval in intervals.items():
            print(f"  {param}: [{interval['lower']:.4f}, {interval['upper']:.4f}]")
    
    # 학습 통계
    stats = learner.get_learning_statistics()
    print("\n학습 통계:")
    print(f"  총 업데이트: {stats['total_updates']}")
    if stats['performance_trend']:
        print(f"  성과 개선 중: {stats['performance_trend']['improving']}")
    
    # 상태 저장
    if args.save:
        learner.save_state(args.save)

if __name__ == "__main__":
    main()