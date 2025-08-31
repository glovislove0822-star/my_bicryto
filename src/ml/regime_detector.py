"""적응형 마켓 레짐 감지 모듈"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any
from datetime import datetime
import logging
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from ..utils.logging import Logger
from ..utils.math import MathUtils

logger = Logger.get_logger(__name__)

class AdaptiveRegimeDetector:
    """적응형 마켓 레짐 감지기
    
    Hidden Markov Model과 클러스터링을 사용한 시장 상태 분류
    """
    
    def __init__(self, n_regimes: int = 4):
        """
        Args:
            n_regimes: 레짐 수 (기본 4: low_vol, normal, high_vol, extreme)
        """
        self.n_regimes = n_regimes
        self.math_utils = MathUtils()
        
        # HMM 모델
        self.hmm_model = hmm.GaussianHMM(
            n_components=n_regimes,
            covariance_type="full",
            n_iter=100,
            random_state=42
        )
        
        # 스케일러
        self.scaler = StandardScaler()
        
        # 레짐 파라미터
        self.regime_params = {}
        self.current_regime = None
        self.regime_history = []
        
        # 레짐 라벨
        self.regime_labels = {
            0: 'low_vol',
            1: 'normal',
            2: 'high_vol',
            3: 'extreme'
        }
        
        # 트렌드 라벨
        self.trend_labels = {
            0: 'strong_down',
            1: 'weak_down',
            2: 'neutral',
            3: 'weak_up',
            4: 'strong_up'
        }
    
    def fit(self, data: pd.DataFrame, features: Optional[List[str]] = None):
        """레짐 모델 학습
        
        Args:
            data: 학습 데이터
            features: 사용할 피처 리스트
        """
        
        if features is None:
            features = self._get_default_features()
        
        # 피처 추출
        X = self._extract_features(data, features)
        
        # 스케일링
        X_scaled = self.scaler.fit_transform(X)
        
        # HMM 학습
        logger.info(f"HMM 학습 시작 (n_regimes={self.n_regimes})")
        self.hmm_model.fit(X_scaled)
        
        # 레짐 파라미터 추출
        self._extract_regime_parameters(X_scaled)
        
        # 레짐 라벨링
        self._label_regimes(X)
        
        logger.info("레짐 모델 학습 완료")
    
    def detect(self, data: pd.DataFrame) -> Dict:
        """현재 레짐 감지
        
        Args:
            data: 현재 데이터
            
        Returns:
            레짐 정보 딕셔너리
        """
        
        # 피처 추출
        features = self._get_default_features()
        X = self._extract_features(data, features)
        
        # 스케일링
        X_scaled = self.scaler.transform(X)
        
        # 레짐 예측
        regimes = self.hmm_model.predict(X_scaled)
        current_regime_idx = regimes[-1]
        
        # 상세 분석
        vol_regime = self._detect_volatility_regime(data)
        trend_regime = self._detect_trend_regime(data)
        liquidity_regime = self._detect_liquidity_regime(data)
        
        # 파라미터 조정 계수
        multipliers = self._calculate_multipliers(
            vol_regime, trend_regime, liquidity_regime
        )
        
        # 레짐 전환 확률
        transition_probs = self._calculate_transition_probabilities(X_scaled)
        
        result = {
            'regime_idx': current_regime_idx,
            'vol_state': vol_regime,
            'trend_state': trend_regime,
            'liquidity_state': liquidity_regime,
            'params_multiplier': multipliers,
            'transition_probs': transition_probs,
            'confidence': self._calculate_regime_confidence(X_scaled)
        }
        
        # 히스토리 업데이트
        self.current_regime = result
        self.regime_history.append({
            'timestamp': datetime.now(),
            'regime': result
        })
        
        return result
    
    def _get_default_features(self) -> List[str]:
        """기본 피처 리스트"""
        return [
            'returns',
            'volatility',
            'volume',
            'spread',
            'trend_strength'
        ]
    
    def _extract_features(self, data: pd.DataFrame, features: List[str]) -> np.ndarray:
        """피처 추출"""
        
        feature_data = []
        
        # 수익률
        if 'returns' in features:
            returns = data['close'].pct_change()
            feature_data.append(returns.fillna(0))
        
        # 변동성
        if 'volatility' in features:
            if 'realized_vol' in data.columns:
                vol = data['realized_vol']
            else:
                vol = returns.rolling(20).std()
            feature_data.append(vol.fillna(vol.mean()))
        
        # 거래량
        if 'volume' in features:
            if 'volume' in data.columns:
                volume_ratio = data['volume'] / data['volume'].rolling(20).mean()
                feature_data.append(volume_ratio.fillna(1))
        
        # 스프레드
        if 'spread' in features:
            if 'spread_bps' in data.columns:
                spread = data['spread_bps']
            else:
                spread = ((data['high'] - data['low']) / data['close'] * 10000)
            feature_data.append(spread.fillna(spread.mean()))
        
        # 트렌드 강도
        if 'trend_strength' in features:
            if 'adx' in data.columns:
                trend = data['adx']
            else:
                # 간단한 트렌드 추정
                sma_short = data['close'].rolling(10).mean()
                sma_long = data['close'].rolling(30).mean()
                trend = abs(sma_short - sma_long) / sma_long * 100
            feature_data.append(trend.fillna(0))
        
        # 배열로 변환
        X = np.column_stack(feature_data)
        
        return X
    
    def _detect_volatility_regime(self, data: pd.DataFrame) -> str:
        """변동성 레짐 감지"""
        
        if 'realized_vol' in data.columns:
            current_vol = data['realized_vol'].iloc[-1]
            vol_percentile = (data['realized_vol'] <= current_vol).mean()
        else:
            returns = data['close'].pct_change()
            current_vol = returns.iloc[-20:].std()
            historical_vol = returns.rolling(20).std()
            vol_percentile = (historical_vol <= current_vol).mean()
        
        if vol_percentile < 0.25:
            return 'low'
        elif vol_percentile < 0.5:
            return 'normal'
        elif vol_percentile < 0.75:
            return 'high'
        else:
            return 'extreme'
    
    def _detect_trend_regime(self, data: pd.DataFrame) -> str:
        """트렌드 레짐 감지"""
        
        # 트렌드 지표
        if 'adx' in data.columns:
            adx = data['adx'].iloc[-1]
        else:
            adx = 20  # 기본값
        
        # 방향성
        if 'ema_slope_slow' in data.columns:
            slope = data['ema_slope_slow'].iloc[-1]
        else:
            sma = data['close'].rolling(20).mean()
            slope = (sma.iloc[-1] - sma.iloc[-5]) / sma.iloc[-5]
        
        # 분류
        if adx > 30:  # 강한 트렌드
            if slope > 0.01:
                return 'strong_up'
            elif slope < -0.01:
                return 'strong_down'
        elif adx > 20:  # 약한 트렌드
            if slope > 0:
                return 'weak_up'
            elif slope < 0:
                return 'weak_down'
        
        return 'neutral'
    
    def _detect_liquidity_regime(self, data: pd.DataFrame) -> str:
        """유동성 레짐 감지"""
        
        if 'depth_total' in data.columns:
            current_depth = data['depth_total'].iloc[-1]
            depth_ma = data['depth_total'].rolling(50).mean().iloc[-1]
            
            ratio = current_depth / depth_ma if depth_ma > 0 else 1
            
            if ratio > 1.2:
                return 'deep'
            elif ratio < 0.8:
                return 'thin'
            else:
                return 'normal'
        
        # 볼륨 기반 대체
        if 'volume' in data.columns:
            current_vol = data['volume'].iloc[-1]
            vol_ma = data['volume'].rolling(50).mean().iloc[-1]
            
            ratio = current_vol / vol_ma if vol_ma > 0 else 1
            
            if ratio > 1.5:
                return 'deep'
            elif ratio < 0.5:
                return 'thin'
        
        return 'normal'
    
    def _calculate_multipliers(self,
                              vol_regime: str,
                              trend_regime: str,
                              liquidity_regime: str) -> Dict:
        """레짐에 따른 파라미터 조정 계수"""
        
        # 변동성 기반 조정
        vol_multipliers = {
            'low': {'position_size': 1.2, 'tp_atr': 0.8, 'sl_atr': 0.9},
            'normal': {'position_size': 1.0, 'tp_atr': 1.0, 'sl_atr': 1.0},
            'high': {'position_size': 0.7, 'tp_atr': 1.3, 'sl_atr': 1.1},
            'extreme': {'position_size': 0.4, 'tp_atr': 1.5, 'sl_atr': 1.2}
        }
        
        # 트렌드 기반 조정
        trend_multipliers = {
            'strong_up': {'long_bias': 1.3, 'short_bias': 0.7},
            'weak_up': {'long_bias': 1.1, 'short_bias': 0.9},
            'neutral': {'long_bias': 1.0, 'short_bias': 1.0},
            'weak_down': {'long_bias': 0.9, 'short_bias': 1.1},
            'strong_down': {'long_bias': 0.7, 'short_bias': 1.3}
        }
        
        # 유동성 기반 조정
        liquidity_multipliers = {
            'deep': {'spread_tolerance': 1.2, 'order_size': 1.1},
            'normal': {'spread_tolerance': 1.0, 'order_size': 1.0},
            'thin': {'spread_tolerance': 0.8, 'order_size': 0.8}
        }
        
        # 통합
        multipliers = {}
        multipliers.update(vol_multipliers.get(vol_regime, {}))
        multipliers.update(trend_multipliers.get(trend_regime, {}))
        multipliers.update(liquidity_multipliers.get(liquidity_regime, {}))
        
        # OFI 임계값 조정
        if vol_regime == 'extreme':
            multipliers['ofi_threshold'] = 0.5  # 더 엄격한 기준
        elif vol_regime == 'low':
            multipliers['ofi_threshold'] = 1.5  # 더 완화된 기준
        else:
            multipliers['ofi_threshold'] = 1.0
        
        return multipliers
    
    def _extract_regime_parameters(self, X_scaled: np.ndarray):
        """레짐별 파라미터 추출"""
        
        # 각 레짐의 평균과 공분산
        for i in range(self.n_regimes):
            self.regime_params[i] = {
                'mean': self.hmm_model.means_[i],
                'covar': self.hmm_model.covars_[i],
                'transition_prob': self.hmm_model.transmat_[i]
            }
    
    def _label_regimes(self, X: np.ndarray):
        """레짐 자동 라벨링"""
        
        # 변동성 기준으로 정렬
        vol_means = []
        for i in range(self.n_regimes):
            # 변동성 피처 인덱스 (1번)
            vol_mean = self.hmm_model.means_[i][1] if len(self.hmm_model.means_[i]) > 1 else 0
            vol_means.append(vol_mean)
        
        # 정렬된 인덱스
        sorted_idx = np.argsort(vol_means)
        
        # 라벨 재할당
        new_labels = {}
        for new_idx, old_idx in enumerate(sorted_idx):
            if self.n_regimes == 4:
                new_labels[old_idx] = self.regime_labels[new_idx]
            else:
                new_labels[old_idx] = f"regime_{new_idx}"
        
        self.regime_labels = new_labels
    
    def _calculate_transition_probabilities(self, X_scaled: np.ndarray) -> Dict:
        """레짐 전환 확률 계산"""
        
        # 현재 레짐
        current_regime = self.hmm_model.predict(X_scaled)[-1]
        
        # 전환 확률
        trans_probs = {}
        for next_regime in range(self.n_regimes):
            prob = self.hmm_model.transmat_[current_regime][next_regime]
            trans_probs[self.regime_labels.get(next_regime, f"regime_{next_regime}")] = prob
        
        return trans_probs
    
    def _calculate_regime_confidence(self, X_scaled: np.ndarray) -> float:
        """레짐 신뢰도 계산"""
        
        # 마지막 관측치의 각 레짐 확률
        _, posteriors = self.hmm_model.score_samples(X_scaled)
        current_probs = posteriors[-1]
        
        # 최대 확률 (신뢰도)
        confidence = np.max(current_probs)
        
        return confidence
    
    def plot_regime_analysis(self, data: pd.DataFrame, save_path: Optional[str] = None):
        """레짐 분석 시각화"""
        
        # 피처 추출
        features = self._get_default_features()
        X = self._extract_features(data, features)
        X_scaled = self.scaler.transform(X)
        
        # 레짐 예측
        regimes = self.hmm_model.predict(X_scaled)
        
        # 플롯
        fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
        
        # 1. 가격과 레짐
        axes[0].plot(data.index, data['close'], label='Price', alpha=0.7)
        
        # 레짐별 배경색
        regime_colors = {
            0: 'green',
            1: 'yellow',
            2: 'orange',
            3: 'red'
        }
        
        for i in range(len(regimes)):
            if i == 0 or regimes[i] != regimes[i-1]:
                axes[0].axvspan(data.index[i], 
                              data.index[min(i+1, len(data)-1)],
                              alpha=0.2,
                              color=regime_colors.get(regimes[i], 'gray'))
        
        axes[0].set_ylabel('Price')
        axes[0].set_title('Price and Market Regimes')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 2. 변동성
        if 'realized_vol' in data.columns:
            vol = data['realized_vol']
        else:
            vol = data['close'].pct_change().rolling(20).std()
        
        axes[1].plot(data.index, vol, label='Volatility', alpha=0.7)
        axes[1].set_ylabel('Volatility')
        axes[1].set_title('Realized Volatility')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # 3. 레짐 확률
        _, posteriors = self.hmm_model.score_samples(X_scaled)
        
        for i in range(self.n_regimes):
            label = self.regime_labels.get(i, f"Regime {i}")
            axes[2].plot(data.index, posteriors[:, i], 
                        label=label, alpha=0.7)
        
        axes[2].set_ylabel('Probability')
        axes[2].set_xlabel('Date')
        axes[2].set_title('Regime Probabilities')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            logger.info(f"레짐 분석 플롯 저장: {save_path}")
        
        plt.close()
    
    def get_regime_statistics(self, data: pd.DataFrame) -> pd.DataFrame:
        """레짐별 통계"""
        
        # 피처 추출
        features = self._get_default_features()
        X = self._extract_features(data, features)
        X_scaled = self.scaler.transform(X)
        
        # 레짐 예측
        regimes = self.hmm_model.predict(X_scaled)
        
        # 레짐별 통계 계산
        stats = []
        
        for regime_idx in range(self.n_regimes):
            mask = regimes == regime_idx
            
            if mask.sum() > 0:
                regime_data = data[mask]
                returns = regime_data['close'].pct_change()
                
                stats.append({
                    'regime': self.regime_labels.get(regime_idx, f"regime_{regime_idx}"),
                    'frequency': mask.mean(),
                    'avg_return': returns.mean(),
                    'volatility': returns.std(),
                    'sharpe': returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0,
                    'max_drawdown': (returns + 1).cumprod().expanding().max().iloc[-1] - 1 if len(returns) > 0 else 0,
                    'avg_duration': self._calculate_avg_duration(regimes, regime_idx)
                })
        
        return pd.DataFrame(stats)
    
    def _calculate_avg_duration(self, regimes: np.ndarray, regime_idx: int) -> float:
        """평균 레짐 지속 기간"""
        
        durations = []
        current_duration = 0
        
        for regime in regimes:
            if regime == regime_idx:
                current_duration += 1
            else:
                if current_duration > 0:
                    durations.append(current_duration)
                    current_duration = 0
        
        if current_duration > 0:
            durations.append(current_duration)
        
        return np.mean(durations) if durations else 0

# CLI 실행용
def main():
    import argparse
    import duckdb
    
    parser = argparse.ArgumentParser(description='마켓 레짐 감지')
    parser.add_argument('--symbol', required=True, help='심볼')
    parser.add_argument('--timeframe', default='3m', help='시간프레임')
    parser.add_argument('--n-regimes', type=int, default=4, help='레짐 수')
    parser.add_argument('--db', default='data/trading.db', help='DB 경로')
    parser.add_argument('--train', action='store_true', help='모델 학습')
    parser.add_argument('--plot', action='store_true', help='시각화')
    
    args = parser.parse_args()
    
    # 데이터 로드
    conn = duckdb.connect(args.db)
    
    query = f"""
        SELECT *
        FROM trading.features_{args.timeframe}
        WHERE symbol = '{args.symbol}'
        ORDER BY ts
    """
    
    data = conn.execute(query).df()
    data['ts'] = pd.to_datetime(data['ts'])
    data.set_index('ts', inplace=True)
    
    # 레짐 감지기
    detector = AdaptiveRegimeDetector(n_regimes=args.n_regimes)
    
    if args.train:
        # 모델 학습
        detector.fit(data)
        
        # 통계
        stats = detector.get_regime_statistics(data)
        print("\n레짐 통계:")
        print(stats.to_string())
    
    # 현재 레짐
    current_regime = detector.detect(data.iloc[-100:])
    
    print(f"\n현재 레짐:")
    print(f"  변동성: {current_regime['vol_state']}")
    print(f"  트렌드: {current_regime['trend_state']}")
    print(f"  유동성: {current_regime['liquidity_state']}")
    print(f"  신뢰도: {current_regime['confidence']:.2%}")
    
    if args.plot:
        detector.plot_regime_analysis(data, 'regime_analysis.png')

if __name__ == "__main__":
    main()