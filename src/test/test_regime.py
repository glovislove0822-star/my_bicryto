"""레짐 감지 테스트"""

import unittest
import numpy as np
import pandas as pd

from src.ml.regime_detector import AdaptiveRegimeDetector

class TestRegimeDetector(unittest.TestCase):
    """레짐 감지기 테스트"""
    
    def setUp(self):
        """테스트 설정"""
        self.detector = AdaptiveRegimeDetector()
        
        # 테스트 특징 데이터
        self.features = {
            'atr': 0.01,
            'parkinson': 0.008,
            'adx': 25,
            'hurst': 0.5,
            'spread': 1.0,
            'depth': 10000
        }
    
    def test_regime_detection(self):
        """레짐 감지 테스트"""
        
        # 정상 시장
        regime = self.detector.detect(self.features)
        
        # 결과 확인
        self.assertIn('vol_state', regime)
        self.assertIn('trend_state', regime)
        self.assertIn('liquidity_state', regime)
        self.assertIn('params_multiplier', regime)
        
        # 기본 상태 확인
        self.assertEqual(regime['vol_state'], 'normal')
    
    def test_high_volatility_regime(self):
        """고변동성 레짐 테스트"""
        
        # 고변동성 특징
        high_vol_features = self.features.copy()
        high_vol_features['atr'] = 0.05  # 5% ATR
        high_vol_features['parkinson'] = 0.04
        
        regime = self.detector.detect(high_vol_features)
        
        # 고변동성 감지 확인
        self.assertIn(regime['vol_state'], ['high', 'extreme'])
        
        # 파라미터 조정 확인
        multipliers = regime['params_multiplier']
        self.assertLess(multipliers['position_size'], 1.0)  # 포지션 축소
        self.assertGreater(multipliers['tp_atr'], 1.0)  # TP 확대
    
    def test_trending_regime(self):
        """트렌드 레짐 테스트"""
        
        # 강한 트렌드 특징
        trend_features = self.features.copy()
        trend_features['adx'] = 50  # 강한 트렌드
        trend_features['hurst'] = 0.7  # 지속성
        
        regime = self.detector.detect(trend_features)
        
        # 트렌드 감지 확인
        self.assertIn('strong', regime['trend_state'])
    
    def test_parameter_adjustment(self):
        """파라미터 조정 테스트"""
        
        # 다양한 레짐 테스트
        test_cases = [
            ('low_vol', {'atr': 0.005}, 'position_size', 'greater'),
            ('high_vol', {'atr': 0.03}, 'position_size', 'less'),
            ('illiquid', {'depth': 1000}, 'ofi_threshold', 'greater')
        ]
        
        for name, feature_update, param, comparison in test_cases:
            with self.subTest(regime=name):
                features = self.features.copy()
                features.update(feature_update)
                
                regime = self.detector.detect(features)
                multipliers = regime['params_multiplier']
                
                if comparison == 'greater':
                    self.assertGreater(multipliers[param], 1.0)
                else:
                    self.assertLess(multipliers[param], 1.0)

if __name__ == '__main__':
    unittest.main()