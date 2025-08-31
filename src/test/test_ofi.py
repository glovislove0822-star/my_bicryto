"""OFI 테스트"""

import unittest
import numpy as np
import pandas as pd
from datetime import datetime

from src.data.ofi import OFICalculator

class TestOFI(unittest.TestCase):
    """OFI 계산 테스트"""
    
    def setUp(self):
        """테스트 설정"""
        self.calculator = OFICalculator()
        
        # 테스트 데이터
        self.test_depth = {
            'best_bid': 50000,
            'best_ask': 50001,
            'bid_size': 100,
            'ask_size': 150,
            'timestamp': datetime.now()
        }
    
    def test_ofi_calculation(self):
        """OFI 계산 테스트"""
        
        # 첫 번째 상태
        self.calculator.update(self.test_depth)
        
        # 두 번째 상태 (호가 변경)
        new_depth = self.test_depth.copy()
        new_depth['best_bid'] = 50002
        new_depth['bid_size'] = 120
        new_depth['timestamp'] = datetime.now()
        
        ofi = self.calculator.update(new_depth)
        
        # OFI가 계산되었는지 확인
        self.assertIsNotNone(ofi)
        
        # OFI 방향 확인 (매수 압력)
        self.assertGreater(ofi, 0)
    
    def test_queue_imbalance(self):
        """큐 불균형 테스트"""
        
        qi = self.calculator.calculate_queue_imbalance(
            bid_size=100,
            ask_size=50
        )
        
        # 매수 우세
        self.assertGreater(qi, 0.5)
        self.assertAlmostEqual(qi, 100 / (100 + 50), places=4)
    
    def test_spread_calculation(self):
        """스프레드 계산 테스트"""
        
        spread_bps = self.calculator.calculate_spread_bps(
            best_bid=50000,
            best_ask=50010
        )
        
        # 10 / 50000 * 10000 = 2 bps
        self.assertAlmostEqual(spread_bps, 2.0, places=2)
    
    def test_voi_calculation(self):
        """VOI 계산 테스트"""
        
        # 여러 업데이트
        depths = [
            {'best_bid': 50000, 'best_ask': 50001, 'bid_size': 100, 'ask_size': 100},
            {'best_bid': 50001, 'best_ask': 50002, 'bid_size': 110, 'ask_size': 90},
            {'best_bid': 50002, 'best_ask': 50003, 'bid_size': 120, 'ask_size': 80},
        ]
        
        for depth in depths:
            depth['timestamp'] = datetime.now()
            self.calculator.update(depth)
        
        voi = self.calculator.get_voi()
        
        # VOI가 계산되었는지 확인
        self.assertIsNotNone(voi)
        
        # 상승 압력 확인
        self.assertGreater(voi, 0)
    
    def test_rolling_statistics(self):
        """롤링 통계 테스트"""
        
        # 여러 업데이트
        for i in range(100):
            depth = {
                'best_bid': 50000 + i,
                'best_ask': 50001 + i,
                'bid_size': 100 + np.random.randn() * 10,
                'ask_size': 100 + np.random.randn() * 10,
                'timestamp': datetime.now()
            }
            self.calculator.update(depth)
        
        stats = self.calculator.get_statistics()
        
        # 통계 확인
        self.assertIn('ofi_mean', stats)
        self.assertIn('ofi_std', stats)
        self.assertIn('qi_mean', stats)
        
        # Z-score 확인
        ofi_z = self.calculator.get_ofi_zscore()
        self.assertIsNotNone(ofi_z)
        
        # Z-score 범위 확인 (대부분 -3 ~ 3)
        self.assertGreater(ofi_z, -5)
        self.assertLess(ofi_z, 5)

if __name__ == '__main__':
    unittest.main()