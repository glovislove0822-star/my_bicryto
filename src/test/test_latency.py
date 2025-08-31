"""레이턴시 테스트"""

import unittest
import asyncio
import time
import numpy as np

from src.ops.latency_probe import LatencyProbe

class TestLatency(unittest.TestCase):
    """레이턴시 측정 테스트"""
    
    def setUp(self):
        """테스트 설정"""
        self.probe = LatencyProbe()
    
    def test_measurement(self):
        """측정 테스트"""
        
        # 측정 시작
        start = self.probe.start_measurement('test_operation')
        
        # 작업 시뮬레이션
        time.sleep(0.01)  # 10ms
        
        # 측정 종료
        latency = self.probe.end_measurement('test_operation', start)
        
        # 레이턴시 확인 (약 10ms)
        self.assertGreater(latency, 5)  # 5ms 이상
        self.assertLess(latency, 20)  # 20ms 이하
    
    def test_statistics(self):
        """통계 테스트"""
        
        # 여러 측정
        for i in range(10):
            start = self.probe.start_measurement('test_op')
            time.sleep(0.001 * (i + 1))  # 1-10ms
            self.probe.end_measurement('test_op', start)
        
        # 통계 계산
        report = self.probe._calculate_statistics()
        
        # 통계 확인
        self.assertIn('points', report)
        self.assertIn('test_op', report['points'])
        
        stats = report['points']['test_op']
        self.assertEqual(stats['count'], 10)
        self.assertGreater(stats['mean'], 0)
        self.assertGreater(stats['max'], stats['min'])
    
    def test_bottleneck_identification(self):
        """병목 지점 식별 테스트"""
        
        # 다양한 작업 측정
        operations = {
            'fast_op': 0.001,  # 1ms
            'slow_op': 0.050,  # 50ms
            'medium_op': 0.010  # 10ms
        }
        
        for op_name, delay in operations.items():
            for _ in range(5):
                start = self.probe.start_measurement(op_name)
                time.sleep(delay)
                self.probe.end_measurement(op_name, start)
        
        # 병목 지점 확인
        bottleneck = self.probe._identify_bottleneck()
        
        # 가장 느린 작업이 병목
        self.assertEqual(bottleneck, 'slow_op')
    
    def test_sla_compliance(self):
        """SLA 준수율 테스트"""
        
        # 다양한 레이턴시 측정
        latencies = [50, 100, 200, 500, 800, 1200, 1500, 2000, 300, 400]
        
        for latency_ms in latencies:
            start = self.probe.start_measurement('total_loop')
            # 직접 버퍼에 추가 (테스트용)
            self.probe.latency_buffer['total_loop'].append(latency_ms)
        
        # SLA 준수율 계산 (1초 이내)
        compliance = self.probe._calculate_sla_compliance()
        
        # 10개 중 7개가 1초 이내 = 70%
        self.assertAlmostEqual(compliance, 70.0, places=1)

class TestAsyncLatency(unittest.TestCase):
    """비동기 레이턴시 테스트"""
    
    def setUp(self):
        """테스트 설정"""
        self.probe = LatencyProbe()
    
    def test_async_measurement(self):
        """비동기 측정 테스트"""
        
        async def async_operation():
            """비동기 작업"""
            start = self.probe.start_measurement('async_op')
            await asyncio.sleep(0.01)  # 10ms
            return self.probe.end_measurement('async_op', start)
        
        # 비동기 실행
        loop = asyncio.new_event_loop()
        latency = loop.run_until_complete(async_operation())
        loop.close()
        
        # 레이턴시 확인
        self.assertGreater(latency, 5)
        self.assertLess(latency, 20)

if __name__ == '__main__':
    unittest.main()