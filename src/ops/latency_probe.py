"""레이턴시 프로브"""

import time
import asyncio
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime
from collections import defaultdict, deque
import logging
import json

from ..utils.logging import Logger

logger = Logger.get_logger(__name__)

class LatencyProbe:
    """레이턴시 측정 프로브
    
    시스템 각 컴포넌트의 레이턴시 측정 및 분석
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Args:
            config: 프로브 설정
        """
        self.config = config or {}
        
        # 측정 포인트
        self.measurement_points = [
            'ws_receive',      # 웹소켓 수신
            'ws_parse',        # 웹소켓 파싱
            'feature_calc',    # 특징 계산
            'ml_inference',    # ML 추론
            'signal_gen',      # 신호 생성
            'risk_check',      # 리스크 체크
            'order_prep',      # 주문 준비
            'order_send',      # 주문 전송
            'order_confirm',   # 주문 확인
            'db_read',         # DB 읽기
            'db_write',        # DB 쓰기
            'total_loop'       # 전체 루프
        ]
        
        # 레이턴시 버퍼
        self.latency_buffer = defaultdict(lambda: deque(maxlen=1000))
        
        # 통계
        self.statistics = {}
        
        # 측정 세션
        self.sessions = []
        self.current_session = None
        
        # 임계값 (밀리초)
        self.thresholds = {
            'ws_receive': 50,
            'ws_parse': 10,
            'feature_calc': 100,
            'ml_inference': 30,
            'signal_gen': 20,
            'risk_check': 10,
            'order_prep': 10,
            'order_send': 100,
            'order_confirm': 200,
            'db_read': 50,
            'db_write': 100,
            'total_loop': 500
        }
        
        if config and 'thresholds' in config:
            self.thresholds.update(config['thresholds'])
    
    def start_measurement(self, point: str) -> float:
        """측정 시작
        
        Args:
            point: 측정 포인트
            
        Returns:
            시작 시간
        """
        
        return time.perf_counter()
    
    def end_measurement(self, point: str, start_time: float) -> float:
        """측정 종료
        
        Args:
            point: 측정 포인트
            start_time: 시작 시간
            
        Returns:
            레이턴시 (밀리초)
        """
        
        latency_ms = (time.perf_counter() - start_time) * 1000
        
        # 버퍼에 추가
        self.latency_buffer[point].append(latency_ms)
        
        # 임계값 체크
        if point in self.thresholds:
            if latency_ms > self.thresholds[point]:
                logger.warning(f"High latency: {point} = {latency_ms:.1f}ms "
                             f"(threshold: {self.thresholds[point]}ms)")
        
        return latency_ms
    
    async def measure(self, minutes: int = 5) -> Dict:
        """레이턴시 측정 실행
        
        Args:
            minutes: 측정 시간 (분)
            
        Returns:
            측정 결과
        """
        
        logger.info(f"레이턴시 측정 시작 ({minutes}분)")
        
        # 새 세션 시작
        self.current_session = {
            'start_time': datetime.now(),
            'measurements': defaultdict(list),
            'duration_minutes': minutes
        }
        
        end_time = time.time() + (minutes * 60)
        measurement_count = 0
        
        while time.time() < end_time:
            measurement_count += 1
            
            # 전체 루프 시작
            loop_start = self.start_measurement('total_loop')
            
            # 각 컴포넌트 측정
            await self._measure_websocket()
            await self._measure_features()
            await self._measure_inference()
            await self._measure_database()
            await self._measure_execution()
            
            # 전체 루프 종료
            loop_latency = self.end_measurement('total_loop', loop_start)
            
            # 진행 상황 로깅
            if measurement_count % 60 == 0:
                elapsed = (time.time() - (end_time - minutes * 60)) / 60
                logger.info(f"측정 진행: {elapsed:.1f}/{minutes}분")
            
            # 1초 대기
            await asyncio.sleep(1)
        
        # 세션 종료
        self.current_session['end_time'] = datetime.now()
        self.current_session['measurement_count'] = measurement_count
        
        # 통계 계산
        report = self._calculate_statistics()
        
        # 세션 저장
        self.sessions.append(self.current_session)
        self.current_session = None
        
        logger.info("레이턴시 측정 완료")
        
        return report
    
    async def _measure_websocket(self):
        """웹소켓 레이턴시 측정"""
        
        # 수신 시뮬레이션
        start = self.start_measurement('ws_receive')
        await asyncio.sleep(0.01 + np.random.exponential(0.02))  # 시뮬레이션
        self.end_measurement('ws_receive', start)
        
        # 파싱 시뮬레이션
        start = self.start_measurement('ws_parse')
        await asyncio.sleep(0.001 + np.random.exponential(0.002))
        self.end_measurement('ws_parse', start)
    
    async def _measure_features(self):
        """특징 계산 레이턴시 측정"""
        
        start = self.start_measurement('feature_calc')
        
        # 특징 계산 시뮬레이션
        await asyncio.sleep(0.02 + np.random.exponential(0.03))
        
        self.end_measurement('feature_calc', start)
    
    async def _measure_inference(self):
        """ML 추론 레이턴시 측정"""
        
        start = self.start_measurement('ml_inference')
        
        # 추론 시뮬레이션
        await asyncio.sleep(0.005 + np.random.exponential(0.01))
        
        self.end_measurement('ml_inference', start)
        
        # 신호 생성
        start = self.start_measurement('signal_gen')
        await asyncio.sleep(0.002 + np.random.exponential(0.005))
        self.end_measurement('signal_gen', start)
        
        # 리스크 체크
        start = self.start_measurement('risk_check')
        await asyncio.sleep(0.001 + np.random.exponential(0.003))
        self.end_measurement('risk_check', start)
    
    async def _measure_database(self):
        """데이터베이스 레이턴시 측정"""
        
        # 읽기
        start = self.start_measurement('db_read')
        await asyncio.sleep(0.005 + np.random.exponential(0.01))
        self.end_measurement('db_read', start)
        
        # 쓰기 (가끔)
        if np.random.random() < 0.1:
            start = self.start_measurement('db_write')
            await asyncio.sleep(0.01 + np.random.exponential(0.02))
            self.end_measurement('db_write', start)
    
    async def _measure_execution(self):
        """주문 실행 레이턴시 측정"""
        
        # 주문 실행 (가끔)
        if np.random.random() < 0.05:
            # 준비
            start = self.start_measurement('order_prep')
            await asyncio.sleep(0.001 + np.random.exponential(0.002))
            self.end_measurement('order_prep', start)
            
            # 전송
            start = self.start_measurement('order_send')
            await asyncio.sleep(0.02 + np.random.exponential(0.03))
            self.end_measurement('order_send', start)
            
            # 확인
            start = self.start_measurement('order_confirm')
            await asyncio.sleep(0.05 + np.random.exponential(0.05))
            self.end_measurement('order_confirm', start)
    
    def _calculate_statistics(self) -> Dict:
        """통계 계산
        
        Returns:
            통계 결과
        """
        
        report = {
            'timestamp': datetime.now(),
            'points': {}
        }
        
        for point in self.measurement_points:
            if point in self.latency_buffer and self.latency_buffer[point]:
                values = list(self.latency_buffer[point])
                
                report['points'][point] = {
                    'count': len(values),
                    'mean': np.mean(values),
                    'median': np.median(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'p50': np.percentile(values, 50),
                    'p95': np.percentile(values, 95),
                    'p99': np.percentile(values, 99),
                    'threshold': self.thresholds.get(point),
                    'violations': sum(1 for v in values if v > self.thresholds.get(point, float('inf')))
                }
        
        # 종합 분석
        if 'total_loop' in report['points']:
            total_stats = report['points']['total_loop']
            
            report['summary'] = {
                'avg_latency_ms': total_stats['mean'],
                'p95_latency_ms': total_stats['p95'],
                'p99_latency_ms': total_stats['p99'],
                'max_latency_ms': total_stats['max'],
                'sla_compliance': self._calculate_sla_compliance(),
                'bottleneck': self._identify_bottleneck()
            }
        
        return report
    
    def _calculate_sla_compliance(self) -> float:
        """SLA 준수율 계산
        
        Returns:
            준수율 (%)
        """
        
        if 'total_loop' not in self.latency_buffer:
            return 100.0
        
        values = list(self.latency_buffer['total_loop'])
        if not values:
            return 100.0
        
        # 1초 이내 처리율
        compliance_count = sum(1 for v in values if v <= 1000)
        
        return (compliance_count / len(values)) * 100
    
    def _identify_bottleneck(self) -> Optional[str]:
        """병목 지점 식별
        
        Returns:
            병목 지점 이름
        """
        
        max_avg_latency = 0
        bottleneck = None
        
        for point in self.measurement_points:
            if point == 'total_loop':
                continue
            
            if point in self.latency_buffer and self.latency_buffer[point]:
                avg_latency = np.mean(list(self.latency_buffer[point]))
                
                if avg_latency > max_avg_latency:
                    max_avg_latency = avg_latency
                    bottleneck = point
        
        return bottleneck
    
    def get_real_time_stats(self) -> Dict:
        """실시간 통계 조회
        
        Returns:
            실시간 통계
        """
        
        stats = {}
        
        for point in self.measurement_points:
            if point in self.latency_buffer and self.latency_buffer[point]:
                recent = list(self.latency_buffer[point])[-10:]  # 최근 10개
                
                if recent:
                    stats[point] = {
                        'latest': recent[-1],
                        'avg_10': np.mean(recent),
                        'max_10': np.max(recent)
                    }
        
        return stats
    
    def export_report(self, filepath: str):
        """리포트 내보내기
        
        Args:
            filepath: 파일 경로
        """
        
        report = self._calculate_statistics()
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"레이턴시 리포트 저장: {filepath}")
    
    def print_summary(self):
        """요약 출력"""
        
        report = self._calculate_statistics()
        
        print("\n" + "="*60)
        print("레이턴시 측정 요약")
        print("="*60)
        
        if 'summary' in report:
            summary = report['summary']
            print(f"\n평균 레이턴시: {summary['avg_latency_ms']:.1f}ms")
            print(f"P95 레이턴시: {summary['p95_latency_ms']:.1f}ms")
            print(f"P99 레이턴시: {summary['p99_latency_ms']:.1f}ms")
            print(f"최대 레이턴시: {summary['max_latency_ms']:.1f}ms")
            print(f"SLA 준수율: {summary['sla_compliance']:.1f}%")
            
            if summary['bottleneck']:
                print(f"병목 지점: {summary['bottleneck']}")
        
        print("\n구간별 레이턴시 (P95):")
        for point, stats in report['points'].items():
            if point != 'total_loop':
                print(f"  {point:20s}: {stats['p95']:6.1f}ms")
        
        print("="*60)

# CLI 실행용
async def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='레이턴시 프로브')
    parser.add_argument('--minutes', type=int, default=5, help='측정 시간 (분)')
    parser.add_argument('--output', help='결과 출력 파일')
    parser.add_argument('--config', help='설정 파일')
    
    args = parser.parse_args()
    
    # 설정 로드
    config = {}
    if args.config:
        with open(args.config) as f:
            config = json.load(f)
    
    # 프로브 생성
    probe = LatencyProbe(config)
    
    # 측정 실행
    report = await probe.measure(args.minutes)
    
    # 요약 출력
    probe.print_summary()
    
    # 파일 저장
    if args.output:
        probe.export_report(args.output)

if __name__ == "__main__":
    asyncio.run(main())