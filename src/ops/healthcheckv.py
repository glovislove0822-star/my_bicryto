"""시스템 헬스 체크"""

import psutil
import asyncio
import aiohttp
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import logging
import json
from pathlib import Path

from ..utils.logging import Logger

logger = Logger.get_logger(__name__)

class HealthStatus:
    """헬스 상태"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    UNKNOWN = "unknown"

class HealthChecker:
    """시스템 헬스 체커
    
    시스템 리소스, 연결, 데이터 품질 등 모니터링
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Args:
            config: 헬스체크 설정
        """
        self.config = config or {}
        
        # 체크 항목
        self.checks = {
            'system': self._check_system_resources,
            'ws_connection': self._check_ws_connection,
            'data_quality': self._check_data_quality,
            'latency': self._check_latency,
            'memory': self._check_memory,
            'disk': self._check_disk,
            'process': self._check_process,
            'network': self._check_network,
            'database': self._check_database,
            'models': self._check_models
        }
        
        # 임계값
        self.thresholds = {
            'cpu_percent': 80,
            'memory_percent': 85,
            'disk_percent': 90,
            'latency_ms': 1000,
            'data_staleness_seconds': 60,
            'error_rate': 0.05,
            'network_loss_percent': 1
        }
        
        if config and 'thresholds' in config:
            self.thresholds.update(config['thresholds'])
        
        # 상태 히스토리
        self.status_history = []
        self.last_check = None
        
        # 메트릭
        self.metrics = {}
    
    async def check_all(self) -> Dict:
        """전체 헬스 체크
        
        Returns:
            헬스 체크 결과
        """
        
        logger.debug("헬스 체크 시작")
        
        results = {}
        overall_status = HealthStatus.HEALTHY
        
        # 각 체크 실행
        for check_name, check_func in self.checks.items():
            try:
                result = await check_func()
                results[check_name] = result
                
                # 전체 상태 업데이트
                if result['status'] == HealthStatus.CRITICAL:
                    overall_status = HealthStatus.CRITICAL
                elif result['status'] == HealthStatus.DEGRADED and overall_status != HealthStatus.CRITICAL:
                    overall_status = HealthStatus.DEGRADED
                    
            except Exception as e:
                logger.error(f"헬스 체크 실패 ({check_name}): {e}")
                results[check_name] = {
                    'status': HealthStatus.UNKNOWN,
                    'error': str(e)
                }
        
        # 종합 결과
        health_report = {
            'timestamp': datetime.now(),
            'overall_status': overall_status,
            'checks': results,
            'metrics': self.metrics.copy()
        }
        
        # 히스토리 업데이트
        self.status_history.append(health_report)
        if len(self.status_history) > 100:
            self.status_history.pop(0)
        
        self.last_check = datetime.now()
        
        # 상태 변경 시 알림
        if self.status_history and len(self.status_history) > 1:
            prev_status = self.status_history[-2]['overall_status']
            if prev_status != overall_status:
                await self._notify_status_change(prev_status, overall_status)
        
        logger.info(f"헬스 체크 완료: {overall_status}")
        
        return health_report
    
    async def _check_system_resources(self) -> Dict:
        """시스템 리소스 체크"""
        
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        
        status = HealthStatus.HEALTHY
        issues = []
        
        if cpu_percent > self.thresholds['cpu_percent']:
            status = HealthStatus.DEGRADED
            issues.append(f"High CPU usage: {cpu_percent:.1f}%")
        
        if memory.percent > self.thresholds['memory_percent']:
            if memory.percent > 95:
                status = HealthStatus.CRITICAL
            else:
                status = HealthStatus.DEGRADED
            issues.append(f"High memory usage: {memory.percent:.1f}%")
        
        self.metrics['cpu_percent'] = cpu_percent
        self.metrics['memory_percent'] = memory.percent
        self.metrics['memory_available_gb'] = memory.available / (1024**3)
        
        return {
            'status': status,
            'cpu_percent': cpu_percent,
            'memory_percent': memory.percent,
            'memory_available_gb': memory.available / (1024**3),
            'issues': issues
        }
    
    async def _check_ws_connection(self) -> Dict:
        """웹소켓 연결 체크"""
        
        # WebSocketStream 인스턴스 접근 필요
        # 여기서는 간단한 구현
        
        status = HealthStatus.HEALTHY
        issues = []
        
        # 실제로는 ws_stream.connected 체크
        ws_connected = True  # 임시
        
        if not ws_connected:
            status = HealthStatus.CRITICAL
            issues.append("WebSocket disconnected")
        
        return {
            'status': status,
            'connected': ws_connected,
            'issues': issues
        }
    
    async def _check_data_quality(self) -> Dict:
        """데이터 품질 체크"""
        
        status = HealthStatus.HEALTHY
        issues = []
        
        # 데이터 최신성 체크
        # 실제로는 state.market_state['last_update'] 체크
        last_update = datetime.now() - timedelta(seconds=10)  # 임시
        staleness = (datetime.now() - last_update).total_seconds()
        
        if staleness > self.thresholds['data_staleness_seconds']:
            if staleness > 300:  # 5분
                status = HealthStatus.CRITICAL
            else:
                status = HealthStatus.DEGRADED
            issues.append(f"Stale data: {staleness:.0f}s old")
        
        self.metrics['data_staleness_seconds'] = staleness
        
        return {
            'status': status,
            'staleness_seconds': staleness,
            'issues': issues
        }
    
    async def _check_latency(self) -> Dict:
        """레이턴시 체크"""
        
        status = HealthStatus.HEALTHY
        issues = []
        latencies = {}
        
        # 각 컴포넌트 레이턴시 측정
        # 실제로는 실제 측정값 사용
        
        # 웹소켓 레이턴시
        ws_latency = 50  # ms, 임시
        latencies['websocket'] = ws_latency
        
        # DB 레이턴시
        db_latency = 20  # ms, 임시
        latencies['database'] = db_latency
        
        # 추론 레이턴시
        inference_latency = 10  # ms, 임시
        latencies['inference'] = inference_latency
        
        # 총 레이턴시
        total_latency = sum(latencies.values())
        
        if total_latency > self.thresholds['latency_ms']:
            if total_latency > 2000:
                status = HealthStatus.CRITICAL
            else:
                status = HealthStatus.DEGRADED
            issues.append(f"High latency: {total_latency:.0f}ms")
        
        self.metrics['latency_ms'] = total_latency
        self.metrics.update({f'latency_{k}_ms': v for k, v in latencies.items()})
        
        return {
            'status': status,
            'latencies': latencies,
            'total_ms': total_latency,
            'issues': issues
        }
    
    async def _check_memory(self) -> Dict:
        """메모리 상세 체크"""
        
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()
        
        status = HealthStatus.HEALTHY
        issues = []
        
        # 메모리 압박 체크
        if memory.percent > self.thresholds['memory_percent']:
            status = HealthStatus.DEGRADED
            issues.append(f"Memory pressure: {memory.percent:.1f}%")
        
        # 스왑 사용 체크
        if swap.percent > 50:
            status = HealthStatus.DEGRADED
            issues.append(f"High swap usage: {swap.percent:.1f}%")
        
        # 프로세스별 메모리
        current_process = psutil.Process()
        process_memory = current_process.memory_info()
        
        return {
            'status': status,
            'total_gb': memory.total / (1024**3),
            'available_gb': memory.available / (1024**3),
            'percent': memory.percent,
            'swap_percent': swap.percent,
            'process_rss_mb': process_memory.rss / (1024**2),
            'process_vms_mb': process_memory.vms / (1024**2),
            'issues': issues
        }
    
    async def _check_disk(self) -> Dict:
        """디스크 체크"""
        
        disk = psutil.disk_usage('/')
        
        status = HealthStatus.HEALTHY
        issues = []
        
        if disk.percent > self.thresholds['disk_percent']:
            if disk.percent > 95:
                status = HealthStatus.CRITICAL
            else:
                status = HealthStatus.DEGRADED
            issues.append(f"Low disk space: {disk.percent:.1f}% used")
        
        # IO 통계
        io_counters = psutil.disk_io_counters()
        
        return {
            'status': status,
            'total_gb': disk.total / (1024**3),
            'free_gb': disk.free / (1024**3),
            'percent': disk.percent,
            'read_mb_s': io_counters.read_bytes / (1024**2) if io_counters else 0,
            'write_mb_s': io_counters.write_bytes / (1024**2) if io_counters else 0,
            'issues': issues
        }
    
    async def _check_process(self) -> Dict:
        """프로세스 체크"""
        
        current_process = psutil.Process()
        
        status = HealthStatus.HEALTHY
        issues = []
        
        # CPU 사용률
        cpu_percent = current_process.cpu_percent(interval=1)
        if cpu_percent > 100:
            status = HealthStatus.DEGRADED
            issues.append(f"High process CPU: {cpu_percent:.1f}%")
        
        # 스레드 수
        num_threads = current_process.num_threads()
        if num_threads > 100:
            status = HealthStatus.DEGRADED
            issues.append(f"Too many threads: {num_threads}")
        
        # 파일 디스크립터
        try:
            num_fds = current_process.num_fds()
            if num_fds > 1000:
                status = HealthStatus.DEGRADED
                issues.append(f"Too many file descriptors: {num_fds}")
        except:
            num_fds = 0
        
        return {
            'status': status,
            'pid': current_process.pid,
            'cpu_percent': cpu_percent,
            'num_threads': num_threads,
            'num_fds': num_fds,
            'create_time': datetime.fromtimestamp(current_process.create_time()),
            'issues': issues
        }
    
    async def _check_network(self) -> Dict:
        """네트워크 체크"""
        
        status = HealthStatus.HEALTHY
        issues = []
        
        # 네트워크 통계
        net_io = psutil.net_io_counters()
        
        # 패킷 손실률
        if net_io.dropin + net_io.dropout > 0:
            total_packets = net_io.packets_sent + net_io.packets_recv
            if total_packets > 0:
                loss_percent = (net_io.dropin + net_io.dropout) / total_packets * 100
                
                if loss_percent > self.thresholds['network_loss_percent']:
                    status = HealthStatus.DEGRADED
                    issues.append(f"Packet loss: {loss_percent:.2f}%")
        
        # 에러율
        if net_io.errin + net_io.errout > 0:
            if net_io.packets_sent + net_io.packets_recv > 0:
                error_rate = (net_io.errin + net_io.errout) / (net_io.packets_sent + net_io.packets_recv)
                
                if error_rate > self.thresholds['error_rate']:
                    status = HealthStatus.DEGRADED
                    issues.append(f"Network errors: {error_rate:.2%}")
        
        return {
            'status': status,
            'bytes_sent_mb': net_io.bytes_sent / (1024**2),
            'bytes_recv_mb': net_io.bytes_recv / (1024**2),
            'packets_sent': net_io.packets_sent,
            'packets_recv': net_io.packets_recv,
            'errors': net_io.errin + net_io.errout,
            'drops': net_io.dropin + net_io.dropout,
            'issues': issues
        }
    
    async def _check_database(self) -> Dict:
        """데이터베이스 체크"""
        
        status = HealthStatus.HEALTHY
        issues = []
        
        # DuckDB 파일 체크
        db_path = Path('data/trading.db')
        
        if not db_path.exists():
            status = HealthStatus.CRITICAL
            issues.append("Database file not found")
            
        else:
            # 파일 크기 체크
            db_size_gb = db_path.stat().st_size / (1024**3)
            
            if db_size_gb > 10:
                status = HealthStatus.DEGRADED
                issues.append(f"Large database: {db_size_gb:.1f}GB")
            
            # 연결 테스트
            try:
                import duckdb
                conn = duckdb.connect(str(db_path), read_only=True)
                
                # 간단한 쿼리
                result = conn.execute("SELECT COUNT(*) FROM information_schema.tables").fetchone()
                table_count = result[0] if result else 0
                
                conn.close()
                
            except Exception as e:
                status = HealthStatus.CRITICAL
                issues.append(f"Database connection failed: {e}")
                table_count = 0
                db_size_gb = 0
        
        return {
            'status': status,
            'exists': db_path.exists(),
            'size_gb': db_size_gb if db_path.exists() else 0,
            'table_count': table_count if db_path.exists() else 0,
            'issues': issues
        }
    
    async def _check_models(self) -> Dict:
        """모델 체크"""
        
        status = HealthStatus.HEALTHY
        issues = []
        models_status = {}
        
        # 모델 파일 체크
        model_files = {
            'lightgbm': Path('models/lightgbm_best.pkl'),
            'threshold_state': Path('models/threshold_state.json'),
            'regime_model': Path('models/regime_model.pkl')
        }
        
        for model_name, model_path in model_files.items():
            if model_path.exists():
                # 파일 크기와 수정 시간
                stat = model_path.stat()
                age_days = (datetime.now() - datetime.fromtimestamp(stat.st_mtime)).days
                
                models_status[model_name] = {
                    'exists': True,
                    'size_mb': stat.st_size / (1024**2),
                    'age_days': age_days
                }
                
                # 오래된 모델 경고
                if age_days > 30:
                    status = HealthStatus.DEGRADED
                    issues.append(f"Old model: {model_name} ({age_days} days)")
                    
            else:
                models_status[model_name] = {
                    'exists': False
                }
                
                if model_name == 'lightgbm':  # 필수 모델
                    status = HealthStatus.CRITICAL
                    issues.append(f"Missing critical model: {model_name}")
        
        return {
            'status': status,
            'models': models_status,
            'issues': issues
        }
    
    async def _notify_status_change(self, old_status: str, new_status: str):
        """상태 변경 알림
        
        Args:
            old_status: 이전 상태
            new_status: 새 상태
        """
        
        message = f"Health status changed: {old_status} → {new_status}"
        
        if new_status == HealthStatus.CRITICAL:
            logger.critical(message)
        elif new_status == HealthStatus.DEGRADED:
            logger.warning(message)
        else:
            logger.info(message)
        
        # Slack 웹훅 알림 (설정된 경우)
        webhook_url = self.config.get('slack_webhook')
        if webhook_url:
            await self._send_slack_notification(webhook_url, message, new_status)
    
    async def _send_slack_notification(self, webhook_url: str, message: str, status: str):
        """Slack 알림 전송
        
        Args:
            webhook_url: Slack 웹훅 URL
            message: 메시지
            status: 상태
        """
        
        # 상태별 색상
        color_map = {
            HealthStatus.HEALTHY: "good",
            HealthStatus.DEGRADED: "warning",
            HealthStatus.CRITICAL: "danger",
            HealthStatus.UNKNOWN: "#808080"
        }
        
        payload = {
            "attachments": [{
                "color": color_map.get(status, "#808080"),
                "title": "System Health Alert",
                "text": message,
                "footer": "TradingBot Health Monitor",
                "ts": int(datetime.now().timestamp())
            }]
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(webhook_url, json=payload) as response:
                    if response.status != 200:
                        logger.error(f"Slack notification failed: {response.status}")
        except Exception as e:
            logger.error(f"Slack notification error: {e}")
    
    def get_metrics_summary(self) -> Dict:
        """메트릭 요약 조회
        
        Returns:
            메트릭 요약
        """
        
        return {
            'last_check': self.last_check,
            'current_metrics': self.metrics.copy(),
            'recent_issues': self._get_recent_issues(),
            'uptime_percent': self._calculate_uptime()
        }
    
    def _get_recent_issues(self) -> List[str]:
        """최근 이슈 조회
        
        Returns:
            최근 이슈 리스트
        """
        
        issues = []
        
        for report in self.status_history[-5:]:  # 최근 5개
            for check_name, check_result in report['checks'].items():
                if 'issues' in check_result:
                    for issue in check_result['issues']:
                        issues.append(f"[{check_name}] {issue}")
        
        return list(set(issues))  # 중복 제거
    
    def _calculate_uptime(self) -> float:
        """가동률 계산
        
        Returns:
            가동률 (%)
        """
        
        if not self.status_history:
            return 100.0
        
        healthy_count = sum(
            1 for report in self.status_history
            if report['overall_status'] == HealthStatus.HEALTHY
        )
        
        return (healthy_count / len(self.status_history)) * 100
    
    async def run_continuous(self, interval_seconds: int = 60):
        """지속적 헬스 체크 실행
        
        Args:
            interval_seconds: 체크 간격 (초)
        """
        
        logger.info(f"헬스 체크 시작 (간격: {interval_seconds}초)")
        
        while True:
            try:
                await self.check_all()
                await asyncio.sleep(interval_seconds)
                
            except asyncio.CancelledError:
                logger.info("헬스 체크 중단")
                break
                
            except Exception as e:
                logger.error(f"헬스 체크 에러: {e}")
                await asyncio.sleep(interval_seconds)

# CLI 실행용
async def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='시스템 헬스 체크')
    parser.add_argument('--config', help='설정 파일')
    parser.add_argument('--continuous', action='store_true', help='지속 실행')
    parser.add_argument('--interval', type=int, default=60, help='체크 간격 (초)')
    parser.add_argument('--output', help='결과 출력 파일')
    
    args = parser.parse_args()
    
    # 설정 로드
    config = {}
    if args.config:
        with open(args.config) as f:
            config = json.load(f)
    
    # 헬스 체커 생성
    checker = HealthChecker(config)
    
    if args.continuous:
        # 지속 실행
        await checker.run_continuous(args.interval)
    else:
        # 단일 체크
        result = await checker.check_all()
        
        # 결과 출력
        print(json.dumps(result, indent=2, default=str))
        
        # 파일 저장
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2, default=str)

if __name__ == "__main__":
    asyncio.run(main())