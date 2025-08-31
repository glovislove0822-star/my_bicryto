"""로깅 유틸리티"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional
from rich.logging import RichHandler
from rich.console import Console

class Logger:
    """향상된 로거 클래스"""
    
    _instances = {}
    
    def __new__(cls, name: str, *args, **kwargs):
        """싱글톤 패턴"""
        if name not in cls._instances:
            cls._instances[name] = super().__new__(cls)
        return cls._instances[name]
    
    def __init__(self, 
                 name: str,
                 log_dir: Optional[Path] = None,
                 level: int = logging.INFO,
                 use_rich: bool = True):
        """
        Args:
            name: 로거 이름
            log_dir: 로그 파일 저장 디렉토리
            level: 로그 레벨
            use_rich: Rich 핸들러 사용 여부
        """
        if hasattr(self, '_initialized'):
            return
            
        self.name = name
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.logger.handlers.clear()
        
        # 포맷터
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # 파일 핸들러
        if log_dir:
            log_dir = Path(log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            
            # 일별 로그 파일
            today = datetime.now().strftime('%Y%m%d')
            log_file = log_dir / f"{name}_{today}.log"
            
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setFormatter(file_formatter)
            file_handler.setLevel(level)
            self.logger.addHandler(file_handler)
        
        # 콘솔 핸들러
        if use_rich:
            console_handler = RichHandler(
                console=Console(stderr=True),
                show_time=True,
                show_path=True,
                enable_link_path=True
            )
        else:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(file_formatter)
        
        console_handler.setLevel(level)
        self.logger.addHandler(console_handler)
        
        self._initialized = True
    
    def debug(self, msg: str, *args, **kwargs):
        """디버그 로그"""
        self.logger.debug(msg, *args, **kwargs)
    
    def info(self, msg: str, *args, **kwargs):
        """정보 로그"""
        self.logger.info(msg, *args, **kwargs)
    
    def warning(self, msg: str, *args, **kwargs):
        """경고 로그"""
        self.logger.warning(msg, *args, **kwargs)
    
    def error(self, msg: str, *args, **kwargs):
        """에러 로그"""
        self.logger.error(msg, *args, **kwargs)
    
    def critical(self, msg: str, *args, **kwargs):
        """크리티컬 로그"""
        self.logger.critical(msg, *args, **kwargs)
    
    def exception(self, msg: str, *args, **kwargs):
        """예외 로그 (스택 트레이스 포함)"""
        self.logger.exception(msg, *args, **kwargs)
    
    def set_level(self, level: int):
        """로그 레벨 변경"""
        self.logger.setLevel(level)
        for handler in self.logger.handlers:
            handler.setLevel(level)
    
    @classmethod
    def get_logger(cls, name: str, **kwargs) -> 'Logger':
        """로거 인스턴스 획득"""
        return cls(name, **kwargs)
    
    def log_performance(self, func_name: str, elapsed_ms: float, threshold_ms: float = 1000):
        """성능 로그"""
        if elapsed_ms > threshold_ms:
            self.warning(f"[PERF] {func_name} 실행시간: {elapsed_ms:.2f}ms (임계값: {threshold_ms}ms)")
        else:
            self.debug(f"[PERF] {func_name} 실행시간: {elapsed_ms:.2f}ms")
    
    def log_trade(self, symbol: str, side: str, price: float, quantity: float, **kwargs):
        """거래 로그"""
        extra_info = ' '.join([f"{k}={v}" for k, v in kwargs.items()])
        self.info(f"[TRADE] {symbol} {side.upper()} price={price:.4f} qty={quantity:.4f} {extra_info}")
    
    def log_signal(self, symbol: str, signal_type: str, strength: float, **kwargs):
        """신호 로그"""
        extra_info = ' '.join([f"{k}={v}" for k, v in kwargs.items()])
        self.info(f"[SIGNAL] {symbol} {signal_type} strength={strength:.3f} {extra_info}")
    
    def log_risk(self, metric: str, value: float, threshold: float, status: str = "OK"):
        """리스크 로그"""
        if status == "OK":
            self.info(f"[RISK] {metric}={value:.4f} (threshold={threshold:.4f}) {status}")
        elif status == "WARNING":
            self.warning(f"[RISK] {metric}={value:.4f} (threshold={threshold:.4f}) {status}")
        else:
            self.error(f"[RISK] {metric}={value:.4f} (threshold={threshold:.4f}) {status}")