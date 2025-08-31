"""
TF-PB-OFI-VT 암호화폐 트레이딩 시스템 v2.0
Enhanced with adaptive regime detection and self-learning capabilities
"""

__version__ = "2.0.0"
__author__ = "Quant Team"

import logging
from pathlib import Path

# 프로젝트 루트 경로
PROJECT_ROOT = Path(__file__).parent.parent

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(PROJECT_ROOT / 'logs' / 'system.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)
logger.info(f"TF-PB-OFI-VT v{__version__} 초기화")