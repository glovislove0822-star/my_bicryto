"""입출력 유틸리티"""

import json
import yaml
import pickle
import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path
from typing import Any, Dict, Optional, Union
import logging

logger = logging.getLogger(__name__)

class IOUtils:
    """파일 입출력 유틸리티"""
    
    @staticmethod
    def load_config(path: Union[str, Path]) -> Dict:
        """YAML 설정 파일 로드"""
        path = Path(path)
        with open(path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        logger.info(f"설정 파일 로드: {path}")
        return config
    
    @staticmethod
    def save_config(config: Dict, path: Union[str, Path]):
        """YAML 설정 파일 저장"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        logger.info(f"설정 파일 저장: {path}")
    
    @staticmethod
    def load_json(path: Union[str, Path]) -> Dict:
        """JSON 파일 로드"""
        path = Path(path)
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    
    @staticmethod
    def save_json(data: Any, path: Union[str, Path], indent: int = 2):
        """JSON 파일 저장"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)
        logger.info(f"JSON 파일 저장: {path}")
    
    @staticmethod
    def load_parquet(path: Union[str, Path]) -> pd.DataFrame:
        """Parquet 파일 로드"""
        path = Path(path)
        df = pd.read_parquet(path)
        logger.info(f"Parquet 파일 로드: {path} (rows: {len(df)})")
        return df
    
    @staticmethod
    def save_parquet(df: pd.DataFrame, path: Union[str, Path], compression: str = 'snappy'):
        """Parquet 파일 저장"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, compression=compression, index=False)
        logger.info(f"Parquet 파일 저장: {path} (rows: {len(df)})")
    
    @staticmethod
    def load_pickle(path: Union[str, Path]) -> Any:
        """Pickle 파일 로드"""
        path = Path(path)
        with open(path, 'rb') as f:
            data = pickle.load(f)
        return data
    
    @staticmethod
    def save_pickle(data: Any, path: Union[str, Path]):
        """Pickle 파일 저장"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(f"Pickle 파일 저장: {path}")
    
    @staticmethod
    def load_csv(path: Union[str, Path], **kwargs) -> pd.DataFrame:
        """CSV 파일 로드"""
        path = Path(path)
        df = pd.read_csv(path, **kwargs)
        logger.info(f"CSV 파일 로드: {path} (rows: {len(df)})")
        return df
    
    @staticmethod
    def save_csv(df: pd.DataFrame, path: Union[str, Path], **kwargs):
        """CSV 파일 저장"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False, **kwargs)
        logger.info(f"CSV 파일 저장: {path} (rows: {len(df)})")
    
    @staticmethod
    def ensure_dir(path: Union[str, Path]) -> Path:
        """디렉토리 생성 (없으면)"""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    @staticmethod
    def list_files(directory: Union[str, Path], pattern: str = "*") -> list:
        """디렉토리 내 파일 목록"""
        directory = Path(directory)
        return sorted(directory.glob(pattern))