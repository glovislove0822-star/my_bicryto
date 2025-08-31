"""특징 버퍼 최적화"""

import numpy as np
from typing import Dict, List, Optional
from collections import deque
import pandas as pd

class FeatureBuffer:
    """메모리 효율적 특징 버퍼"""
    
    def __init__(self, 
                 features: List[str],
                 capacity: int = 1000,
                 dtype: np.dtype = np.float32):
        """
        Args:
            features: 특징 이름 리스트
            capacity: 버퍼 용량
            dtype: 데이터 타입
        """
        self.features = features
        self.capacity = capacity
        self.dtype = dtype
        
        # 구조화된 배열
        self.buffer = np.zeros(
            capacity,
            dtype=[(f, dtype) for f in features]
        )
        
        self.position = 0
        self.size = 0
    
    def append(self, feature_dict: Dict[str, float]):
        """특징 추가"""
        
        for feature, value in feature_dict.items():
            if feature in self.features:
                self.buffer[self.position][feature] = value
        
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def get_latest(self, n: int = 1) -> np.ndarray:
        """최신 N개 조회"""
        
        n = min(n, self.size)
        
        if n == 0:
            return np.array([])
        
        if self.size < self.capacity:
            return self.buffer[max(0, self.size-n):self.size]
        else:
            # 순환 버퍼 처리
            start = (self.position - n) % self.capacity
            
            if start < self.position:
                return self.buffer[start:self.position]
            else:
                return np.concatenate([
                    self.buffer[start:],
                    self.buffer[:self.position]
                ])
    
    def to_dataframe(self) -> pd.DataFrame:
        """DataFrame 변환"""
        
        if self.size == 0:
            return pd.DataFrame(columns=self.features)
        
        data = self.get_latest(self.size)
        return pd.DataFrame(data)