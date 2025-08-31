"""슬리피지 캘리브레이션"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging
from collections import deque
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

from ..utils.logging import Logger

logger = Logger.get_logger(__name__)

class SlippageCalibrator:
    """슬리피지 캘리브레이터
    
    실제 슬리피지 측정 및 모델 캘리브레이션
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Args:
            config: 캘리브레이션 설정
        """
        self.config = config or {}
        
        # 슬리피지 모델 파라미터
        self.model_params = {
            'linear_k': 0.0001,  # 선형 계수
            'sqrt_k': 0.001,     # 제곱근 계수
            'impact_k': 0.0005,  # 시장 충격 계수
            'spread_factor': 0.5, # 스프레드 팩터
            'volatility_factor': 0.1  # 변동성 팩터
        }
        
        if config and 'initial_params' in config:
            self.model_params.update(config['initial_params'])
        
        # 측정 데이터
        self.measurements = deque(maxlen=10000)
        
        # 캘리브레이션 모델
        self.calibration_models = {}
        
        # 통계
        self.statistics = {
            'total_measurements': 0,
            'avg_slippage_bps': 0,
            'max_slippage_bps': 0,
            'model_accuracy': 0,
            'last_calibration': None
        }
        
        # 심볼별 파라미터
        self.symbol_params = {}
    
    def record_execution(self,
                        symbol: str,
                        side: str,
                        intended_price: float,
                        executed_price: float,
                        quantity: float,
                        spread_bps: float,
                        depth: float,
                        volatility: float,
                        execution_time_ms: float):
        """실행 기록
        
        Args:
            symbol: 심볼
            side: 방향 (buy/sell)
            intended_price: 의도 가격
            executed_price: 실행 가격
            quantity: 수량
            spread_bps: 스프레드 (bps)
            depth: 호가 심도
            volatility: 변동성
            execution_time_ms: 실행 시간 (밀리초)
        """
        
        # 슬리피지 계산
        if side == 'buy':
            slippage = (executed_price - intended_price) / intended_price
        else:
            slippage = (intended_price - executed_price) / intended_price
        
        slippage_bps = slippage * 10000
        
        # 측정 데이터 저장
        measurement = {
            'timestamp': datetime.now(),
            'symbol': symbol,
            'side': side,
            'intended_price': intended_price,
            'executed_price': executed_price,
            'quantity': quantity,
            'notional': quantity * executed_price,
            'slippage_bps': slippage_bps,
            'spread_bps': spread_bps,
            'depth': depth,
            'volatility': volatility,
            'execution_time_ms': execution_time_ms,
            'size_depth_ratio': quantity / depth if depth > 0 else 0
        }
        
        self.measurements.append(measurement)
        
        # 통계 업데이트
        self.statistics['total_measurements'] += 1
        
        # 이동 평균 업데이트
        n = self.statistics['total_measurements']
        self.statistics['avg_slippage_bps'] = (
            (self.statistics['avg_slippage_bps'] * (n - 1) + abs(slippage_bps)) / n
        )
        
        self.statistics['max_slippage_bps'] = max(
            self.statistics['max_slippage_bps'],
            abs(slippage_bps)
        )
        
        # 임계값 체크
        if abs(slippage_bps) > 10:  # 10 bps 이상
            logger.warning(f"High slippage: {symbol} {side} {slippage_bps:.1f}bps")
        
        # 자동 캘리브레이션 체크
        if self.statistics['total_measurements'] % 100 == 0:
            self.calibrate()
    
    def calibrate(self, min_samples: int = 50) -> Dict:
        """모델 캘리브레이션
        
        Args:
            min_samples: 최소 샘플 수
            
        Returns:
            캘리브레이션 결과
        """
        
        if len(self.measurements) < min_samples:
            logger.warning(f"샘플 부족: {len(self.measurements)} < {min_samples}")
            return {'success': False, 'reason': 'insufficient_samples'}
        
        logger.info("슬리피지 모델 캘리브레이션 시작")
        
        # DataFrame 변환
        df = pd.DataFrame(list(self.measurements))
        
        # 전체 모델 캘리브레이션
        global_model = self._calibrate_global_model(df)
        
        # 심볼별 캘리브레이션
        symbol_models = {}
        
        for symbol in df['symbol'].unique():
            symbol_df = df[df['symbol'] == symbol]
            
            if len(symbol_df) >= min_samples // 5:  # 심볼별 최소 샘플
                symbol_models[symbol] = self._calibrate_symbol_model(symbol_df)
        
        # 모델 저장
        self.calibration_models = {
            'global': global_model,
            'symbols': symbol_models
        }
        
        # 정확도 평가
        accuracy = self._evaluate_model_accuracy(df)
        self.statistics['model_accuracy'] = accuracy
        self.statistics['last_calibration'] = datetime.now()
        
        logger.info(f"캘리브레이션 완료: 정확도={accuracy:.2%}")
        
        return {
            'success': True,
            'global_params': global_model['params'],
            'symbol_count': len(symbol_models),
            'accuracy': accuracy,
            'samples_used': len(df)
        }
    
    def _calibrate_global_model(self, df: pd.DataFrame) -> Dict:
        """전역 모델 캘리브레이션
        
        Args:
            df: 측정 데이터
            
        Returns:
            캘리브레이션된 모델
        """
        
        # 특징 추출
        X = df[['size_depth_ratio', 'spread_bps', 'volatility', 'execution_time_ms']].values
        y = df['slippage_bps'].abs().values
        
        # 다항 특징 생성
        poly = PolynomialFeatures(degree=2, include_bias=False)
        X_poly = poly.fit_transform(X)
        
        # 선형 회귀
        model = LinearRegression()
        model.fit(X_poly, y)
        
        # 계수 추출
        feature_names = poly.get_feature_names_out(['size_depth', 'spread', 'vol', 'time'])
        coefficients = dict(zip(feature_names, model.coef_))
        
        # 주요 파라미터 업데이트
        new_params = {
            'linear_k': coefficients.get('size_depth', 0.0001),
            'spread_factor': coefficients.get('spread', 0.5),
            'volatility_factor': coefficients.get('vol', 0.1),
            'intercept': model.intercept_
        }
        
        # 스무딩 (급격한 변화 방지)
        alpha = 0.3  # 학습률
        
        for key in ['linear_k', 'spread_factor', 'volatility_factor']:
            if key in new_params:
                self.model_params[key] = (
                    alpha * new_params[key] + 
                    (1 - alpha) * self.model_params.get(key, new_params[key])
                )
        
        return {
            'model': model,
            'poly': poly,
            'params': self.model_params.copy(),
            'coefficients': coefficients,
            'r2_score': model.score(X_poly, y)
        }
    
    def _calibrate_symbol_model(self, df: pd.DataFrame) -> Dict:
        """심볼별 모델 캘리브레이션
        
        Args:
            df: 심볼 데이터
            
        Returns:
            캘리브레이션된 모델
        """
        
        symbol = df['symbol'].iloc[0]
        
        # 간단한 통계 기반 모델
        stats = {
            'mean_slippage': df['slippage_bps'].abs().mean(),
            'std_slippage': df['slippage_bps'].abs().std(),
            'p95_slippage': df['slippage_bps'].abs().quantile(0.95),
            'spread_sensitivity': self._calculate_spread_sensitivity(df),
            'size_sensitivity': self._calculate_size_sensitivity(df)
        }
        
        # 심볼별 파라미터
        self.symbol_params[symbol] = {
            'base_slippage': stats['mean_slippage'],
            'spread_multiplier': stats['spread_sensitivity'],
            'size_multiplier': stats['size_sensitivity']
        }
        
        return stats
    
    def _calculate_spread_sensitivity(self, df: pd.DataFrame) -> float:
        """스프레드 민감도 계산
        
        Args:
            df: 데이터
            
        Returns:
            스프레드 민감도
        """
        
        if len(df) < 10:
            return 0.5
        
        # 스프레드와 슬리피지 상관관계
        correlation = df['spread_bps'].corr(df['slippage_bps'].abs())
        
        # 회귀 계수
        X = df[['spread_bps']].values
        y = df['slippage_bps'].abs().values
        
        model = LinearRegression()
        model.fit(X, y)
        
        return model.coef_[0]
    
    def _calculate_size_sensitivity(self, df: pd.DataFrame) -> float:
        """크기 민감도 계산
        
        Args:
            df: 데이터
            
        Returns:
            크기 민감도
        """
        
        if len(df) < 10:
            return 0.0001
        
        # 크기/심도 비율과 슬리피지 관계
        X = df[['size_depth_ratio']].values
        y = df['slippage_bps'].abs().values
        
        model = LinearRegression()
        model.fit(X, y)
        
        return model.coef_[0]
    
    def _evaluate_model_accuracy(self, df: pd.DataFrame) -> float:
        """모델 정확도 평가
        
        Args:
            df: 데이터
            
        Returns:
            정확도 (0-1)
        """
        
        if 'global' not in self.calibration_models:
            return 0
        
        model_data = self.calibration_models['global']
        model = model_data['model']
        poly = model_data['poly']
        
        # 예측
        X = df[['size_depth_ratio', 'spread_bps', 'volatility', 'execution_time_ms']].values
        X_poly = poly.transform(X)
        
        y_true = df['slippage_bps'].abs().values
        y_pred = model.predict(X_poly)
        
        # MAPE (Mean Absolute Percentage Error)
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1))) * 100
        
        # 정확도 (100% - MAPE, 0-1 범위)
        accuracy = max(0, min(1, 1 - mape / 100))
        
        return accuracy
    
    def estimate_slippage(self,
                         symbol: str,
                         side: str,
                         quantity: float,
                         spread_bps: float,
                         depth: float,
                         volatility: float) -> float:
        """슬리피지 추정
        
        Args:
            symbol: 심볼
            side: 방향
            quantity: 수량
            spread_bps: 스프레드
            depth: 심도
            volatility: 변동성
            
        Returns:
            예상 슬리피지 (bps)
        """
        
        # 심볼별 모델 사용
        if symbol in self.symbol_params:
            params = self.symbol_params[symbol]
            
            base_slippage = params['base_slippage']
            spread_component = spread_bps * params['spread_multiplier']
            size_component = (quantity / depth) * params['size_multiplier'] if depth > 0 else 0
            
            estimated_slippage = base_slippage + spread_component + size_component
            
        else:
            # 전역 모델 사용
            size_depth_ratio = quantity / depth if depth > 0 else 0
            
            # 선형 모델
            linear_component = self.model_params['linear_k'] * size_depth_ratio * 10000
            
            # 스프레드 컴포넌트
            spread_component = spread_bps * self.model_params['spread_factor']
            
            # 변동성 컴포넌트
            volatility_component = volatility * self.model_params['volatility_factor'] * 10000
            
            estimated_slippage = linear_component + spread_component + volatility_component
        
        # 최소값 보장
        min_slippage = spread_bps * 0.5  # 최소 스프레드의 절반
        
        return max(estimated_slippage, min_slippage)
    
    def get_statistics(self) -> Dict:
        """통계 조회
        
        Returns:
            통계 딕셔너리
        """
        
        stats = self.statistics.copy()
        
        # 최근 측정 분석
        if self.measurements:
            recent = list(self.measurements)[-100:]  # 최근 100개
            
            recent_df = pd.DataFrame(recent)
            
            stats['recent_avg_slippage'] = recent_df['slippage_bps'].abs().mean()
            stats['recent_max_slippage'] = recent_df['slippage_bps'].abs().max()
            
            # 심볼별 통계
            symbol_stats = {}
            
            for symbol in recent_df['symbol'].unique():
                symbol_data = recent_df[recent_df['symbol'] == symbol]
                
                symbol_stats[symbol] = {
                    'count': len(symbol_data),
                    'avg_slippage': symbol_data['slippage_bps'].abs().mean(),
                    'max_slippage': symbol_data['slippage_bps'].abs().max()
                }
            
            stats['symbol_stats'] = symbol_stats
        
        # 모델 파라미터
        stats['model_params'] = self.model_params.copy()
        
        return stats
    
    def generate_report(self) -> pd.DataFrame:
        """리포트 생성
        
        Returns:
            리포트 DataFrame
        """
        
        if not self.measurements:
            return pd.DataFrame()
        
        df = pd.DataFrame(list(self.measurements))
        
        # 시간대별 집계
        df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
        
        hourly_stats = df.groupby('hour').agg({
            'slippage_bps': ['mean', 'std', 'max'],
            'execution_time_ms': 'mean'
        })
        
        # 심볼별 집계
        symbol_stats = df.groupby('symbol').agg({
            'slippage_bps': ['mean', 'std', 'max', 'count'],
            'spread_bps': 'mean',
            'execution_time_ms': 'mean'
        })
        
        # 크기별 집계
        df['size_bucket'] = pd.qcut(df['notional'], q=5, labels=['XS', 'S', 'M', 'L', 'XL'])
        
        size_stats = df.groupby('size_bucket').agg({
            'slippage_bps': ['mean', 'std', 'max'],
            'size_depth_ratio': 'mean'
        })
        
        logger.info("슬리피지 리포트 생성 완료")
        
        return {
            'hourly': hourly_stats,
            'symbol': symbol_stats,
            'size': size_stats,
            'summary': self.get_statistics()
        }

# CLI 실행용  
def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='슬리피지 캘리브레이션')
    parser.add_argument('--action', choices=['calibrate', 'report', 'test'], 
                       required=True, help='액션')
    parser.add_argument('--data', help='측정 데이터 파일')
    parser.add_argument('--output', help='출력 파일')
    
    args = parser.parse_args()
    
    # 캘리브레이터 생성
    calibrator = SlippageCalibrator()
    
    if args.action == 'calibrate':
        # 데이터 로드
        if args.data:
            df = pd.read_csv(args.data)
            
            # 측정 데이터 추가
            for _, row in df.iterrows():
                calibrator.record_execution(
                    symbol=row['symbol'],
                    side=row['side'],
                    intended_price=row['intended_price'],
                    executed_price=row['executed_price'],
                    quantity=row['quantity'],
                    spread_bps=row.get('spread_bps', 1),
                    depth=row.get('depth', 10000),
                    volatility=row.get('volatility', 0.01),
                    execution_time_ms=row.get('execution_time_ms', 50)
                )
        
        # 캘리브레이션
        result = calibrator.calibrate()
        print(json.dumps(result, indent=2, default=str))
        
    elif args.action == 'report':
        # 리포트 생성
        report = calibrator.generate_report()
        
        if args.output:
            # 엑셀로 저장
            with pd.ExcelWriter(args.output) as writer:
                for name, df in report.items():
                    if isinstance(df, pd.DataFrame):
                        df.to_excel(writer, sheet_name=name)
        
        # 요약 출력
        stats = calibrator.get_statistics()
        print("\n슬리피지 통계:")
        print(f"총 측정: {stats['total_measurements']}")
        print(f"평균 슬리피지: {stats['avg_slippage_bps']:.2f} bps")
        print(f"최대 슬리피지: {stats['max_slippage_bps']:.2f} bps")
        print(f"모델 정확도: {stats['model_accuracy']:.2%}")
        
    elif args.action == 'test':
        # 테스트 추정
        estimated = calibrator.estimate_slippage(
            symbol='BTCUSDT',
            side='buy',
            quantity=1.0,
            spread_bps=1.0,
            depth=10000,
            volatility=0.01
        )
        
        print(f"예상 슬리피지: {estimated:.2f} bps")

if __name__ == "__main__":
    main()