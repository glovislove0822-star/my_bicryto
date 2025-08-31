"""펀딩 레이트 수집 및 분석 모듈"""

import pandas as pd
import numpy as np
import duckdb
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
import asyncio
import aiohttp
import logging
from binance.um_futures import UMFutures
from binance.error import ClientError
import time

from ..utils.logging import Logger
from ..utils.time import TimeUtils
from ..utils.io import IOUtils

logger = Logger.get_logger(__name__)

class FundingRateCollector:
    """펀딩 레이트 수집 및 분석 클래스"""
    
    def __init__(self, 
                 db_path: str = "data/trading.db",
                 api_key: Optional[str] = None,
                 api_secret: Optional[str] = None):
        """
        Args:
            db_path: DuckDB 데이터베이스 경로
            api_key: 바이낸스 API 키
            api_secret: 바이낸스 API 시크릿
        """
        self.db_path = Path(db_path)
        self.conn = duckdb.connect(str(self.db_path))
        self.client = UMFutures(key=api_key, secret=api_secret)
        self.time_utils = TimeUtils()
        
        # 펀딩 레이트 설정
        self.funding_interval = 8  # 8시간마다
        self.funding_times = [0, 8, 16]  # UTC 시간
        
        # 펀딩 차익거래 파라미터
        self.arb_params = {
            'min_funding_rate': 0.01,  # 최소 펀딩 레이트 (1%)
            'min_spread': 0.0005,  # 최소 스프레드
            'holding_period': 8,  # 홀딩 기간 (시간)
            'position_ratio': 0.5  # 헤지 비율
        }
    
    async def collect_funding_history(self,
                                     symbol: str,
                                     start_time: datetime,
                                     end_time: datetime) -> pd.DataFrame:
        """과거 펀딩 레이트 수집
        
        Args:
            symbol: 심볼
            start_time: 시작 시간
            end_time: 종료 시간
        
        Returns:
            펀딩 레이트 데이터프레임
        """
        
        all_funding = []
        current_start = self.time_utils.to_unix_ms(start_time)
        end_ms = self.time_utils.to_unix_ms(end_time)
        
        logger.info(f"{symbol} 펀딩 레이트 수집 시작")
        
        while current_start < end_ms:
            try:
                # API 호출
                funding_data = self.client.funding_rate(
                    symbol=symbol,
                    startTime=current_start,
                    endTime=min(current_start + 100 * 8 * 3600 * 1000, end_ms),
                    limit=1000
                )
                
                if not funding_data:
                    break
                
                all_funding.extend(funding_data)
                
                # 다음 시작점
                current_start = funding_data[-1]['fundingTime'] + 1
                
                await asyncio.sleep(0.1)
                
            except ClientError as e:
                logger.error(f"펀딩 레이트 API 에러: {e}")
                if e.error_code == -1003:
                    await asyncio.sleep(60)
                else:
                    raise
        
        # DataFrame 변환
        if all_funding:
            df = pd.DataFrame(all_funding)
            df['symbol'] = symbol
            df['ts'] = pd.to_datetime(df['fundingTime'], unit='ms')
            df['funding_rate'] = df['fundingRate'].astype(float)
            df['mark_price'] = df.get('markPrice', 0).astype(float)
            
            # 필요 컬럼만 선택
            df = df[['symbol', 'ts', 'funding_rate', 'mark_price']]
            
            # DB 저장
            self._save_funding_data(df)
            
            logger.info(f"{symbol} 펀딩 레이트 수집 완료: {len(df)} records")
            return df
        
        return pd.DataFrame()
    
    async def get_current_funding(self, symbols: List[str]) -> pd.DataFrame:
        """현재 펀딩 레이트 조회
        
        Args:
            symbols: 심볼 리스트
        
        Returns:
            현재 펀딩 레이트 데이터프레임
        """
        
        funding_data = []
        
        for symbol in symbols:
            try:
                # 현재 펀딩 레이트
                premium = self.client.premium_index(symbol=symbol)
                
                funding_data.append({
                    'symbol': symbol,
                    'ts': datetime.now(),
                    'funding_rate': float(premium['lastFundingRate']),
                    'next_funding_time': pd.to_datetime(premium['nextFundingTime'], unit='ms'),
                    'mark_price': float(premium['markPrice']),
                    'index_price': float(premium['indexPrice']),
                    'estimated_rate': float(premium.get('estimatedSettlePrice', 0))
                })
                
            except Exception as e:
                logger.error(f"{symbol} 현재 펀딩 레이트 조회 실패: {e}")
        
        return pd.DataFrame(funding_data)
    
    def analyze_funding_patterns(self, 
                                symbol: str,
                                lookback_days: int = 30) -> Dict:
        """펀딩 레이트 패턴 분석
        
        Args:
            symbol: 심볼
            lookback_days: 분석 기간 (일)
        
        Returns:
            분석 결과 딕셔너리
        """
        
        # 데이터 로드
        end_time = datetime.now()
        start_time = end_time - timedelta(days=lookback_days)
        
        query = f"""
            SELECT 
                ts,
                funding_rate,
                mark_price
            FROM trading.funding
            WHERE symbol = '{symbol}'
                AND ts >= '{start_time}'
                AND ts <= '{end_time}'
            ORDER BY ts
        """
        
        df = self.conn.execute(query).df()
        
        if df.empty:
            logger.warning(f"{symbol} 펀딩 데이터 없음")
            return {}
        
        df['ts'] = pd.to_datetime(df['ts'])
        df.set_index('ts', inplace=True)
        
        # 기본 통계
        stats = {
            'mean': df['funding_rate'].mean(),
            'std': df['funding_rate'].std(),
            'min': df['funding_rate'].min(),
            'max': df['funding_rate'].max(),
            'skew': df['funding_rate'].skew(),
            'kurtosis': df['funding_rate'].kurtosis()
        }
        
        # 시간대별 패턴
        df['hour'] = df.index.hour
        hourly_mean = df.groupby('hour')['funding_rate'].mean()
        stats['hourly_pattern'] = hourly_mean.to_dict()
        
        # 트렌드 분석
        df['funding_ma_3'] = df['funding_rate'].rolling(3).mean()  # 24시간 (3 * 8h)
        df['funding_ma_7'] = df['funding_rate'].rolling(7).mean()  # 56시간
        
        # 현재 트렌드
        if len(df) > 7:
            current = df['funding_rate'].iloc[-1]
            ma_3 = df['funding_ma_3'].iloc[-1]
            ma_7 = df['funding_ma_7'].iloc[-1]
            
            stats['current_rate'] = current
            stats['vs_ma_3'] = (current - ma_3) / abs(ma_3) if ma_3 != 0 else 0
            stats['vs_ma_7'] = (current - ma_7) / abs(ma_7) if ma_7 != 0 else 0
            
            # 트렌드 방향
            if current > ma_3 > ma_7:
                stats['trend'] = 'strongly_positive'
            elif current > ma_3:
                stats['trend'] = 'positive'
            elif current < ma_3 < ma_7:
                stats['trend'] = 'strongly_negative'
            elif current < ma_3:
                stats['trend'] = 'negative'
            else:
                stats['trend'] = 'neutral'
        
        # 극단값 빈도
        extreme_threshold = stats['mean'] + 2 * stats['std']
        stats['extreme_positive_pct'] = (df['funding_rate'] > extreme_threshold).mean()
        stats['extreme_negative_pct'] = (df['funding_rate'] < -extreme_threshold).mean()
        
        # 펀딩 비용 계산 (연율화)
        stats['annual_funding_cost'] = stats['mean'] * 3 * 365  # 하루 3번 * 365일
        
        # 차익거래 기회
        arb_opportunities = self._identify_arbitrage_opportunities(df)
        stats['arb_opportunities'] = arb_opportunities
        
        return stats
    
    def _identify_arbitrage_opportunities(self, df: pd.DataFrame) -> Dict:
        """펀딩 차익거래 기회 식별
        
        Args:
            df: 펀딩 데이터프레임
        
        Returns:
            차익거래 기회 정보
        """
        
        opportunities = {
            'total_count': 0,
            'long_funding_arb': [],
            'short_funding_arb': [],
            'funding_harvesting': []
        }
        
        # 극단적 펀딩 레이트 찾기
        threshold = self.arb_params['min_funding_rate']
        
        # Long 펀딩 차익 (음수 펀딩 - 롱 포지션 유리)
        negative_funding = df[df['funding_rate'] < -threshold]
        for idx, row in negative_funding.iterrows():
            opportunities['long_funding_arb'].append({
                'ts': idx,
                'funding_rate': row['funding_rate'],
                'expected_profit': abs(row['funding_rate']) * self.arb_params['position_ratio']
            })
        
        # Short 펀딩 차익 (양수 펀딩 - 숏 포지션 유리)
        positive_funding = df[df['funding_rate'] > threshold]
        for idx, row in positive_funding.iterrows():
            opportunities['short_funding_arb'].append({
                'ts': idx,
                'funding_rate': row['funding_rate'],
                'expected_profit': row['funding_rate'] * self.arb_params['position_ratio']
            })
        
        # 펀딩 하베스팅 (지속적인 편향)
        rolling_sum = df['funding_rate'].rolling(9).sum()  # 3일 (9 * 8시간)
        harvesting = rolling_sum[abs(rolling_sum) > threshold * 3]
        
        for idx, cumulative_funding in harvesting.items():
            opportunities['funding_harvesting'].append({
                'ts': idx,
                'cumulative_funding': cumulative_funding,
                'direction': 'short' if cumulative_funding > 0 else 'long',
                'expected_profit': abs(cumulative_funding) * self.arb_params['position_ratio']
            })
        
        opportunities['total_count'] = (
            len(opportunities['long_funding_arb']) +
            len(opportunities['short_funding_arb']) +
            len(opportunities['funding_harvesting'])
        )
        
        return opportunities
    
    def calculate_funding_adjusted_returns(self,
                                          symbol: str,
                                          returns: pd.Series,
                                          position_side: str = 'long') -> pd.Series:
        """펀딩 조정 수익률 계산
        
        Args:
            symbol: 심볼
            returns: 원시 수익률
            position_side: 포지션 방향 ('long' or 'short')
        
        Returns:
            펀딩 조정 수익률
        """
        
        # 펀딩 데이터 로드
        query = f"""
            SELECT 
                ts,
                funding_rate
            FROM trading.funding
            WHERE symbol = '{symbol}'
            ORDER BY ts
        """
        
        funding_df = self.conn.execute(query).df()
        
        if funding_df.empty:
            return returns
        
        funding_df['ts'] = pd.to_datetime(funding_df['ts'])
        funding_df.set_index('ts', inplace=True)
        
        # 수익률과 병합
        combined = pd.DataFrame({'returns': returns})
        combined = pd.merge_asof(
            combined.sort_index(),
            funding_df.sort_index(),
            left_index=True,
            right_index=True,
            direction='backward'
        )
        
        # 펀딩 비용 계산
        if position_side == 'long':
            # 롱 포지션은 펀딩을 지불 (양수 펀딩 시)
            combined['funding_cost'] = combined['funding_rate']
        else:
            # 숏 포지션은 펀딩을 받음 (양수 펀딩 시)
            combined['funding_cost'] = -combined['funding_rate']
        
        # 조정 수익률
        combined['adjusted_returns'] = combined['returns'] - combined['funding_cost']
        
        return combined['adjusted_returns']
    
    def generate_funding_signals(self,
                                symbol: str,
                                current_funding: float,
                                lookback_periods: int = 20) -> Dict:
        """펀딩 기반 트레이딩 시그널 생성
        
        Args:
            symbol: 심볼
            current_funding: 현재 펀딩 레이트
            lookback_periods: 룩백 기간
        
        Returns:
            시그널 정보
        """
        
        # 과거 펀딩 통계
        query = f"""
            SELECT 
                funding_rate
            FROM trading.funding
            WHERE symbol = '{symbol}'
            ORDER BY ts DESC
            LIMIT {lookback_periods}
        """
        
        result = self.conn.execute(query).fetchall()
        
        if not result:
            return {'signal': 'neutral', 'confidence': 0}
        
        historical_rates = [r[0] for r in result]
        
        # 통계 계산
        mean_rate = np.mean(historical_rates)
        std_rate = np.std(historical_rates)
        
        if std_rate == 0:
            return {'signal': 'neutral', 'confidence': 0}
        
        # Z-score
        z_score = (current_funding - mean_rate) / std_rate
        
        # 시그널 생성
        signal = {
            'current_funding': current_funding,
            'mean_funding': mean_rate,
            'std_funding': std_rate,
            'z_score': z_score,
            'signal': 'neutral',
            'confidence': 0,
            'reason': ''
        }
        
        # 극단적 펀딩 레이트
        if z_score > 2:
            # 매우 높은 펀딩 - 숏 유리
            signal['signal'] = 'short'
            signal['confidence'] = min(1.0, z_score / 3)
            signal['reason'] = 'extreme_positive_funding'
            
        elif z_score < -2:
            # 매우 낮은 펀딩 - 롱 유리
            signal['signal'] = 'long'
            signal['confidence'] = min(1.0, abs(z_score) / 3)
            signal['reason'] = 'extreme_negative_funding'
            
        elif abs(current_funding) > self.arb_params['min_funding_rate']:
            # 차익거래 기회
            if current_funding > 0:
                signal['signal'] = 'funding_arb_short'
                signal['confidence'] = min(1.0, current_funding / 0.02)
                signal['reason'] = 'positive_funding_arbitrage'
            else:
                signal['signal'] = 'funding_arb_long'
                signal['confidence'] = min(1.0, abs(current_funding) / 0.02)
                signal['reason'] = 'negative_funding_arbitrage'
        
        # 추가 필터: 지속성 체크
        recent_rates = historical_rates[:5]
        if all(r > threshold for r in recent_rates for threshold in [mean_rate + std_rate]):
            signal['persistence'] = 'consistent_high'
            signal['confidence'] *= 1.2
        elif all(r < threshold for r in recent_rates for threshold in [mean_rate - std_rate]):
            signal['persistence'] = 'consistent_low'
            signal['confidence'] *= 1.2
        else:
            signal['persistence'] = 'mixed'
        
        signal['confidence'] = min(1.0, signal['confidence'])
        
        return signal
    
    def _save_funding_data(self, df: pd.DataFrame):
        """펀딩 데이터 DB 저장"""
        
        if df.empty:
            return
        
        # 임시 테이블
        temp_table = f"temp_funding_{int(time.time())}"
        
        self.conn.execute(f"CREATE TEMP TABLE {temp_table} AS SELECT * FROM df")
        
        # UPSERT
        self.conn.execute(f"""
            INSERT OR REPLACE INTO trading.funding
            SELECT * FROM {temp_table}
        """)
        
        self.conn.execute(f"DROP TABLE {temp_table}")
        
        logger.info(f"펀딩 데이터 저장: {len(df)} records")
    
    def monitor_funding_divergence(self, symbols: List[str]) -> pd.DataFrame:
        """여러 심볼 간 펀딩 다이버전스 모니터링
        
        Args:
            symbols: 심볼 리스트
        
        Returns:
            다이버전스 정보
        """
        
        divergence_data = []
        
        # 각 심볼의 현재 펀딩
        for i, symbol1 in enumerate(symbols):
            for symbol2 in symbols[i+1:]:
                # 최근 펀딩 레이트
                query = f"""
                    SELECT 
                        f1.ts,
                        f1.funding_rate as rate1,
                        f2.funding_rate as rate2
                    FROM 
                        (SELECT * FROM trading.funding WHERE symbol = '{symbol1}' ORDER BY ts DESC LIMIT 10) f1
                    JOIN 
                        (SELECT * FROM trading.funding WHERE symbol = '{symbol2}' ORDER BY ts DESC LIMIT 10) f2
                    ON f1.ts = f2.ts
                """
                
                result = self.conn.execute(query).df()
                
                if not result.empty:
                    spread = result['rate1'] - result['rate2']
                    
                    divergence_data.append({
                        'pair': f"{symbol1}/{symbol2}",
                        'symbol1': symbol1,
                        'symbol2': symbol2,
                        'funding_spread': spread.iloc[-1] if len(spread) > 0 else 0,
                        'spread_mean': spread.mean(),
                        'spread_std': spread.std(),
                        'z_score': (spread.iloc[-1] - spread.mean()) / spread.std() if spread.std() > 0 else 0
                    })
        
        return pd.DataFrame(divergence_data)

# CLI 실행용
async def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='펀딩 레이트 수집 및 분석')
    parser.add_argument('--symbols', nargs='+', required=True, help='심볼 리스트')
    parser.add_argument('--action', choices=['collect', 'analyze', 'monitor'], 
                       default='analyze', help='실행 작업')
    parser.add_argument('--start', help='시작 날짜')
    parser.add_argument('--end', help='종료 날짜')
    parser.add_argument('--db', default='data/trading.db', help='DB 경로')
    
    args = parser.parse_args()
    
    collector = FundingRateCollector(db_path=args.db)
    
    if args.action == 'collect':
        # 과거 데이터 수집
        for symbol in args.symbols:
            await collector.collect_funding_history(
                symbol=symbol,
                start_time=pd.to_datetime(args.start),
                end_time=pd.to_datetime(args.end) if args.end else datetime.now()
            )
    
    elif args.action == 'analyze':
        # 패턴 분석
        for symbol in args.symbols:
            analysis = collector.analyze_funding_patterns(symbol)
            
            logger.info(f"\n=== {symbol} 펀딩 분석 ===")
            logger.info(f"평균: {analysis.get('mean', 0):.4%}")
            logger.info(f"표준편차: {analysis.get('std', 0):.4%}")
            logger.info(f"현재: {analysis.get('current_rate', 0):.4%}")
            logger.info(f"트렌드: {analysis.get('trend', 'N/A')}")
            logger.info(f"연간 비용: {analysis.get('annual_funding_cost', 0):.2%}")
            logger.info(f"차익거래 기회: {analysis.get('arb_opportunities', {}).get('total_count', 0)}")
    
    elif args.action == 'monitor':
        # 실시간 모니터링
        current = await collector.get_current_funding(args.symbols)
        
        logger.info("\n=== 현재 펀딩 레이트 ===")
        for _, row in current.iterrows():
            logger.info(f"{row['symbol']}: {row['funding_rate']:.4%} "
                       f"(다음: {row['next_funding_time']})")
        
        # 다이버전스
        divergence = collector.monitor_funding_divergence(args.symbols)
        if not divergence.empty:
            logger.info("\n=== 펀딩 다이버전스 ===")
            for _, row in divergence.iterrows():
                if abs(row['z_score']) > 2:
                    logger.warning(f"{row['pair']}: "
                                 f"스프레드={row['funding_spread']:.4%} "
                                 f"(Z={row['z_score']:.2f})")

if __name__ == "__main__":
    asyncio.run(main())