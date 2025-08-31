"""메타 라벨링 모듈"""

import pandas as pd
import numpy as np
import duckdb
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from datetime import datetime
import logging
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_curve, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

from ..utils.logging import Logger
from ..utils.io import IOUtils
from ..utils.math import MathUtils

logger = Logger.get_logger(__name__)

class MetaLabeler:
    """메타 라벨링 클래스
    
    1차 모델의 신호에 대해 실행 여부를 결정하는 2차 분류기
    """
    
    def __init__(self, db_path: str = "data/trading.db"):
        """
        Args:
            db_path: DuckDB 데이터베이스 경로
        """
        self.db_path = Path(db_path)
        self.conn = duckdb.connect(str(self.db_path))
        self.math_utils = MathUtils()
        
        # 메타 라벨링 설정
        self.config = {
            'primary_model': 'trend_following',  # 1차 모델 타입
            'min_precision': 0.6,  # 최소 정밀도
            'min_samples': 100,    # 최소 샘플 수
            'lookback_bars': 20,   # 룩백 기간
            'use_purged_cv': True, # Purged CV 사용
            'embargo_bars': 2,     # 엠바고 기간
            'confidence_threshold': 0.5  # 신뢰도 임계값
        }
        
        # 1차 모델 규칙
        self.primary_rules = {
            'trend_following': self._trend_following_signal,
            'mean_reversion': self._mean_reversion_signal,
            'momentum': self._momentum_signal,
            'breakout': self._breakout_signal
        }
    
    def generate_meta_labels(self,
                            timeframe: str = '3m',
                            symbols: Optional[List[str]] = None,
                            primary_model: str = 'trend_following',
                            start_date: Optional[str] = None,
                            end_date: Optional[str] = None):
        """메타 라벨 생성
        
        Args:
            timeframe: 시간 프레임
            symbols: 심볼 리스트
            primary_model: 1차 모델 타입
            start_date: 시작 날짜
            end_date: 종료 날짜
        """
        
        self.config['primary_model'] = primary_model
        
        # 심볼 목록
        if symbols is None:
            result = self.conn.execute(f"""
                SELECT DISTINCT symbol
                FROM trading.labels_{timeframe}
            """).fetchall()
            symbols = [r[0] for r in result]
        
        logger.info(f"메타 라벨링 시작: {symbols}")
        logger.info(f"1차 모델: {primary_model}")
        
        for symbol in tqdm(symbols, desc="메타 라벨링"):
            try:
                # 데이터 로드
                df = self._load_data_with_labels(symbol, timeframe, start_date, end_date)
                
                if df.empty or len(df) < self.config['min_samples']:
                    logger.warning(f"{symbol} 데이터 부족, 스킵")
                    continue
                
                # 1차 신호 생성
                df = self._generate_primary_signals(df, primary_model)
                
                # 메타 라벨 생성
                df = self._generate_meta_labels_for_symbol(df)
                
                # 메타 라벨 품질 평가
                metrics = self._evaluate_meta_labels(df)
                
                # DB 업데이트
                self._update_meta_labels(df, symbol, timeframe)
                
                # 결과 로깅
                logger.info(f"{symbol} 메타 라벨링 완료:")
                logger.info(f"  - 정밀도: {metrics['precision']:.3f}")
                logger.info(f"  - 재현율: {metrics['recall']:.3f}")
                logger.info(f"  - F1 스코어: {metrics['f1_score']:.3f}")
                
            except Exception as e:
                logger.error(f"{symbol} 메타 라벨링 실패: {e}")
                continue
        
        logger.info("메타 라벨링 완료!")
        self._print_meta_label_statistics(timeframe)
    
    def _load_data_with_labels(self,
                              symbol: str,
                              timeframe: str,
                              start_date: Optional[str],
                              end_date: Optional[str]) -> pd.DataFrame:
        """라벨과 함께 데이터 로드"""
        
        where_clause = f"WHERE f.symbol = '{symbol}'"
        if start_date:
            where_clause += f" AND f.ts >= '{start_date}'"
        if end_date:
            where_clause += f" AND f.ts <= '{end_date}'"
        
        query = f"""
            SELECT 
                f.*,
                l.label,
                l.pnl_pct,
                l.tp_hit,
                l.sl_hit,
                l.time_exit,
                b.open,
                b.high,
                b.low,
                b.close,
                b.volume
            FROM trading.features_{timeframe} f
            JOIN trading.labels_{timeframe} l
                ON f.symbol = l.symbol AND f.ts = l.ts
            JOIN trading.bars_{timeframe} b
                ON f.symbol = b.symbol AND f.ts = b.open_time
            {where_clause}
            ORDER BY f.ts
        """
        
        df = self.conn.execute(query).df()
        
        if not df.empty:
            df['ts'] = pd.to_datetime(df['ts'])
            df.set_index('ts', inplace=True)
        
        return df
    
    def _generate_primary_signals(self, 
                                 df: pd.DataFrame,
                                 model_type: str) -> pd.DataFrame:
        """1차 모델 신호 생성"""
        
        if model_type not in self.primary_rules:
            raise ValueError(f"Unknown primary model: {model_type}")
        
        # 선택된 규칙으로 신호 생성
        signal_func = self.primary_rules[model_type]
        df = signal_func(df)
        
        return df
    
    def _trend_following_signal(self, df: pd.DataFrame) -> pd.DataFrame:
        """트렌드 팔로잉 신호"""
        
        # TSMOM (Time Series Momentum)
        df['tsmom_signal'] = np.where(df['ret_60'] > 0, 1, 
                                      np.where(df['ret_60'] < 0, -1, 0))
        
        # Donchian 채널 돌파
        df['donchian_long'] = (df['close'] > df['donchian_upper'].shift(1)).astype(int)
        df['donchian_short'] = (df['close'] < df['donchian_lower'].shift(1)).astype(int)
        
        # EMA 크로스
        df['ema_signal'] = np.where(df['ema_fast'] > df['ema_slow'], 1,
                                    np.where(df['ema_fast'] < df['ema_slow'], -1, 0))
        
        # 종합 신호
        df['primary_signal'] = (
            df['tsmom_signal'] * 0.4 +
            (df['donchian_long'] - df['donchian_short']) * 0.3 +
            df['ema_signal'] * 0.3
        )
        
        # 이진화
        df['primary_direction'] = np.where(df['primary_signal'] > 0.3, 1,
                                          np.where(df['primary_signal'] < -0.3, -1, 0))
        
        return df
    
    def _mean_reversion_signal(self, df: pd.DataFrame) -> pd.DataFrame:
        """평균 회귀 신호"""
        
        # RSI 극단값
        df['rsi_oversold'] = (df['rsi_2'] < 20).astype(int)
        df['rsi_overbought'] = (df['rsi_2'] > 80).astype(int)
        
        # Bollinger Band 터치
        df['bb_lower_touch'] = (df['close'] <= df['bb_lower']).astype(int)
        df['bb_upper_touch'] = (df['close'] >= df['bb_upper']).astype(int)
        
        # VWAP 이탈
        df['vwap_oversold'] = (df['vwap_z'] < -1.5).astype(int)
        df['vwap_overbought'] = (df['vwap_z'] > 1.5).astype(int)
        
        # 종합 신호
        df['primary_signal'] = (
            (df['rsi_oversold'] + df['bb_lower_touch'] + df['vwap_oversold']) -
            (df['rsi_overbought'] + df['bb_upper_touch'] + df['vwap_overbought'])
        ) / 3
        
        df['primary_direction'] = np.where(df['primary_signal'] > 0.5, 1,
                                          np.where(df['primary_signal'] < -0.5, -1, 0))
        
        return df
    
    def _momentum_signal(self, df: pd.DataFrame) -> pd.DataFrame:
        """모멘텀 신호"""
        
        # 가격 모멘텀
        df['price_momentum'] = df['close'].pct_change(20)
        
        # 볼륨 모멘텀
        df['volume_momentum'] = df['volume'] / df['volume'].rolling(20).mean()
        
        # ADX 필터
        df['strong_trend'] = (df['adx'] > 25).astype(int)
        
        # 종합 신호
        df['primary_signal'] = np.where(
            (df['price_momentum'] > 0.02) & (df['volume_momentum'] > 1.2) & df['strong_trend'],
            1,
            np.where(
                (df['price_momentum'] < -0.02) & (df['volume_momentum'] > 1.2) & df['strong_trend'],
                -1,
                0
            )
        )
        
        df['primary_direction'] = df['primary_signal']
        
        return df
    
    def _breakout_signal(self, df: pd.DataFrame) -> pd.DataFrame:
        """브레이크아웃 신호"""
        
        # 변동성 축소
        df['bb_squeeze'] = df['bb_width'] / df['bb_width'].rolling(50).mean()
        df['low_volatility'] = (df['bb_squeeze'] < 0.5).astype(int)
        
        # 볼륨 스파이크
        df['volume_spike'] = (df['volume'] > df['volume'].rolling(20).mean() * 2).astype(int)
        
        # 가격 브레이크아웃
        df['price_breakout_up'] = (df['close'] > df['high'].rolling(20).max().shift(1)).astype(int)
        df['price_breakout_down'] = (df['close'] < df['low'].rolling(20).min().shift(1)).astype(int)
        
        # 종합 신호
        df['primary_signal'] = np.where(
            df['low_volatility'].shift(5) & df['volume_spike'] & df['price_breakout_up'],
            1,
            np.where(
                df['low_volatility'].shift(5) & df['volume_spike'] & df['price_breakout_down'],
                -1,
                0
            )
        )
        
        df['primary_direction'] = df['primary_signal']
        
        return df
    
    def _generate_meta_labels_for_symbol(self, df: pd.DataFrame) -> pd.DataFrame:
        """심볼별 메타 라벨 생성"""
        
        # 1차 신호가 있는 경우만
        signal_mask = df['primary_direction'] != 0
        
        # 메타 라벨 초기화
        df['meta_label'] = 0
        df['meta_prob'] = 0.5
        
        # 각 1차 신호에 대해
        signal_indices = df[signal_mask].index
        
        for idx in signal_indices:
            # 컨텍스트 피처 추출
            context_features = self._extract_context_features(df, idx)
            
            # 메타 라벨 결정
            meta_decision = self._decide_meta_label(
                df.loc[idx],
                context_features
            )
            
            df.loc[idx, 'meta_label'] = meta_decision['label']
            df.loc[idx, 'meta_prob'] = meta_decision['probability']
        
        return df
    
    def _extract_context_features(self, 
                                 df: pd.DataFrame,
                                 idx: pd.Timestamp) -> Dict:
        """컨텍스트 피처 추출"""
        
        # 인덱스 위치
        loc = df.index.get_loc(idx)
        lookback = self.config['lookback_bars']
        
        if loc < lookback:
            lookback = loc
        
        # 룩백 윈도우
        window = df.iloc[loc-lookback:loc]
        
        context = {
            # 최근 성과
            'recent_win_rate': (window['label'] == 1).mean() if 'label' in window.columns else 0.5,
            'recent_avg_pnl': window['pnl_pct'].mean() if 'pnl_pct' in window.columns else 0,
            
            # 시장 상태
            'current_volatility': df.loc[idx, 'realized_vol'] if 'realized_vol' in df.columns else 0,
            'volatility_percentile': (df.loc[:idx, 'realized_vol'].rank(pct=True).iloc[-1] 
                                     if 'realized_vol' in df.columns else 0.5),
            
            # 마켓 마이크로구조
            'spread': df.loc[idx, 'spread_bps'] if 'spread_bps' in df.columns else 1,
            'depth': df.loc[idx, 'depth_total'] if 'depth_total' in df.columns else 10000,
            'ofi_z': df.loc[idx, 'ofi_z'] if 'ofi_z' in df.columns else 0,
            
            # 트렌드
            'trend_strength': df.loc[idx, 'adx'] if 'adx' in df.columns else 20,
            'trend_consistency': (window['primary_direction'] == df.loc[idx, 'primary_direction']).mean()
                                if 'primary_direction' in window.columns else 0.5,
            
            # 시간
            'hour': df.loc[idx, 'hour_utc'] if 'hour_utc' in df.columns else 12,
            'is_major_session': (df.loc[idx, 'is_american_session'] or df.loc[idx, 'is_european_session'])
                               if 'is_american_session' in df.columns else True,
            
            # 펀딩
            'funding_bias': df.loc[idx, 'funding_bias'] if 'funding_bias' in df.columns else 0,
            
            # 레짐
            'vol_regime': df.loc[idx, 'vol_regime_encoded'] if 'vol_regime_encoded' in df.columns else 1,
            'trend_regime': df.loc[idx, 'trend_regime_encoded'] if 'trend_regime_encoded' in df.columns else 2
        }
        
        return context
    
    def _decide_meta_label(self,
                          row: pd.Series,
                          context: Dict) -> Dict:
        """메타 라벨 결정 로직"""
        
        # 기본 점수
        score = 0.5
        
        # 1. 최근 성과 기반
        if context['recent_win_rate'] > 0.6:
            score += 0.1
        elif context['recent_win_rate'] < 0.4:
            score -= 0.1
        
        if context['recent_avg_pnl'] > 0.002:  # 0.2%
            score += 0.1
        elif context['recent_avg_pnl'] < -0.002:
            score -= 0.1
        
        # 2. 시장 상태 필터
        # 고변동성에서는 신중하게
        if context['volatility_percentile'] > 0.8:
            score -= 0.15
            
            # 단, 트렌드가 강하면 보정
            if context['trend_strength'] > 30:
                score += 0.1
        
        # 저변동성에서는 평균회귀 유리
        elif context['volatility_percentile'] < 0.2:
            if self.config['primary_model'] == 'mean_reversion':
                score += 0.1
            else:
                score -= 0.05
        
        # 3. 마켓 마이크로구조
        # 스프레드가 넓으면 불리
        if context['spread'] > 2:  # 2 bps
            score -= 0.1
        
        # 심도가 얕으면 불리
        if context['depth'] < 5000:
            score -= 0.1
        
        # OFI 방향 일치
        if row['primary_direction'] == 1 and context['ofi_z'] > 0.5:
            score += 0.1
        elif row['primary_direction'] == -1 and context['ofi_z'] < -0.5:
            score += 0.1
        elif abs(context['ofi_z']) > 1 and np.sign(context['ofi_z']) != row['primary_direction']:
            score -= 0.15  # OFI 역방향이면 위험
        
        # 4. 시간 필터
        # 주요 세션이 아니면 신중
        if not context['is_major_session']:
            score -= 0.05
        
        # 5. 펀딩 고려
        if context['funding_bias'] != 0:
            # 펀딩이 편향되어 있으면
            if row['primary_direction'] == -context['funding_bias']:
                score += 0.05  # 펀딩 반대 방향 유리
            else:
                score -= 0.05
        
        # 6. 트렌드 일관성
        if context['trend_consistency'] > 0.7:
            score += 0.1
        elif context['trend_consistency'] < 0.3:
            score -= 0.1
        
        # 7. 레짐별 조정
        # 극단 변동성 레짐
        if context['vol_regime'] == 3:  # extreme
            score -= 0.2
        
        # 강한 트렌드 레짐
        if context['trend_regime'] in [0, 4]:  # strong down/up
            if self.config['primary_model'] == 'trend_following':
                score += 0.15
            elif self.config['primary_model'] == 'mean_reversion':
                score -= 0.1
        
        # 최종 확률로 변환 (시그모이드)
        probability = 1 / (1 + np.exp(-10 * (score - 0.5)))
        
        # 이진 라벨 결정
        label = 1 if probability >= self.config['confidence_threshold'] else 0
        
        return {
            'label': label,
            'probability': probability,
            'score': score
        }
    
    def _evaluate_meta_labels(self, df: pd.DataFrame) -> Dict:
        """메타 라벨 품질 평가"""
        
        # 1차 신호가 있는 경우만
        mask = df['primary_direction'] != 0
        eval_df = df[mask].copy()
        
        if eval_df.empty or 'label' not in eval_df.columns:
            return {
                'precision': 0,
                'recall': 0,
                'f1_score': 0,
                'filtered_ratio': 0
            }
        
        # 실제 결과 (수익성)
        y_true = (eval_df['label'] == 1).astype(int)
        
        # 메타 라벨 예측
        y_pred = eval_df['meta_label']
        
        # True Positives, False Positives, etc.
        tp = ((y_true == 1) & (y_pred == 1)).sum()
        fp = ((y_true == 0) & (y_pred == 1)).sum()
        tn = ((y_true == 0) & (y_pred == 0)).sum()
        fn = ((y_true == 1) & (y_pred == 0)).sum()
        
        # 메트릭 계산
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # 필터링 비율
        filtered_ratio = (y_pred == 0).mean()
        
        # 필터링 후 성과 개선
        before_pnl = eval_df['pnl_pct'].mean()
        after_pnl = eval_df[eval_df['meta_label'] == 1]['pnl_pct'].mean() if (eval_df['meta_label'] == 1).any() else 0
        improvement = (after_pnl - before_pnl) / abs(before_pnl) if before_pnl != 0 else 0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'filtered_ratio': filtered_ratio,
            'before_pnl': before_pnl,
            'after_pnl': after_pnl,
            'improvement': improvement
        }
    
    def _update_meta_labels(self,
                          df: pd.DataFrame,
                          symbol: str,
                          timeframe: str):
        """메타 라벨 DB 업데이트"""
        
        # 업데이트할 데이터 준비
        updates = df[['meta_label', 'meta_prob']].copy()
        updates['symbol'] = symbol
        updates.reset_index(inplace=True)
        
        # 배치 업데이트
        for _, row in updates.iterrows():
            query = f"""
                UPDATE trading.labels_{timeframe}
                SET meta_label = {row['meta_label']}
                WHERE symbol = '{symbol}'
                    AND ts = '{row['ts']}'
            """
            self.conn.execute(query)
        
        logger.debug(f"{symbol} 메타 라벨 업데이트: {len(updates)} rows")
    
    def _print_meta_label_statistics(self, timeframe: str):
        """메타 라벨 통계 출력"""
        
        stats = self.conn.execute(f"""
            SELECT 
                symbol,
                COUNT(*) as total_signals,
                SUM(meta_label) as executed_signals,
                AVG(CASE WHEN meta_label = 1 THEN pnl_pct ELSE NULL END) as avg_executed_pnl,
                AVG(CASE WHEN meta_label = 0 THEN pnl_pct ELSE NULL END) as avg_filtered_pnl,
                SUM(CASE WHEN meta_label = 1 AND label = 1 THEN 1 ELSE 0 END) as true_positives,
                SUM(CASE WHEN meta_label = 1 AND label != 1 THEN 1 ELSE 0 END) as false_positives
            FROM trading.labels_{timeframe}
            WHERE label IS NOT NULL
            GROUP BY symbol
        """).fetchall()
        
        logger.info(f"\n=== {timeframe} 메타 라벨 통계 ===")
        
        for row in stats:
            symbol = row[0]
            total = row[1]
            executed = row[2] or 0
            avg_exec_pnl = row[3] or 0
            avg_filt_pnl = row[4] or 0
            tp = row[5] or 0
            fp = row[6] or 0
            
            filter_ratio = 1 - (executed / total) if total > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            
            logger.info(f"\n{symbol}:")
            logger.info(f"  총 신호: {total:,}")
            logger.info(f"  실행 신호: {executed:,} ({executed/total*100:.1f}%)")
            logger.info(f"  필터링 비율: {filter_ratio:.1%}")
            logger.info(f"  실행 평균 PnL: {avg_exec_pnl:.4%}")
            logger.info(f"  필터링 평균 PnL: {avg_filt_pnl:.4%}")
            logger.info(f"  정밀도: {precision:.3f}")
            
            # 개선도
            if avg_filt_pnl != 0:
                improvement = (avg_exec_pnl - avg_filt_pnl) / abs(avg_filt_pnl)
                logger.info(f"  개선도: {improvement:+.1%}")

# CLI 실행용
def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='메타 라벨링')
    parser.add_argument('--bars', default='3m', help='시간프레임')
    parser.add_argument('--symbols', nargs='+', help='심볼 리스트')
    parser.add_argument('--model', default='trend_following', 
                       choices=['trend_following', 'mean_reversion', 'momentum', 'breakout'],
                       help='1차 모델 타입')
    parser.add_argument('--start', help='시작 날짜')
    parser.add_argument('--end', help='종료 날짜')
    parser.add_argument('--db', default='data/trading.db', help='DB 경로')
    
    args = parser.parse_args()
    
    meta_labeler = MetaLabeler(db_path=args.db)
    
    meta_labeler.generate_meta_labels(
        timeframe=args.bars,
        symbols=args.symbols,
        primary_model=args.model,
        start_date=args.start,
        end_date=args.end
    )

if __name__ == "__main__":
    main()