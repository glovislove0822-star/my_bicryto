"""성과 측정 메트릭"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import logging
from scipy import stats

from ..utils.logging import Logger

logger = Logger.get_logger(__name__)

class PerformanceMetrics:
    """포괄적인 성과 측정 메트릭
    
    수익성, 리스크, 효율성 등 다양한 메트릭 계산
    """
    
    def __init__(self):
        """초기화"""
        self.metrics = {}
        
    def calculate_all_metrics(self,
                            trades_df: pd.DataFrame,
                            equity_df: pd.DataFrame,
                            initial_capital: float) -> Dict:
        """모든 메트릭 계산
        
        Args:
            trades_df: 거래 데이터프레임
            equity_df: 자산 곡선 데이터프레임
            initial_capital: 초기 자본
            
        Returns:
            메트릭 딕셔너리
        """
        
        # 기본 통계
        self.metrics.update(self._calculate_basic_stats(trades_df))
        
        # 수익성 메트릭
        self.metrics.update(self._calculate_return_metrics(trades_df, equity_df, initial_capital))
        
        # 리스크 메트릭
        self.metrics.update(self._calculate_risk_metrics(equity_df, initial_capital))
        
        # 리스크 조정 메트릭
        self.metrics.update(self._calculate_risk_adjusted_metrics())
        
        # 거래 효율성 메트릭
        self.metrics.update(self._calculate_efficiency_metrics(trades_df))
        
        # 드로우다운 분석
        self.metrics.update(self._calculate_drawdown_metrics(equity_df))
        
        # 안정성 메트릭
        self.metrics.update(self._calculate_stability_metrics(equity_df))
        
        # 분포 메트릭
        self.metrics.update(self._calculate_distribution_metrics(trades_df))
        
        # 시장 메트릭
        self.metrics.update(self._calculate_market_metrics(trades_df))
        
        return self.metrics
    
    def _calculate_basic_stats(self, trades_df: pd.DataFrame) -> Dict:
        """기본 통계"""
        
        if trades_df.empty:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0
            }
        
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['net_pnl'] > 0])
        losing_trades = len(trades_df[trades_df['net_pnl'] <= 0])
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': winning_trades / total_trades if total_trades > 0 else 0,
            'avg_trades_per_day': total_trades / trades_df['entry_time'].dt.date.nunique() if not trades_df.empty else 0
        }
    
    def _calculate_return_metrics(self,
                                 trades_df: pd.DataFrame,
                                 equity_df: pd.DataFrame,
                                 initial_capital: float) -> Dict:
        """수익성 메트릭"""
        
        if trades_df.empty or equity_df.empty:
            return {
                'total_return': 0,
                'annual_return': 0,
                'total_pnl': 0,
                'avg_pnl': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'best_trade': 0,
                'worst_trade': 0
            }
        
        # 총 수익
        final_equity = equity_df['equity'].iloc[-1]
        total_return = (final_equity - initial_capital) / initial_capital
        
        # 연율화 수익
        days = (equity_df.index[-1] - equity_df.index[0]).days
        if days > 0:
            annual_return = (1 + total_return) ** (365 / days) - 1
        else:
            annual_return = 0
        
        # PnL 통계
        total_pnl = trades_df['net_pnl'].sum()
        avg_pnl = trades_df['net_pnl'].mean()
        
        # 승/패 평균
        wins = trades_df[trades_df['net_pnl'] > 0]['net_pnl']
        losses = trades_df[trades_df['net_pnl'] <= 0]['net_pnl']
        
        avg_win = wins.mean() if len(wins) > 0 else 0
        avg_loss = losses.mean() if len(losses) > 0 else 0
        
        # 최고/최악 거래
        best_trade = trades_df['net_pnl'].max()
        worst_trade = trades_df['net_pnl'].min()
        
        # 월별 수익
        trades_df['month'] = pd.to_datetime(trades_df['exit_time']).dt.to_period('M')
        monthly_returns = trades_df.groupby('month')['net_pnl'].sum() / initial_capital
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'total_pnl': total_pnl,
            'avg_pnl': avg_pnl,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': abs(wins.sum() / losses.sum()) if losses.sum() != 0 else 0,
            'best_trade': best_trade,
            'worst_trade': worst_trade,
            'avg_monthly_return': monthly_returns.mean() if len(monthly_returns) > 0 else 0,
            'best_month': monthly_returns.max() if len(monthly_returns) > 0 else 0,
            'worst_month': monthly_returns.min() if len(monthly_returns) > 0 else 0
        }
    
    def _calculate_risk_metrics(self,
                               equity_df: pd.DataFrame,
                               initial_capital: float) -> Dict:
        """리스크 메트릭"""
        
        if equity_df.empty:
            return {
                'volatility': 0,
                'downside_volatility': 0,
                'var_95': 0,
                'cvar_95': 0,
                'max_leverage': 0
            }
        
        # 일일 수익률
        daily_returns = equity_df['equity'].pct_change().dropna()
        
        if len(daily_returns) == 0:
            return {
                'volatility': 0,
                'downside_volatility': 0,
                'var_95': 0,
                'cvar_95': 0,
                'max_leverage': 0
            }
        
        # 변동성 (연율화)
        volatility = daily_returns.std() * np.sqrt(252)
        
        # 하방 변동성
        downside_returns = daily_returns[daily_returns < 0]
        downside_volatility = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        
        # Value at Risk (95%)
        var_95 = np.percentile(daily_returns, 5)
        
        # Conditional VaR
        cvar_95 = daily_returns[daily_returns <= var_95].mean() if len(daily_returns[daily_returns <= var_95]) > 0 else 0
        
        # 최대 레버리지
        max_exposure = equity_df['equity'].max()
        max_leverage = max_exposure / initial_capital if initial_capital > 0 else 0
        
        return {
            'volatility': volatility,
            'downside_volatility': downside_volatility,
            'var_95': var_95,
            'cvar_95': cvar_95,
            'max_leverage': max_leverage,
            'avg_exposure': (equity_df['equity'] - equity_df['cash']).abs().mean() / initial_capital
        }
    
    def _calculate_risk_adjusted_metrics(self) -> Dict:
        """리스크 조정 메트릭"""
        
        metrics = {}
        
        # Sharpe Ratio
        if self.metrics.get('volatility', 0) > 0:
            metrics['sharpe_ratio'] = self.metrics.get('annual_return', 0) / self.metrics['volatility']
        else:
            metrics['sharpe_ratio'] = 0
        
        # Sortino Ratio
        if self.metrics.get('downside_volatility', 0) > 0:
            metrics['sortino_ratio'] = self.metrics.get('annual_return', 0) / self.metrics['downside_volatility']
        else:
            metrics['sortino_ratio'] = 0
        
        # Calmar Ratio
        if self.metrics.get('max_drawdown', 0) < 0:
            metrics['calmar_ratio'] = self.metrics.get('annual_return', 0) / abs(self.metrics['max_drawdown'])
        else:
            metrics['calmar_ratio'] = 0
        
        # Information Ratio (벤치마크 없이 간단히)
        metrics['information_ratio'] = metrics['sharpe_ratio']  # 간단화
        
        # Omega Ratio
        threshold = 0
        if self.metrics.get('avg_loss', 0) != 0:
            metrics['omega_ratio'] = (
                self.metrics.get('avg_win', 0) * self.metrics.get('win_rate', 0) /
                abs(self.metrics.get('avg_loss', 0) * (1 - self.metrics.get('win_rate', 0)))
            )
        else:
            metrics['omega_ratio'] = 0
        
        return metrics
    
    def _calculate_efficiency_metrics(self, trades_df: pd.DataFrame) -> Dict:
        """거래 효율성 메트릭"""
        
        if trades_df.empty:
            return {
                'avg_holding_time': 0,
                'turnover': 0,
                'cost_ratio': 0,
                'breakeven_win_rate': 0
            }
        
        # 평균 보유 시간
        avg_holding_time = trades_df['holding_time'].mean()
        
        # 회전율 (연율화)
        total_volume = trades_df['size'].sum() * trades_df['entry_price'].mean()
        days = (trades_df['exit_time'].max() - trades_df['entry_time'].min()).days
        
        if days > 0:
            annual_turnover = total_volume * 365 / days
            turnover = annual_turnover / self.metrics.get('initial_capital', 100000)
        else:
            turnover = 0
        
        # 비용 비율
        total_costs = trades_df['total_costs'].sum()
        gross_pnl = trades_df['gross_pnl'].sum()
        
        cost_ratio = total_costs / abs(gross_pnl) if gross_pnl != 0 else 0
        
        # 손익분기 승률
        avg_win = abs(self.metrics.get('avg_win', 0))
        avg_loss = abs(self.metrics.get('avg_loss', 0))
        
        if avg_win + avg_loss > 0:
            breakeven_win_rate = avg_loss / (avg_win + avg_loss)
        else:
            breakeven_win_rate = 0.5
        
        # 엣지
        actual_win_rate = self.metrics.get('win_rate', 0)
        edge = actual_win_rate - breakeven_win_rate
        
        return {
            'avg_holding_time': avg_holding_time,
            'turnover': turnover,
            'cost_ratio': cost_ratio,
            'breakeven_win_rate': breakeven_win_rate,
            'edge': edge,
            'expectancy': self.metrics.get('avg_pnl', 0)
        }
    
    def _calculate_drawdown_metrics(self, equity_df: pd.DataFrame) -> Dict:
        """드로우다운 분석"""
        
        if equity_df.empty:
            return {
                'max_drawdown': 0,
                'max_drawdown_duration': 0,
                'avg_drawdown': 0,
                'recovery_factor': 0
            }
        
        # 누적 최고점
        cummax = equity_df['equity'].expanding().max()
        
        # 드로우다운
        drawdown = (equity_df['equity'] - cummax) / cummax
        
        # 최대 드로우다운
        max_drawdown = drawdown.min()
        
        # 드로우다운 기간
        drawdown_start = None
        drawdown_periods = []
        
        for i, dd in enumerate(drawdown):
            if dd < 0 and drawdown_start is None:
                drawdown_start = i
            elif dd >= 0 and drawdown_start is not None:
                drawdown_periods.append(i - drawdown_start)
                drawdown_start = None
        
        max_drawdown_duration = max(drawdown_periods) if drawdown_periods else 0
        avg_drawdown_duration = np.mean(drawdown_periods) if drawdown_periods else 0
        
        # 평균 드로우다운
        avg_drawdown = drawdown[drawdown < 0].mean() if len(drawdown[drawdown < 0]) > 0 else 0
        
        # Recovery Factor
        recovery_factor = self.metrics.get('total_return', 0) / abs(max_drawdown) if max_drawdown < 0 else 0
        
        # Ulcer Index
        ulcer_index = np.sqrt(np.mean(drawdown[drawdown < 0] ** 2)) if len(drawdown[drawdown < 0]) > 0 else 0
        
        return {
            'max_drawdown': max_drawdown,
            'max_drawdown_duration': max_drawdown_duration,
            'avg_drawdown': avg_drawdown,
            'avg_drawdown_duration': avg_drawdown_duration,
            'recovery_factor': recovery_factor,
            'ulcer_index': ulcer_index,
            'n_drawdowns': len(drawdown_periods)
        }
    
    def _calculate_stability_metrics(self, equity_df: pd.DataFrame) -> Dict:
        """안정성 메트릭"""
        
        if equity_df.empty or len(equity_df) < 2:
            return {
                'stability': 0,
                'tail_ratio': 0,
                'max_consecutive_wins': 0,
                'max_consecutive_losses': 0
            }
        
        # R-squared (안정성)
        x = np.arange(len(equity_df))
        y = equity_df['equity'].values
        
        if len(x) > 1:
            slope, intercept = np.polyfit(x, y, 1)
            y_pred = slope * x + intercept
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            
            if ss_tot > 0:
                r_squared = 1 - (ss_res / ss_tot)
            else:
                r_squared = 0
        else:
            r_squared = 0
        
        # Tail Ratio
        returns = equity_df['equity'].pct_change().dropna()
        
        if len(returns) > 0:
            percentile_95 = np.percentile(returns, 95)
            percentile_05 = np.percentile(returns, 5)
            
            if abs(percentile_05) > 0:
                tail_ratio = percentile_95 / abs(percentile_05)
            else:
                tail_ratio = 0
        else:
            tail_ratio = 0
        
        # 연속 승/패
        if 'daily_pnl' in equity_df.columns:
            pnl_signs = np.sign(equity_df['daily_pnl'])
            
            max_consecutive_wins = 0
            max_consecutive_losses = 0
            current_streak = 0
            
            for i, sign in enumerate(pnl_signs):
                if i == 0:
                    current_streak = 1
                elif sign == pnl_signs.iloc[i-1]:
                    current_streak += 1
                else:
                    if pnl_signs.iloc[i-1] > 0:
                        max_consecutive_wins = max(max_consecutive_wins, current_streak)
                    else:
                        max_consecutive_losses = max(max_consecutive_losses, current_streak)
                    current_streak = 1
        else:
            max_consecutive_wins = 0
            max_consecutive_losses = 0
        
        return {
            'stability': r_squared,
            'tail_ratio': tail_ratio,
            'max_consecutive_wins': max_consecutive_wins,
            'max_consecutive_losses': max_consecutive_losses
        }
    
    def _calculate_distribution_metrics(self, trades_df: pd.DataFrame) -> Dict:
        """분포 메트릭"""
        
        if trades_df.empty:
            return {
                'skewness': 0,
                'kurtosis': 0,
                'median_return': 0,
                'return_distribution': {}
            }
        
        returns = trades_df['return_pct']
        
        # 왜도와 첨도
        skewness = returns.skew()
        kurtosis = returns.kurtosis()
        
        # 중앙값
        median_return = returns.median()
        
        # 분포
        distribution = {
            'p10': returns.quantile(0.1),
            'p25': returns.quantile(0.25),
            'p50': median_return,
            'p75': returns.quantile(0.75),
            'p90': returns.quantile(0.9)
        }
        
        # Jarque-Bera 검정 (정규성)
        if len(returns) > 2:
            jb_stat, jb_pvalue = stats.jarque_bera(returns)
            is_normal = jb_pvalue > 0.05
        else:
            is_normal = False
        
        return {
            'skewness': skewness,
            'kurtosis': kurtosis,
            'median_return': median_return,
            'return_distribution': distribution,
            'is_normal_distribution': is_normal
        }
    
    def _calculate_market_metrics(self, trades_df: pd.DataFrame) -> Dict:
        """시장 메트릭"""
        
        if trades_df.empty:
            return {
                'long_ratio': 0,
                'avg_long_return': 0,
                'avg_short_return': 0,
                'best_symbol': None,
                'worst_symbol': None
            }
        
        # 롱/숏 비율
        long_trades = trades_df[trades_df['side'] == 'long']
        short_trades = trades_df[trades_df['side'] == 'short']
        
        long_ratio = len(long_trades) / len(trades_df) if len(trades_df) > 0 else 0
        
        # 평균 수익
        avg_long_return = long_trades['return_pct'].mean() if len(long_trades) > 0 else 0
        avg_short_return = short_trades['return_pct'].mean() if len(short_trades) > 0 else 0
        
        # 심볼별 성과
        symbol_performance = trades_df.groupby('symbol')['net_pnl'].sum().sort_values(ascending=False)
        
        best_symbol = symbol_performance.index[0] if len(symbol_performance) > 0 else None
        worst_symbol = symbol_performance.index[-1] if len(symbol_performance) > 0 else None
        
        # 시간대별 성과
        trades_df['hour'] = pd.to_datetime(trades_df['entry_time']).dt.hour
        hourly_performance = trades_df.groupby('hour')['return_pct'].mean()
        
        best_hour = hourly_performance.idxmax() if len(hourly_performance) > 0 else None
        worst_hour = hourly_performance.idxmin() if len(hourly_performance) > 0 else None
        
        return {
            'long_ratio': long_ratio,
            'avg_long_return': avg_long_return,
            'avg_short_return': avg_short_return,
            'best_symbol': best_symbol,
            'worst_symbol': worst_symbol,
            'best_hour': best_hour,
            'worst_hour': worst_hour,
            'symbol_performance': symbol_performance.to_dict() if len(symbol_performance) > 0 else {}
        }
    
    def print_summary(self):
        """메트릭 요약 출력"""
        
        print("\n" + "="*60)
        print("성과 메트릭 요약")
        print("="*60)
        
        # 수익성
        print("\n[수익성]")
        print(f"총 수익률: {self.metrics.get('total_return', 0):.2%}")
        print(f"연율화 수익률: {self.metrics.get('annual_return', 0):.2%}")
        print(f"승률: {self.metrics.get('win_rate', 0):.2%}")
        print(f"Profit Factor: {self.metrics.get('profit_factor', 0):.2f}")
        
        # 리스크
        print("\n[리스크]")
        print(f"변동성: {self.metrics.get('volatility', 0):.2%}")
        print(f"최대 드로우다운: {self.metrics.get('max_drawdown', 0):.2%}")
        print(f"VaR (95%): {self.metrics.get('var_95', 0):.2%}")
        
        # 리스크 조정
        print("\n[리스크 조정]")
        print(f"Sharpe Ratio: {self.metrics.get('sharpe_ratio', 0):.2f}")
        print(f"Sortino Ratio: {self.metrics.get('sortino_ratio', 0):.2f}")
        print(f"Calmar Ratio: {self.metrics.get('calmar_ratio', 0):.2f}")
        
        # 효율성
        print("\n[효율성]")
        print(f"평균 보유 시간: {self.metrics.get('avg_holding_time', 0):.1f}시간")
        print(f"회전율: {self.metrics.get('turnover', 0):.1f}x")
        print(f"비용 비율: {self.metrics.get('cost_ratio', 0):.2%}")
        
        # 안정성
        print("\n[안정성]")
        print(f"안정성 (R²): {self.metrics.get('stability', 0):.2f}")
        print(f"Tail Ratio: {self.metrics.get('tail_ratio', 0):.2f}")
        print(f"최대 연속 승: {self.metrics.get('max_consecutive_wins', 0)}")
        print(f"최대 연속 패: {self.metrics.get('max_consecutive_losses', 0)}")
        
        print("="*60)