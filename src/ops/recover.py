"""시스템 복구 관리"""

import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import logging
import shutil
import asyncio
import duckdb

from ..utils.logging import Logger
from ..utils.io import IOUtils

logger = Logger.get_logger(__name__)

class RecoveryManager:
    """시스템 복구 관리자
    
    장애 발생 시 시스템 상태 복구 및 데이터 정합성 보장
    """
    
    def __init__(self, config: Dict):
        """
        Args:
            config: 복구 설정
        """
        self.config = config
        
        # 복구 경로
        self.backup_dir = Path(config.get('backup_dir', 'backups'))
        self.snapshot_dir = Path(config.get('snapshot_dir', 'snapshots'))
        self.checkpoint_dir = Path(config.get('checkpoint_dir', 'checkpoints'))
        
        # 디렉토리 생성
        self.backup_dir.mkdir(exist_ok=True)
        self.snapshot_dir.mkdir(exist_ok=True)
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # 복구 정책
        self.recovery_policy = {
            'max_data_loss_minutes': config.get('max_data_loss_minutes', 5),
            'auto_recovery': config.get('auto_recovery', True),
            'verify_integrity': config.get('verify_integrity', True),
            'rollback_on_error': config.get('rollback_on_error', True)
        }
        
        # 복구 히스토리
        self.recovery_history = []
        
        # 체크포인트 관리
        self.last_checkpoint = None
        self.checkpoint_interval = config.get('checkpoint_interval_minutes', 15)
    
    async def create_checkpoint(self, components: Dict[str, Any]) -> str:
        """체크포인트 생성
        
        Args:
            components: 저장할 컴포넌트들
            
        Returns:
            체크포인트 ID
        """
        
        checkpoint_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        checkpoint_path = self.checkpoint_dir / checkpoint_id
        checkpoint_path.mkdir(exist_ok=True)
        
        logger.info(f"체크포인트 생성: {checkpoint_id}")
        
        checkpoint_data = {
            'id': checkpoint_id,
            'timestamp': datetime.now(),
            'components': {}
        }
        
        # 각 컴포넌트 저장
        for name, component in components.items():
            try:
                if name == 'state':
                    # 거래 상태
                    self._save_state(component, checkpoint_path)
                    checkpoint_data['components']['state'] = True
                    
                elif name == 'positions':
                    # 포지션
                    self._save_positions(component, checkpoint_path)
                    checkpoint_data['components']['positions'] = True
                    
                elif name == 'models':
                    # ML 모델
                    self._save_models(component, checkpoint_path)
                    checkpoint_data['components']['models'] = True
                    
                elif name == 'database':
                    # 데이터베이스
                    await self._backup_database(checkpoint_path)
                    checkpoint_data['components']['database'] = True
                    
                elif name == 'config':
                    # 설정
                    self._save_config(component, checkpoint_path)
                    checkpoint_data['components']['config'] = True
                    
            except Exception as e:
                logger.error(f"컴포넌트 저장 실패 ({name}): {e}")
                
                if self.recovery_policy['rollback_on_error']:
                    # 롤백
                    shutil.rmtree(checkpoint_path)
                    raise
        
        # 메타데이터 저장
        with open(checkpoint_path / 'metadata.json', 'w') as f:
            json.dump(checkpoint_data, f, indent=2, default=str)
        
        self.last_checkpoint = checkpoint_id
        
        # 오래된 체크포인트 정리
        self._cleanup_old_checkpoints()
        
        logger.info(f"체크포인트 완료: {checkpoint_id}")
        
        return checkpoint_id
    
    def _save_state(self, state: Any, checkpoint_path: Path):
        """상태 저장
        
        Args:
            state: TradingState 객체
            checkpoint_path: 체크포인트 경로
        """
        
        state_data = {
            'market_state': state.market_state,
            'features': state.features,
            'signals': state.signals,
            'positions': state.positions,
            'performance': state.performance,
            'risk_state': state.risk_state,
            'regime_state': state.regime_state,
            'system_state': state.system_state
        }
        
        with open(checkpoint_path / 'state.json', 'w') as f:
            json.dump(state_data, f, indent=2, default=str)
    
    def _save_positions(self, position_manager: Any, checkpoint_path: Path):
        """포지션 저장
        
        Args:
            position_manager: PositionManager 객체
            checkpoint_path: 체크포인트 경로
        """
        
        positions_data = {
            'positions': {},
            'history': [],
            'stats': position_manager.stats
        }
        
        # 현재 포지션
        for symbol, position in position_manager.positions.items():
            positions_data['positions'][symbol] = {
                'symbol': position.symbol,
                'side': position.side.value,
                'entry_price': position.entry_price,
                'size': position.size,
                'entry_time': position.entry_time.isoformat(),
                'stop_loss': position.stop_loss,
                'take_profit': position.take_profit,
                'metadata': position.metadata
            }
        
        # 히스토리 (최근 100개)
        for pos in position_manager.position_history[-100:]:
            positions_data['history'].append({
                'symbol': pos.symbol,
                'side': pos.side.value,
                'entry_price': pos.entry_price,
                'exit_price': pos.exit_price,
                'size': pos.size,
                'realized_pnl': pos.realized_pnl,
                'entry_time': pos.entry_time.isoformat() if pos.entry_time else None,
                'exit_time': pos.exit_time.isoformat() if pos.exit_time else None
            })
        
        with open(checkpoint_path / 'positions.json', 'w') as f:
            json.dump(positions_data, f, indent=2)
    
    def _save_models(self, models: Dict, checkpoint_path: Path):
        """모델 저장
        
        Args:
            models: 모델 딕셔너리
            checkpoint_path: 체크포인트 경로
        """
        
        models_path = checkpoint_path / 'models'
        models_path.mkdir(exist_ok=True)
        
        for name, model in models.items():
            if model is not None:
                model_file = models_path / f'{name}.pkl'
                
                with open(model_file, 'wb') as f:
                    pickle.dump(model, f)
    
    async def _backup_database(self, checkpoint_path: Path):
        """데이터베이스 백업
        
        Args:
            checkpoint_path: 체크포인트 경로
        """
        
        db_path = Path('data/trading.db')
        
        if not db_path.exists():
            logger.warning("데이터베이스 파일 없음")
            return
        
        # 백업 파일 경로
        backup_path = checkpoint_path / 'trading.db'
        
        # DuckDB 백업
        conn = duckdb.connect(str(db_path), read_only=True)
        
        try:
            # EXPORT 명령으로 백업
            conn.execute(f"EXPORT DATABASE '{checkpoint_path / 'db_export'}'")
            
            # 또는 파일 복사
            shutil.copy2(db_path, backup_path)
            
            logger.info(f"데이터베이스 백업 완료: {backup_path}")
            
        finally:
            conn.close()
    
    def _save_config(self, config: Dict, checkpoint_path: Path):
        """설정 저장
        
        Args:
            config: 설정 딕셔너리
            checkpoint_path: 체크포인트 경로
        """
        
        with open(checkpoint_path / 'config.yaml', 'w') as f:
            IOUtils.save_yaml(config, f)
    
    async def recover_from_checkpoint(self, 
                                     checkpoint_id: Optional[str] = None) -> Dict:
        """체크포인트에서 복구
        
        Args:
            checkpoint_id: 체크포인트 ID (None이면 최신)
            
        Returns:
            복구 결과
        """
        
        if checkpoint_id is None:
            # 최신 체크포인트 찾기
            checkpoint_id = self._find_latest_checkpoint()
            
            if checkpoint_id is None:
                logger.error("사용 가능한 체크포인트 없음")
                return {'success': False, 'error': 'No checkpoint available'}
        
        checkpoint_path = self.checkpoint_dir / checkpoint_id
        
        if not checkpoint_path.exists():
            logger.error(f"체크포인트 없음: {checkpoint_id}")
            return {'success': False, 'error': 'Checkpoint not found'}
        
        logger.info(f"체크포인트 복구 시작: {checkpoint_id}")
        
        # 메타데이터 로드
        with open(checkpoint_path / 'metadata.json') as f:
            metadata = json.load(f)
        
        recovery_result = {
            'checkpoint_id': checkpoint_id,
            'timestamp': metadata['timestamp'],
            'components': {},
            'errors': []
        }
        
        # 각 컴포넌트 복구
        if metadata['components'].get('state'):
            try:
                state_data = self._load_state(checkpoint_path)
                recovery_result['components']['state'] = state_data
                logger.info("상태 복구 완료")
            except Exception as e:
                logger.error(f"상태 복구 실패: {e}")
                recovery_result['errors'].append(f"State recovery failed: {e}")
        
        if metadata['components'].get('positions'):
            try:
                positions_data = self._load_positions(checkpoint_path)
                recovery_result['components']['positions'] = positions_data
                logger.info("포지션 복구 완료")
            except Exception as e:
                logger.error(f"포지션 복구 실패: {e}")
                recovery_result['errors'].append(f"Position recovery failed: {e}")
        
        if metadata['components'].get('models'):
            try:
                models = self._load_models(checkpoint_path)
                recovery_result['components']['models'] = models
                logger.info("모델 복구 완료")
            except Exception as e:
                logger.error(f"모델 복구 실패: {e}")
                recovery_result['errors'].append(f"Model recovery failed: {e}")
        
        if metadata['components'].get('database'):
            try:
                await self._restore_database(checkpoint_path)
                recovery_result['components']['database'] = True
                logger.info("데이터베이스 복구 완료")
            except Exception as e:
                logger.error(f"데이터베이스 복구 실패: {e}")
                recovery_result['errors'].append(f"Database recovery failed: {e}")
        
        if metadata['components'].get('config'):
            try:
                config = self._load_config(checkpoint_path)
                recovery_result['components']['config'] = config
                logger.info("설정 복구 완료")
            except Exception as e:
                logger.error(f"설정 복구 실패: {e}")
                recovery_result['errors'].append(f"Config recovery failed: {e}")
        
        # 정합성 검증
        if self.recovery_policy['verify_integrity']:
            integrity_check = await self._verify_integrity(recovery_result)
            recovery_result['integrity_check'] = integrity_check
        
        # 복구 히스토리 기록
        self.recovery_history.append({
            'timestamp': datetime.now(),
            'checkpoint_id': checkpoint_id,
            'success': len(recovery_result['errors']) == 0,
            'errors': recovery_result['errors']
        })
        
        recovery_result['success'] = len(recovery_result['errors']) == 0
        
        logger.info(f"체크포인트 복구 완료: 성공={recovery_result['success']}")
        
        return recovery_result
    
    def _load_state(self, checkpoint_path: Path) -> Dict:
        """상태 로드
        
        Args:
            checkpoint_path: 체크포인트 경로
            
        Returns:
            상태 데이터
        """
        
        with open(checkpoint_path / 'state.json') as f:
            return json.load(f)
    
    def _load_positions(self, checkpoint_path: Path) -> Dict:
        """포지션 로드
        
        Args:
            checkpoint_path: 체크포인트 경로
            
        Returns:
            포지션 데이터
        """
        
        with open(checkpoint_path / 'positions.json') as f:
            return json.load(f)
    
    def _load_models(self, checkpoint_path: Path) -> Dict:
        """모델 로드
        
        Args:
            checkpoint_path: 체크포인트 경로
            
        Returns:
            모델 딕셔너리
        """
        
        models = {}
        models_path = checkpoint_path / 'models'
        
        if models_path.exists():
            for model_file in models_path.glob('*.pkl'):
                name = model_file.stem
                
                with open(model_file, 'rb') as f:
                    models[name] = pickle.load(f)
        
        return models
    
    async def _restore_database(self, checkpoint_path: Path):
        """데이터베이스 복원
        
        Args:
            checkpoint_path: 체크포인트 경로
        """
        
        backup_path = checkpoint_path / 'trading.db'
        db_path = Path('data/trading.db')
        
        if backup_path.exists():
            # 기존 DB 백업
            if db_path.exists():
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                shutil.move(db_path, db_path.parent / f'trading_old_{timestamp}.db')
            
            # 복원
            shutil.copy2(backup_path, db_path)
            
            logger.info("데이터베이스 복원 완료")
        
        # 또는 EXPORT 복원
        export_path = checkpoint_path / 'db_export'
        if export_path.exists():
            conn = duckdb.connect(str(db_path))
            try:
                conn.execute(f"IMPORT DATABASE '{export_path}'")
            finally:
                conn.close()
    
    def _load_config(self, checkpoint_path: Path) -> Dict:
        """설정 로드
        
        Args:
            checkpoint_path: 체크포인트 경로
            
        Returns:
            설정 딕셔너리
        """
        
        config_path = checkpoint_path / 'config.yaml'
        
        if config_path.exists():
            return IOUtils.load_yaml(str(config_path))
        
        return {}
    
    async def _verify_integrity(self, recovery_result: Dict) -> Dict:
        """정합성 검증
        
        Args:
            recovery_result: 복구 결과
            
        Returns:
            검증 결과
        """
        
        integrity = {
            'passed': True,
            'checks': {}
        }
        
        # 상태-포지션 일치 검증
        if 'state' in recovery_result['components'] and 'positions' in recovery_result['components']:
            state_positions = recovery_result['components']['state'].get('positions', {})
            position_positions = recovery_result['components']['positions'].get('positions', {})
            
            if set(state_positions.keys()) != set(position_positions.keys()):
                integrity['passed'] = False
                integrity['checks']['position_mismatch'] = {
                    'state_positions': list(state_positions.keys()),
                    'position_manager_positions': list(position_positions.keys())
                }
        
        # 데이터베이스 검증
        if recovery_result['components'].get('database'):
            try:
                conn = duckdb.connect('data/trading.db', read_only=True)
                
                # 테이블 존재 확인
                tables = conn.execute("SELECT table_name FROM information_schema.tables").fetchall()
                table_names = [t[0] for t in tables]
                
                required_tables = ['trades', 'klines_1m', 'features_3m', 'features_5m']
                
                for table in required_tables:
                    if table not in table_names:
                        integrity['passed'] = False
                        integrity['checks'][f'missing_table_{table}'] = True
                
                conn.close()
                
            except Exception as e:
                integrity['passed'] = False
                integrity['checks']['database_error'] = str(e)
        
        return integrity
    
    def _find_latest_checkpoint(self) -> Optional[str]:
        """최신 체크포인트 찾기
        
        Returns:
            체크포인트 ID 또는 None
        """
        
        checkpoints = list(self.checkpoint_dir.glob('*'))
        
        if not checkpoints:
            return None
        
        # 타임스탬프 기준 정렬
        checkpoints.sort(key=lambda x: x.name, reverse=True)
        
        return checkpoints[0].name
    
    def _cleanup_old_checkpoints(self):
        """오래된 체크포인트 정리"""
        
        max_checkpoints = self.config.get('max_checkpoints', 10)
        
        checkpoints = list(self.checkpoint_dir.glob('*'))
        
        if len(checkpoints) > max_checkpoints:
            # 오래된 순으로 정렬
            checkpoints.sort(key=lambda x: x.name)
            
            # 삭제할 개수
            to_delete = len(checkpoints) - max_checkpoints
            
            for checkpoint in checkpoints[:to_delete]:
                shutil.rmtree(checkpoint)
                logger.info(f"오래된 체크포인트 삭제: {checkpoint.name}")
    
    async def disaster_recovery(self) -> Dict:
        """재해 복구
        
        Returns:
            복구 결과
        """
        
        logger.critical("재해 복구 시작")
        
        recovery_steps = []
        
        # 1. 최신 백업 찾기
        latest_checkpoint = self._find_latest_checkpoint()
        
        if latest_checkpoint:
            recovery_steps.append({
                'step': 'checkpoint_found',
                'checkpoint_id': latest_checkpoint
            })
            
            # 2. 체크포인트 복구
            recovery_result = await self.recover_from_checkpoint(latest_checkpoint)
            recovery_steps.append({
                'step': 'checkpoint_recovery',
                'result': recovery_result
            })
            
            # 3. 데이터 갭 채우기
            if recovery_result['success']:
                gap_result = await self._fill_data_gap(latest_checkpoint)
                recovery_steps.append({
                    'step': 'gap_filling',
                    'result': gap_result
                })
        
        else:
            # 백업 없음 - 초기화
            recovery_steps.append({
                'step': 'no_checkpoint',
                'action': 'initialize_from_scratch'
            })
        
        logger.critical(f"재해 복구 완료: {len(recovery_steps)} 단계")
        
        return {
            'timestamp': datetime.now(),
            'steps': recovery_steps,
            'success': all(s.get('result', {}).get('success', True) for s in recovery_steps)
        }
    
    async def _fill_data_gap(self, checkpoint_id: str) -> Dict:
        """데이터 갭 채우기
        
        Args:
            checkpoint_id: 체크포인트 ID
            
        Returns:
            갭 채우기 결과
        """
        
        # 체크포인트 시간
        checkpoint_time = datetime.strptime(checkpoint_id, '%Y%m%d_%H%M%S')
        
        # 현재 시간과의 차이
        gap_minutes = (datetime.now() - checkpoint_time).total_seconds() / 60
        
        result = {
            'gap_minutes': gap_minutes,
            'filled': False
        }
        
        if gap_minutes <= self.recovery_policy['max_data_loss_minutes']:
            # 허용 범위 내 - 데이터 수집으로 채우기
            logger.info(f"데이터 갭 채우기: {gap_minutes:.1f}분")
            
            # TODO: 실제 데이터 수집 구현
            result['filled'] = True
            result['method'] = 'data_collection'
        
        else:
            # 허용 범위 초과 - 수동 개입 필요
            logger.warning(f"데이터 갭이 너무 큼: {gap_minutes:.1f}분")
            result['method'] = 'manual_intervention_required'
        
        return result
    
    def get_recovery_status(self) -> Dict:
        """복구 상태 조회
        
        Returns:
            복구 상태
        """
        
        return {
            'last_checkpoint': self.last_checkpoint,
            'available_checkpoints': len(list(self.checkpoint_dir.glob('*'))),
            'recovery_history': self.recovery_history[-10:],  # 최근 10개
            'auto_recovery_enabled': self.recovery_policy['auto_recovery']
        }

# CLI 실행용
async def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='시스템 복구')
    parser.add_argument('--action', choices=['backup', 'restore', 'disaster'], 
                       required=True, help='복구 액션')
    parser.add_argument('--checkpoint', help='체크포인트 ID')
    parser.add_argument('--config', required=True, help='설정 파일')
    
    args = parser.parse_args()
    
    # 설정 로드
    config = IOUtils.load_config(args.config)
    
    # 복구 관리자 생성
    manager = RecoveryManager(config)
    
    if args.action == 'backup':
        # 백업 생성
        components = {
            'config': config
            # 실제로는 다른 컴포넌트도 추가
        }
        
        checkpoint_id = await manager.create_checkpoint(components)
        print(f"체크포인트 생성: {checkpoint_id}")
        
    elif args.action == 'restore':
        # 복구
        result = await manager.recover_from_checkpoint(args.checkpoint)
        print(json.dumps(result, indent=2, default=str))
        
    elif args.action == 'disaster':
        # 재해 복구
        result = await manager.disaster_recovery()
        print(json.dumps(result, indent=2, default=str))

if __name__ == "__main__":
    asyncio.run(main())