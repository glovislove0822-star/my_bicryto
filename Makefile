Makefile.PHONY: setup backtest optuna live-dryrun health probe recover clean help

# 색상 정의
GREEN := \033[0;32m
NC := \033[0m # No Color

help: ## 도움말 표시
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "$(GREEN)%-20s$(NC) %s\n", $$1, $$2}'

setup: ## 환경 설정 및 초기화
	@echo "$(GREEN)환경 설정 중...$(NC)"
	pip install -r requirements.txt
	python src/data/duckdb_init.py
	@echo "$(GREEN)환경 설정 완료!$(NC)"

backtest: ## 백테스트 실행
	@echo "$(GREEN)백테스트 시작...$(NC)"
	python src/backtest/simulator.py --config config.yaml --bars 3m
	@echo "$(GREEN)백테스트 완료!$(NC)"

optuna: ## 하이퍼파라미터 최적화
	@echo "$(GREEN)Optuna 최적화 시작...$(NC)"
	python src/optuna/runner.py --trials 200 --bars 3m
	@echo "$(GREEN)최적화 완료!$(NC)"

live-dryrun: ## 드라이런 실행
	@echo "$(GREEN)드라이런 모드 시작...$(NC)"
	python src/live/live_loop.py --dry-run --bars 3m

health: ## 헬스 체크
	@echo "$(GREEN)시스템 헬스 체크...$(NC)"
	python src/ops/healthcheck.py

probe: ## 성능 측정
	@echo "$(GREEN)성능 프로브 실행...$(NC)"
	python src/ops/latency_probe.py --minutes 5

recover: ## 데이터 복구
	@echo "$(GREEN)데이터 복구 시작...$(NC)"
	python src/ops/recover.py

clean: ## 임시 파일 정리
	@echo "$(GREEN)정리 중...$(NC)"
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	@echo "$(GREEN)정리 완료!$(NC)"

test: ## 테스트 실행
	@echo "$(GREEN)테스트 실행 중...$(NC)"
	pytest tests/ -v --cov=src --cov-report=html
	@echo "$(GREEN)테스트 완료!$(NC)"