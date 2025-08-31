-- DuckDB 초기화 스크립트
-- 데이터베이스 설정 및 초기 테이블 생성

-- 성능 최적화 설정
SET memory_limit = '4GB';
SET threads = 8;
SET enable_progress_bar = true;

-- 스키마 생성
CREATE SCHEMA IF NOT EXISTS trading;

-- 원시 거래 데이터 테이블
CREATE TABLE IF NOT EXISTS trading.trades (
    symbol VARCHAR NOT NULL,
    ts TIMESTAMP NOT NULL,
    price DOUBLE NOT NULL,
    qty DOUBLE NOT NULL,
    side VARCHAR NOT NULL,
    trade_id BIGINT,
    PRIMARY KEY (symbol, ts, trade_id)
);

-- 호가 데이터 테이블
CREATE TABLE IF NOT EXISTS trading.depth (
    symbol VARCHAR NOT NULL,
    ts TIMESTAMP NOT NULL,
    best_bid DOUBLE NOT NULL,
    best_ask DOUBLE NOT NULL,
    bid_sz DOUBLE NOT NULL,
    ask_sz DOUBLE NOT NULL,
    spread DOUBLE NOT NULL,
    bid_levels JSON,  -- [{price, size}, ...]
    ask_levels JSON,  -- [{price, size}, ...]
    PRIMARY KEY (symbol, ts)
);

-- 1분 캔들 데이터
CREATE TABLE IF NOT EXISTS trading.klines_1m (
    symbol VARCHAR NOT NULL,
    open_time TIMESTAMP NOT NULL,
    open DOUBLE NOT NULL,
    high DOUBLE NOT NULL,
    low DOUBLE NOT NULL,
    close DOUBLE NOT NULL,
    volume DOUBLE NOT NULL,
    quote_volume DOUBLE NOT NULL,
    trade_count INTEGER,
    taker_buy_volume DOUBLE,
    taker_buy_quote_volume DOUBLE,
    PRIMARY KEY (symbol, open_time)
);

-- 펀딩 레이트 데이터
CREATE TABLE IF NOT EXISTS trading.funding (
    symbol VARCHAR NOT NULL,
    ts TIMESTAMP NOT NULL,
    funding_rate DOUBLE NOT NULL,
    mark_price DOUBLE,
    index_price DOUBLE,
    PRIMARY KEY (symbol, ts)
);

-- 3분 바 데이터
CREATE TABLE IF NOT EXISTS trading.bars_3m (
    symbol VARCHAR NOT NULL,
    open_time TIMESTAMP NOT NULL,
    open DOUBLE NOT NULL,
    high DOUBLE NOT NULL,
    low DOUBLE NOT NULL,
    close DOUBLE NOT NULL,
    volume DOUBLE NOT NULL,
    quote_volume DOUBLE NOT NULL,
    trade_count INTEGER,
    vwap DOUBLE,
    PRIMARY KEY (symbol, open_time)
);

-- 5분 바 데이터
CREATE TABLE IF NOT EXISTS trading.bars_5m (
    symbol VARCHAR NOT NULL,
    open_time TIMESTAMP NOT NULL,
    open DOUBLE NOT NULL,
    high DOUBLE NOT NULL,
    low DOUBLE NOT NULL,
    close DOUBLE NOT NULL,
    volume DOUBLE NOT NULL,
    quote_volume DOUBLE NOT NULL,
    trade_count INTEGER,
    vwap DOUBLE,
    PRIMARY KEY (symbol, open_time)
);

-- 피처 테이블 (3분)
CREATE TABLE IF NOT EXISTS trading.features_3m (
    symbol VARCHAR NOT NULL,
    ts TIMESTAMP NOT NULL,
    -- 방향성 피처
    ret_60 DOUBLE,
    ret_120 DOUBLE,
    ema_fast DOUBLE,
    ema_slow DOUBLE,
    ema_slope_fast DOUBLE,
    ema_slope_slow DOUBLE,
    donchian_upper DOUBLE,
    donchian_lower DOUBLE,
    donchian_signal INTEGER,
    adx DOUBLE,
    hurst_exp DOUBLE,
    
    -- 엔트리 피처
    rsi_2 DOUBLE,
    rsi_3 DOUBLE,
    vwap DOUBLE,
    vwap_z DOUBLE,
    bb_upper DOUBLE,
    bb_lower DOUBLE,
    bb_width DOUBLE,
    
    -- 마켓 마이크로구조
    ofi DOUBLE,
    ofi_z DOUBLE,
    queue_imbalance DOUBLE,
    spread_bps DOUBLE,
    depth_total DOUBLE,
    trade_intensity DOUBLE,
    liquidity_score DOUBLE,
    
    -- 리스크/변동성
    atr DOUBLE,
    parkinson_vol DOUBLE,
    realized_vol DOUBLE,
    vol_cluster DOUBLE,
    
    -- 펀딩/캐리
    funding_rate DOUBLE,
    funding_ma_8h DOUBLE,
    funding_ma_24h DOUBLE,
    funding_std DOUBLE,
    
    -- 시간 피처
    hour_utc INTEGER,
    day_of_week INTEGER,
    is_asian_session BOOLEAN,
    is_european_session BOOLEAN,
    is_american_session BOOLEAN,
    
    PRIMARY KEY (symbol, ts)
);

-- 피처 테이블 (5분)
CREATE TABLE IF NOT EXISTS trading.features_5m 
AS SELECT * FROM trading.features_3m WHERE 1=0;

-- 라벨 테이블 (3분)
CREATE TABLE IF NOT EXISTS trading.labels_3m (
    symbol VARCHAR NOT NULL,
    ts TIMESTAMP NOT NULL,
    label INTEGER,  -- -1, 0, 1
    tp_hit BOOLEAN,
    sl_hit BOOLEAN,
    time_exit BOOLEAN,
    exit_price DOUBLE,
    exit_time TIMESTAMP,
    pnl DOUBLE,
    pnl_pct DOUBLE,
    meta_label INTEGER,  -- 0, 1
    PRIMARY KEY (symbol, ts)
);

-- 라벨 테이블 (5분)
CREATE TABLE IF NOT EXISTS trading.labels_5m
AS SELECT * FROM trading.labels_3m WHERE 1=0;

-- 마켓 레짐 테이블 (v2.0)
CREATE TABLE IF NOT EXISTS trading.market_regimes (
    symbol VARCHAR NOT NULL,
    ts TIMESTAMP NOT NULL,
    vol_state VARCHAR,  -- 'low', 'normal', 'high', 'extreme'
    trend_state VARCHAR,  -- 'strong_up', 'weak_up', 'neutral', 'weak_down', 'strong_down'
    liquidity_state VARCHAR,  -- 'deep', 'normal', 'thin'
    params_json JSON,  -- 동적 파라미터 조정값
    PRIMARY KEY (symbol, ts)
);

-- 거래 성과 테이블 (v2.0)
CREATE TABLE IF NOT EXISTS trading.trade_performance (
    trade_id VARCHAR NOT NULL,
    symbol VARCHAR NOT NULL,
    entry_ts TIMESTAMP NOT NULL,
    exit_ts TIMESTAMP,
    side VARCHAR NOT NULL,
    entry_price DOUBLE NOT NULL,
    exit_price DOUBLE,
    quantity DOUBLE NOT NULL,
    pnl DOUBLE,
    pnl_pct DOUBLE,
    fees DOUBLE,
    slippage DOUBLE,
    meta_label_prob DOUBLE,
    actual_outcome INTEGER,
    regime_at_entry VARCHAR,
    strategy_type VARCHAR,
    PRIMARY KEY (trade_id)
);

-- 인덱스 생성
CREATE INDEX IF NOT EXISTS idx_trades_symbol_ts ON trading.trades(symbol, ts);
CREATE INDEX IF NOT EXISTS idx_depth_symbol_ts ON trading.depth(symbol, ts);
CREATE INDEX IF NOT EXISTS idx_klines_1m_symbol_time ON trading.klines_1m(symbol, open_time);
CREATE INDEX IF NOT EXISTS idx_funding_symbol_ts ON trading.funding(symbol, ts);
CREATE INDEX IF NOT EXISTS idx_bars_3m_symbol_time ON trading.bars_3m(symbol, open_time);
CREATE INDEX IF NOT EXISTS idx_bars_5m_symbol_time ON trading.bars_5m(symbol, open_time);
CREATE INDEX IF NOT EXISTS idx_features_3m_symbol_ts ON trading.features_3m(symbol, ts);
CREATE INDEX IF NOT EXISTS idx_features_5m_symbol_ts ON trading.features_5m(symbol, ts);
CREATE INDEX IF NOT EXISTS idx_regimes_symbol_ts ON trading.market_regimes(symbol, ts);
CREATE INDEX IF NOT EXISTS idx_performance_symbol_entry ON trading.trade_performance(symbol, entry_ts);

-- 뷰 생성
CREATE OR REPLACE VIEW trading.latest_features_3m AS
SELECT * FROM trading.features_3m
WHERE ts = (SELECT MAX(ts) FROM trading.features_3m);

CREATE OR REPLACE VIEW trading.latest_regime AS
SELECT * FROM trading.market_regimes
WHERE ts = (SELECT MAX(ts) FROM trading.market_regimes);

-- 통계 테이블
CREATE TABLE IF NOT EXISTS trading.stats (
    key VARCHAR PRIMARY KEY,
    value JSON,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 초기 통계 삽입
INSERT OR REPLACE INTO trading.stats (key, value) VALUES
('system_status', '{"status": "initialized", "version": "2.0.0"}'),
('last_update', '{"klines": null, "features": null, "labels": null}');