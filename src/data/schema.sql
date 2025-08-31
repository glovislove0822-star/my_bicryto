-- 추가 스키마 정의 및 헬퍼 함수

-- 헬퍼 함수: 시계열 리샘플링
CREATE OR REPLACE FUNCTION resample_klines(
    p_symbol VARCHAR,
    p_timeframe VARCHAR,
    p_start_time TIMESTAMP,
    p_end_time TIMESTAMP
) RETURNS TABLE (
    symbol VARCHAR,
    open_time TIMESTAMP,
    open DOUBLE,
    high DOUBLE,
    low DOUBLE,
    close DOUBLE,
    volume DOUBLE,
    quote_volume DOUBLE,
    trade_count INTEGER,
    vwap DOUBLE
) AS $$
BEGIN
    RETURN QUERY
    WITH resampled AS (
        SELECT 
            k.symbol,
            date_trunc(p_timeframe, k.open_time) as open_time,
            FIRST(k.open) as open,
            MAX(k.high) as high,
            MIN(k.low) as low,
            LAST(k.close) as close,
            SUM(k.volume) as volume,
            SUM(k.quote_volume) as quote_volume,
            SUM(k.trade_count) as trade_count,
            SUM(k.close * k.volume) / NULLIF(SUM(k.volume), 0) as vwap
        FROM trading.klines_1m k
        WHERE k.symbol = p_symbol
            AND k.open_time >= p_start_time
            AND k.open_time < p_end_time
        GROUP BY k.symbol, date_trunc(p_timeframe, k.open_time)
        ORDER BY open_time
    )
    SELECT * FROM resampled;
END;
$$ LANGUAGE plpgsql;

-- 헬퍼 함수: 롤링 통계 계산
CREATE OR REPLACE FUNCTION calculate_rolling_stats(
    p_symbol VARCHAR,
    p_window INTEGER,
    p_metric VARCHAR
) RETURNS TABLE (
    ts TIMESTAMP,
    value DOUBLE
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        open_time as ts,
        CASE 
            WHEN p_metric = 'mean' THEN 
                AVG(close) OVER (PARTITION BY symbol ORDER BY open_time ROWS BETWEEN p_window PRECEDING AND CURRENT ROW)
            WHEN p_metric = 'std' THEN
                STDDEV(close) OVER (PARTITION BY symbol ORDER BY open_time ROWS BETWEEN p_window PRECEDING AND CURRENT ROW)
            WHEN p_metric = 'max' THEN
                MAX(high) OVER (PARTITION BY symbol ORDER BY open_time ROWS BETWEEN p_window PRECEDING AND CURRENT ROW)
            WHEN p_metric = 'min' THEN
                MIN(low) OVER (PARTITION BY symbol ORDER BY open_time ROWS BETWEEN p_window PRECEDING AND CURRENT ROW)
        END as value
    FROM trading.klines_1m
    WHERE symbol = p_symbol
    ORDER BY ts;
END;
$$ LANGUAGE plpgsql;

-- 매터리얼라이즈드 뷰: 일별 통계
CREATE MATERIALIZED VIEW IF NOT EXISTS trading.daily_stats AS
SELECT 
    symbol,
    DATE(open_time) as date,
    COUNT(*) as bar_count,
    SUM(volume) as total_volume,
    SUM(quote_volume) as total_quote_volume,
    AVG(close) as avg_price,
    STDDEV(close) as price_std,
    MAX(high) as daily_high,
    MIN(low) as daily_low,
    LAST(close) - FIRST(open) as daily_change,
    (LAST(close) - FIRST(open)) / FIRST(open) * 100 as daily_change_pct
FROM trading.klines_1m
GROUP BY symbol, DATE(open_time);

-- 인덱스 추가
CREATE INDEX IF NOT EXISTS idx_daily_stats_symbol_date ON trading.daily_stats(symbol, date);

-- 트리거: 자동 타임스탬프 업데이트
CREATE OR REPLACE FUNCTION update_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- 파티셔닝 준비 (대용량 데이터용)
-- 월별 파티션 예시
CREATE TABLE IF NOT EXISTS trading.klines_1m_template (
    LIKE trading.klines_1m INCLUDING ALL
) PARTITION BY RANGE (open_time);

-- 파티션 생성 함수
CREATE OR REPLACE FUNCTION create_monthly_partition(
    table_name TEXT,
    start_date DATE
) RETURNS VOID AS $$
DECLARE
    partition_name TEXT;
    start_timestamp TIMESTAMP;
    end_timestamp TIMESTAMP;
BEGIN
    partition_name := table_name || '_' || TO_CHAR(start_date, 'YYYY_MM');
    start_timestamp := start_date;
    end_timestamp := start_date + INTERVAL '1 month';
    
    EXECUTE format('
        CREATE TABLE IF NOT EXISTS %I PARTITION OF %I
        FOR VALUES FROM (%L) TO (%L)',
        partition_name, table_name, start_timestamp, end_timestamp
    );
END;
$$ LANGUAGE plpgsql;