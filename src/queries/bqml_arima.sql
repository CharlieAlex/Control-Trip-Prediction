-- 建立模型: ARIMA
CREATE OR REPLACE MODEL `taxigo-production.Heisenberg.test_pred_control_trips_model`
OPTIONS(
  model_type = 'ARIMA_PLUS',
  time_series_timestamp_col = 'timestamp', -- 時間戳記欄位
  time_series_data_col = 'trip_per_user',           -- 要預測的數值欄位
  auto_arima = TRUE,                         -- 讓 BQML 自動幫你選最強參數
  data_frequency = 'DAILY',                  -- 數據頻率（天/小時/月等）
  holiday_region = 'TW'                      -- 自動加入台灣節慶效應
) AS
SELECT
  `timestamp`, trip_per_user
FROM `taxigo-production.Heisenberg.test_pred_control_trips`
WHERE PERCENT_RANK() OVER (ORDER BY timestamp) <= 0.8
;


-- 評估模型
SELECT * FROM ML.EVALUATE(MODEL `taxigo-production.Heisenberg.test_pred_control_trips_model`);


-- 預測模型
SELECT
  *
FROM ML.FORECAST(
    MODEL `taxigo-production.Heisenberg.test_pred_control_trips_model`,
    STRUCT(30 AS horizon, 0.9 AS confidence_level)
);


-- 解釋模型
SELECT
  *
FROM ML.EXPLAIN_FORECAST(
    MODEL `taxigo-production.Heisenberg.test_pred_control_trips_model`,
    STRUCT(30 AS horizon, 0.9 AS confidence_level)
);


-- 檢查模型
SELECT
  *
FROM ML.FEATURE_INFO(
  MODEL `taxigo-production.Heisenberg.test_pred_control_trips_model`
);


-- cf. 使用 foundation model
SELECT * FROM AI.FORECAST(
  TABLE `taxigo-production.Heisenberg.test_pred_control_trips`,
  -- 指定參數
  timestamp_col => 'timestamp',
  data_col => 'trip_per_user',
  horizon => 10,                         -- 預測未來 10 天
  model => 'TimesFM 2.5'                 -- 使用最新的 TimesFM 2.5 版本
);