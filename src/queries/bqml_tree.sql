CREATE OR REPLACE VIEW `taxigo-production.Heisenberg.test_pred_control_trips_feature` AS
WITH base AS (
  SELECT
    timestamp,
    trip_per_user,
    EXTRACT(DAYOFWEEK FROM timestamp) AS day_of_week,
    LAG(trip_per_user, 1) OVER (ORDER BY timestamp) AS lag_1,
    LAG(trip_per_user, 7) OVER (ORDER BY timestamp) AS lag_7,
    AVG(trip_per_user) OVER (ORDER BY timestamp ROWS BETWEEN 7 PRECEDING AND 1 PRECEDING) AS rolling_mean_7,
    STDDEV(trip_per_user) OVER (ORDER BY timestamp ROWS BETWEEN 7 PRECEDING AND 1 PRECEDING) AS rolling_std_7
  FROM `taxigo-production.Heisenberg.test_pred_control_trips`
)
SELECT * FROM base WHERE lag_7 IS NOT NULL;


-- WARNING: 跑不動!!!
CREATE OR REPLACE MODEL `taxigo-production.Heisenberg.test_pred_control_trips_model_v2`
TRANSFORM(
  -- 只有這裡的邏輯會在預測時被自動重複使用
  ML.STANDARD_SCALER(lag_1) OVER() AS scaled_lag_1,
  ML.STANDARD_SCALER(lag_7) OVER() AS scaled_lag_7,
  ML.STANDARD_SCALER(rolling_mean_7) OVER() AS scaled_rm_7,
  ML.STANDARD_SCALER(rolling_std_7) OVER() AS scaled_rs_7,
  SIN(2 * 3.14159 * day_of_week / 7) AS sin_dow,
  COS(2 * 3.14159 * day_of_week / 7) AS cos_dow,
  `timestamp`, -- 時間
  trip_per_user -- 標籤
)
OPTIONS(
  model_type = 'RANDOM_FOREST_REGRESSOR',
  input_label_cols = ['trip_per_user'],
  -- 隨機森林專屬參數
  NUM_TREES = 60,            -- 小數據量 60 棵樹足矣
  MAX_DEPTH = 6,             -- 限制深度防止過擬合
  SUBSAMPLE = 0.8,           -- 每次採樣 80% 資料，增加隨機性
  MIN_REL_PROGRESS = 0.01,   -- 類似於節點分裂的閾值
  -- 時間序列切分
  DATA_SPLIT_METHOD = 'SEQ', -- 時間序列必須用順序切分
  DATA_SPLIT_COL = 'timestamp',
  DATA_SPLIT_EVAL_FRACTION = 0.2,
  -- 加速啟動 (不解釋模型以節省時間)
  ENABLE_GLOBAL_EXPLAIN = FALSE
) AS
SELECT * FROM `taxigo-production.Heisenberg.test_pred_control_trips_feature`;
