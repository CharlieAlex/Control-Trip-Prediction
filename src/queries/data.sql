DECLARE target_date_start DATE DEFAULT DATE("2025-07-28");
DECLARE target_date_end DATE DEFAULT DATE("2025-12-10");


WITH clean_sample AS (
    SELECT DISTINCT
        user_id,
        experiment_date
    FROM `taxigo-production.Heisenberg.lp_experiment_clean_sample`
    WHERE experiment_date BETWEEN target_date_start AND target_date_end
)

, target_user AS (
    SELECT DISTINCT
        user_id,
        experiment_date
    FROM `taxigo-production.Heisenberg.random_experiment_list`
    INNER JOIN clean_sample
    USING (user_id, experiment_date)
    WHERE TRUE
        AND experiment_date BETWEEN target_date_start AND target_date_end
        AND treatment = "不發"
)

, target_user_agg AS (
    SELECT
        experiment_date,
        COUNT(DISTINCT user_id) AS user_count
    FROM target_user
    GROUP BY 1
)

, trip_label AS (
    SELECT
      a.user_id,
      a.trip_id,
      a.trip_date,
    FROM `taxigo-production.Dumbo_CRM.trip_label` a
    INNER JOIN target_user b
    ON a.user_id = b.user_id
        AND a.trip_date BETWEEN b.experiment_date AND DATE_ADD(b.experiment_date, INTERVAL 6 DAY)
    WHERE TRUE
        AND trip_date BETWEEN target_date_start AND target_date_end
        AND trip_id NOT IN (SELECT trip_id FROM `trip.test_trip_id`)
)

, trip_label_agg AS (
    SELECT
        trip_date,
        COUNT(DISTINCT trip_id) AS trip_count,
    FROM trip_label
    LEFT JOIN target_user
    USING (user_id)
    GROUP BY 1
)

SELECT
    a.trip_date,
    1 AS item_id,
    a.trip_count,
    b.user_count,
    a.trip_count / b.user_count AS trip_per_user
FROM trip_label_agg a
LEFT JOIN target_user_agg b
ON a.trip_date BETWEEN b.experiment_date AND DATE_ADD(b.experiment_date, INTERVAL 6 DAY)
ORDER BY 1
