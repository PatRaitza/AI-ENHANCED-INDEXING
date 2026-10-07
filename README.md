install.packages("quadprog")

# ============================================================
# AI-ENHANCED INDEXING — VERSION 2
# Portfolio Optimization + Tracking Error Budgets
#
# Research question:
# Can machine-learning signals improve an equity index while
# controlling active risk relative to the benchmark?
#
# IMPORTANT LIMITATION:
# Current S&P 500 constituents are used historically.
# Therefore, the universe contains survivorship bias.
# ============================================================


# ============================================================
# 0. PACKAGES
# ============================================================
set.seed(123)
packages <- c(
  "tidyverse",
  "tidyquant",
  "rvest",
  "lubridate",
  "slider",
  "zoo",
  "xgboost",
  "PerformanceAnalytics",
  "quadprog"
)

new_packages <- packages[
  !(packages %in% installed.packages()[, "Package"])
]

if (length(new_packages) > 0) {
  install.packages(new_packages)
}

library(tidyverse)
library(tidyquant)
library(rvest)
library(lubridate)
library(slider)
library(zoo)
library(xgboost)
library(PerformanceAnalytics)
library(quadprog)

set.seed(123)


# ============================================================
# 1. SETTINGS
# ============================================================

START_DATE <- "2010-01-01"
END_DATE   <- "2025-12-31"

FIRST_TEST_YEAR <- 2019

TRANSACTION_COST <- 0.001

# Tracking error targets
TE_TARGETS <- c(
  0.005,
  0.010,
  0.020
)

# Maximum absolute active weight per security
MAX_ACTIVE_WEIGHT <- 0.005

# Covariance lookback
COV_LOOKBACK_MONTHS <- 36

# Minimum history required
MIN_COV_OBSERVATIONS <- 24


# ============================================================
# 2. S&P 500 CONSTITUENTS
# ============================================================

sp500_url <-
  "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

sp500_tables <- read_html(sp500_url) |>
  html_table(fill = TRUE)

sp500_table <- sp500_tables[[1]]

tickers <- sp500_table |>
  transmute(
    symbol = Symbol,
    company = Security,
    sector = `GICS Sector`
  ) |>
  mutate(
    symbol = str_replace_all(
      symbol,
      "\\.",
      "-"
    )
  )

cat(
  "Number of securities:",
  nrow(tickers),
  "\n"
)


# ============================================================
# 3. DOWNLOAD PRICES
# ============================================================

stock_prices <- tq_get(
  tickers$symbol,
  from = START_DATE,
  to = END_DATE,
  get = "stock.prices"
)


sp500_index <- tq_get(
  "^GSPC",
  from = START_DATE,
  to = END_DATE,
  get = "stock.prices"
)


# ============================================================
# 4. DAILY RETURNS
# ============================================================

stock_daily <- stock_prices |>
  arrange(symbol, date) |>
  group_by(symbol) |>
  mutate(
    daily_return =
      adjusted / lag(adjusted) - 1
  ) |>
  ungroup()


# ============================================================
# 5. MONTH-END PRICES
# ============================================================

monthly_prices <- stock_prices |>
  mutate(
    month = floor_date(
      date,
      "month"
    )
  ) |>
  group_by(symbol, month) |>
  slice_max(
    date,
    n = 1,
    with_ties = FALSE
  ) |>
  ungroup() |>
  select(
    symbol,
    month,
    adjusted,
    volume
  )


# ============================================================
# 6. MONTHLY STOCK RETURNS
# ============================================================

monthly_prices <- monthly_prices |>
  arrange(symbol, month) |>
  group_by(symbol) |>
  mutate(
    monthly_return =
      adjusted / lag(adjusted) - 1
  ) |>
  ungroup()


# ============================================================
# 7. MONTHLY MARKET RETURNS
# ============================================================

market_monthly <- sp500_index |>
  mutate(
    month = floor_date(
      date,
      "month"
    )
  ) |>
  group_by(month) |>
  slice_max(
    date,
    n = 1,
    with_ties = FALSE
  ) |>
  ungroup() |>
  arrange(month) |>
  mutate(
    market_return =
      adjusted / lag(adjusted) - 1
  ) |>
  select(
    month,
    market_return
  )


# ============================================================
# 8. FEATURES — MOMENTUM
# ============================================================

features <- monthly_prices |>
  arrange(symbol, month) |>
  group_by(symbol) |>
  mutate(
    
    momentum_3m =
      lag(adjusted, 1) /
      lag(adjusted, 4) - 1,
    
    momentum_6m =
      lag(adjusted, 1) /
      lag(adjusted, 7) - 1,
    
    momentum_12m =
      lag(adjusted, 1) /
      lag(adjusted, 12) - 1
    
  ) |>
  ungroup()


# ============================================================
# 9. 12-MONTH VOLATILITY
# ============================================================

features <- features |>
  arrange(symbol, month) |>
  group_by(symbol) |>
  mutate(
    
    volatility_12m =
      slide_dbl(
        monthly_return,
        ~ sd(
          .x,
          na.rm = TRUE
        ) * sqrt(12),
        .before = 11,
        .complete = TRUE
      )
    
  ) |>
  ungroup()


# ============================================================
# 10. 52-WEEK HIGH
# ============================================================

features <- features |>
  arrange(symbol, month) |>
  group_by(symbol) |>
  mutate(
    
    high_12m =
      slide_dbl(
        adjusted,
        ~ max(
          .x,
          na.rm = TRUE
        ),
        .before = 11,
        .complete = TRUE
      ),
    
    distance_52w_high =
      adjusted / high_12m
    
  ) |>
  ungroup()


# ============================================================
# 11. MOVING-AVERAGE SIGNAL
# ============================================================

features <- features |>
  arrange(symbol, month) |>
  group_by(symbol) |>
  mutate(
    
    moving_average_10m =
      slide_dbl(
        adjusted,
        ~ mean(
          .x,
          na.rm = TRUE
        ),
        .before = 9,
        .complete = TRUE
      ),
    
    ma_signal =
      adjusted /
      moving_average_10m
    
  ) |>
  ungroup()


# ============================================================
# 12. MARKET DATA
# ============================================================

features <- features |>
  left_join(
    market_monthly,
    by = "month"
  )


# ============================================================
# 13. BETA
# ============================================================

rolling_beta <- function(
    stock_returns,
    market_returns
) {
  
  valid <-
    complete.cases(
      stock_returns,
      market_returns
    )
  
  if (
    sum(valid) < 18
  ) {
    return(NA_real_)
  }
  
  cov(
    stock_returns[valid],
    market_returns[valid]
  ) /
    var(
      market_returns[valid]
    )
}


features <- features |>
  arrange(symbol, month) |>
  group_by(symbol) |>
  mutate(
    
    beta =
      slide2_dbl(
        monthly_return,
        market_return,
        ~ rolling_beta(
          .x,
          .y
        ),
        .before = 23,
        .complete = TRUE
      )
    
  ) |>
  ungroup()


# ============================================================
# 14. TARGET
# ============================================================

features <- features |>
  arrange(symbol, month) |>
  group_by(symbol) |>
  mutate(
    
    future_stock_return =
      lead(
        monthly_return,
        1
      ),
    
    future_market_return =
      lead(
        market_return,
        1
      ),
    
    future_excess_return =
      future_stock_return -
      future_market_return
    
  ) |>
  ungroup()


# ============================================================
# 15. MODEL DATA
# ============================================================

feature_names <- c(
  "momentum_3m",
  "momentum_6m",
  "momentum_12m",
  "volatility_12m",
  "beta",
  "distance_52w_high",
  "ma_signal"
)


model_data <- features |>
  select(
    symbol,
    month,
    all_of(feature_names),
    monthly_return,
    future_stock_return,
    future_market_return,
    future_excess_return
  ) |>
  filter(
    complete.cases(
      across(
        all_of(
          c(
            feature_names,
            "future_excess_return"
          )
        )
      )
    )
  )


# ============================================================
# 16. CROSS-SECTIONAL WINSORIZATION
# ============================================================

winsorize_cs <- function(x) {
  
  if (
    sum(!is.na(x)) < 10
  ) {
    return(x)
  }
  
  limits <- quantile(
    x,
    probs = c(
      0.01,
      0.99
    ),
    na.rm = TRUE
  )
  
  pmin(
    pmax(
      x,
      limits[1]
    ),
    limits[2]
  )
}


model_data <- model_data |>
  group_by(month) |>
  mutate(
    across(
      all_of(feature_names),
      winsorize_cs
    )
  ) |>
  ungroup()


# ============================================================
# 17. CROSS-SECTIONAL STANDARDIZATION
# ============================================================

safe_zscore <- function(x) {
  
  s <- sd(
    x,
    na.rm = TRUE
  )
  
  if (
    is.na(s) ||
    s == 0
  ) {
    return(
      rep(
        0,
        length(x)
      )
    )
  }
  
  (
    x -
      mean(
        x,
        na.rm = TRUE
      )
  ) / s
}


model_data <- model_data |>
  group_by(month) |>
  mutate(
    across(
      all_of(feature_names),
      safe_zscore
    )
  ) |>
  ungroup()


# ============================================================
# 18. TRADITIONAL FACTOR SCORE
# ============================================================

model_data <- model_data |>
  mutate(
    
    traditional_score =
      
      0.25 * momentum_3m +
      
      0.25 * momentum_6m +
      
      0.25 * momentum_12m +
      
      0.15 * distance_52w_high +
      
      0.10 * ma_signal -
      
      0.20 * volatility_12m
    
  )


# ============================================================
# 19. WALK-FORWARD XGBOOST
# ============================================================

test_months <- model_data |>
  filter(
    year(month) >= FIRST_TEST_YEAR
  ) |>
  distinct(month) |>
  arrange(month) |>
  pull(month)


predictions_list <- list()


for (
  current_month_raw in test_months
) {
  
  current_month <- as.Date(
    current_month_raw,
    origin = "1970-01-01"
  )
  
  cat(
    "Predicting:",
    as.character(
      current_month
    ),
    "\n"
  )
  
  
  # Important:
  # exclude the immediately preceding month because its
  # future return would not yet have been observable.
  
  train_data <- model_data |>
    filter(
      month <
        current_month %m-%
        months(1)
    )
  
  
  test_data <- model_data |>
    filter(
      month ==
        current_month
    )
  
  
  if (
    nrow(train_data) < 5000 ||
    nrow(test_data) < 100
  ) {
    next
  }
  
  
  X_train <- as.matrix(
    train_data |>
      select(
        all_of(
          feature_names
        )
      )
  )
  
  
  y_train <-
    train_data$future_excess_return
  
  
  X_test <- as.matrix(
    test_data |>
      select(
        all_of(
          feature_names
        )
      )
  )
  
  
  model <- xgboost(
    
    x = X_train,
    
    y = y_train,
    
    objective =
      "reg:squarederror",
    
    nrounds = 150,
    
    max_depth = 3,
    
    learning_rate = 0.05,
    
    subsample = 0.8,
    
    colsample_bytree = 0.8,
    
    min_child_weight = 10
    
  )
  
  
  test_data$ml_prediction <-
    predict(
      model,
      X_test
    )
  
  
  predictions_list[[
    as.character(current_month)
  ]] <- test_data
  
}


predictions <- bind_rows(
  predictions_list
)


# ============================================================
# 20. STANDARDIZE ALPHA SIGNALS
# ============================================================

predictions <- predictions |>
  group_by(month) |>
  mutate(
    
    ml_score =
      safe_zscore(
        ml_prediction
      ),
    
    traditional_score_z =
      safe_zscore(
        traditional_score
      )
    
  ) |>
  ungroup()


# ============================================================
# 21. CREATE MONTHLY RETURN MATRIX
# ============================================================

return_matrix <- monthly_prices |>
  select(
    month,
    symbol,
    monthly_return
  ) |>
  pivot_wider(
    names_from = symbol,
    values_from = monthly_return
  ) |>
  arrange(month)


# ============================================================
# 22. COVARIANCE ESTIMATION
# ============================================================

estimate_covariance <- function(
    current_month,
    symbols
) {
  
  start_month <-
    current_month %m-%
    months(
      COV_LOOKBACK_MONTHS
    )
  
  
  historical <- return_matrix |>
    filter(
      month < current_month,
      month >= start_month
    ) |>
    select(
      month,
      any_of(symbols)
    )
  
  
  X <- historical |>
    select(
      -month
    ) |>
    as.matrix()
  
  
  # Keep only requested symbols that actually exist
  valid_symbols <-
    intersect(
      symbols,
      colnames(X)
    )
  
  
  X <-
    X[
      ,
      valid_symbols,
      drop = FALSE
    ]
  
  
  # Fill missing returns using each stock's historical mean.
  # This is still a simplification.
  
  for (
    j in seq_len(
      ncol(X)
    )
  ) {
    
    column_mean <-
      mean(
        X[, j],
        na.rm = TRUE
      )
    
    if (
      is.nan(
        column_mean
      )
    ) {
      column_mean <- 0
    }
    
    X[
      is.na(
        X[, j]
      ),
      j
    ] <- column_mean
    
  }
  
  
  if (
    nrow(X) <
    MIN_COV_OBSERVATIONS
  ) {
    return(NULL)
  }
  
  
  Sigma <- cov(X)
  
  
  # Regularization to reduce numerical instability
  
  lambda <- 0.10
  
  
  diagonal_matrix <-
    diag(
      diag(
        Sigma
      )
    )
  
  
  Sigma_regularized <-
    (
      1 - lambda
    ) * Sigma +
    lambda *
    diagonal_matrix
  
  
  # Additional tiny ridge term
  
  Sigma_regularized <-
    Sigma_regularized +
    diag(
      1e-6,
      ncol(
        Sigma_regularized
      )
    )
  
  
  list(
    covariance =
      Sigma_regularized,
    
    symbols =
      valid_symbols
  )
}

# ============================================================
# 23. FAST ACTIVE PORTFOLIO OPTIMIZER
# ============================================================

optimize_active_portfolio <- function(
    alpha,
    Sigma,
    benchmark_weights,
    te_target,
    max_active_weight
) {
  
  n <- length(alpha)
  
  alpha <- as.numeric(alpha)
  benchmark_weights <- as.numeric(benchmark_weights)
  
  # Regularize covariance matrix
  Sigma <- Sigma + diag(1e-5, n)
  
  # Covariance-adjusted alpha direction
  direction <- as.numeric(
    solve(Sigma, alpha)
  )
  
  # Active weights must sum to zero
  direction <- direction - mean(direction)
  
  # Normalize direction
  if (max(abs(direction)) < 1e-12) {
    return(benchmark_weights)
  }
  
  direction <- direction / max(abs(direction))
  
  # Calculate ex-ante tracking error
  calculate_te <- function(active) {
    sqrt(
      max(0, as.numeric(
        t(active) %*% Sigma %*% active
      )) * 12
    )
  }
  
  # Find feasible scaling
  create_portfolio <- function(scale) {
    
    active <- direction * scale
    
    lower <- pmax(
      -benchmark_weights,
      -max_active_weight
    )
    
    upper <- pmin(
      1 - benchmark_weights,
      max_active_weight
    )
    
    # Enforce bounds and zero net active exposure
    for (iteration in 1:30) {
      
      active <- pmax(
        lower,
        pmin(upper, active)
      )
      
      difference <- sum(active)
      
      if (abs(difference) < 1e-10) {
        break
      }
      
      if (difference > 0) {
        capacity <- active - lower
        total_capacity <- sum(capacity)
        
        if (total_capacity > 0) {
          active <- active -
            difference * capacity / total_capacity
        }
      } else {
        capacity <- upper - active
        total_capacity <- sum(capacity)
        
        if (total_capacity > 0) {
          active <- active -
            difference * capacity / total_capacity
        }
      }
    }
    
    active <- pmax(lower, pmin(upper, active))
    
    active
  }
  
  # Binary search for tracking-error target
  lower_scale <- 0
  upper_scale <- 1
  
  for (iteration in 1:25) {
    
    mid <- (lower_scale + upper_scale) / 2
    
    active <- create_portfolio(mid)
    
    current_te <- calculate_te(active)
    
    if (current_te > te_target) {
      upper_scale <- mid
    } else {
      lower_scale <- mid
    }
  }
  
  final_active <- create_portfolio(lower_scale)
  
  final_weights <- benchmark_weights + final_active
  
  # Numerical cleanup
  final_weights <- pmax(final_weights, 0)
  final_weights <- final_weights / sum(final_weights)
  
  return(final_weights)
}
# ============================================================
# 24. FAST MONTHLY PORTFOLIO OPTIMIZATION
# ============================================================

optimized_results <- list()

months_to_optimize <- predictions |>
  distinct(month) |>
  arrange(month) |>
  pull(month)

for (current_month_raw in months_to_optimize) {
  
  current_month <- as.Date(
    current_month_raw,
    origin = "1970-01-01"
  )
  
  cat(
    "Optimizing:",
    as.character(current_month),
    "\n"
  )
  
  month_data <- predictions |>
    filter(month == current_month)
  
  covariance_result <- estimate_covariance(
    current_month,
    month_data$symbol
  )
  
  if (is.null(covariance_result)) {
    next
  }
  
  valid_symbols <- covariance_result$symbols
  
  month_data <- month_data |>
    filter(symbol %in% valid_symbols) |>
    arrange(match(symbol, valid_symbols))
  
  Sigma <- covariance_result$covariance
  
  n <- nrow(month_data)
  
  benchmark_weights <- rep(1 / n, n)
  
  for (te_target in TE_TARGETS) {
    
    ml_weights <- optimize_active_portfolio(
      alpha = month_data$ml_score,
      Sigma = Sigma,
      benchmark_weights = benchmark_weights,
      te_target = te_target,
      max_active_weight = MAX_ACTIVE_WEIGHT
    )
    
    traditional_weights <- optimize_active_portfolio(
      alpha = month_data$traditional_score_z,
      Sigma = Sigma,
      benchmark_weights = benchmark_weights,
      te_target = te_target,
      max_active_weight = MAX_ACTIVE_WEIGHT
    )
    
    temp <- month_data |>
      mutate(
        benchmark_weight = benchmark_weights,
        ml_weight = ml_weights,
        traditional_weight = traditional_weights,
        te_target = te_target
      )
    
    optimized_results[[
      paste0(current_month, "_", te_target)
    ]] <- temp
  }
}

optimized_portfolios <- bind_rows(optimized_results)

cat(
  "Optimization complete:",
  nrow(optimized_portfolios),
  "portfolio-stock observations\n"
)


# ============================================================
# 25. MONTHLY PORTFOLIO RETURNS
# ============================================================

portfolio_returns <- optimized_portfolios |>
  group_by(
    month,
    te_target
  ) |>
  summarise(
    
    benchmark_return =
      sum(
        benchmark_weight *
          future_stock_return,
        na.rm = TRUE
      ),
    
    traditional_return =
      sum(
        traditional_weight *
          future_stock_return,
        na.rm = TRUE
      ),
    
    ml_return =
      sum(
        ml_weight *
          future_stock_return,
        na.rm = TRUE
      ),
    
    .groups = "drop"
    
  )


# ============================================================
# 26. TURNOVER
# ============================================================

turnover_data <- optimized_portfolios |>
  arrange(
    te_target,
    symbol,
    month
  ) |>
  group_by(
    te_target,
    symbol
  ) |>
  mutate(
    
    previous_ml_weight =
      lag(
        ml_weight
      ),
    
    previous_traditional_weight =
      lag(
        traditional_weight
      ),
    
    ml_change =
      abs(
        ml_weight -
          coalesce(
            previous_ml_weight,
            benchmark_weight
          )
      ),
    
    traditional_change =
      abs(
        traditional_weight -
          coalesce(
            previous_traditional_weight,
            benchmark_weight
          )
      )
    
  ) |>
  ungroup() |>
  group_by(
    month,
    te_target
  ) |>
  summarise(
    
    ml_turnover =
      0.5 *
      sum(
        ml_change,
        na.rm = TRUE
      ),
    
    traditional_turnover =
      0.5 *
      sum(
        traditional_change,
        na.rm = TRUE
      ),
    
    .groups = "drop"
    
  )


portfolio_returns <- portfolio_returns |>
  left_join(
    turnover_data,
    by = c(
      "month",
      "te_target"
    )
  )


# ============================================================
# 27. TRANSACTION COSTS
# ============================================================

portfolio_returns <- portfolio_returns |>
  mutate(
    
    ml_return_net =
      ml_return -
      ml_turnover *
      TRANSACTION_COST,
    
    traditional_return_net =
      traditional_return -
      traditional_turnover *
      TRANSACTION_COST,
    
    ml_active_return =
      ml_return_net -
      benchmark_return,
    
    traditional_active_return =
      traditional_return_net -
      benchmark_return
    
  )


# ============================================================
# 28. PERFORMANCE FUNCTIONS
# ============================================================

annualized_return <- function(r) {
  
  r <- na.omit(r)
  
  prod(
    1 + r
  ) ^
    (
      12 /
        length(r)
    ) - 1
}


annualized_volatility <- function(r) {
  
  sd(
    r,
    na.rm = TRUE
  ) *
    sqrt(12)
}


tracking_error <- function(
    portfolio,
    benchmark
) {
  
  sd(
    portfolio -
      benchmark,
    na.rm = TRUE
  ) *
    sqrt(12)
}


information_ratio <- function(
    portfolio,
    benchmark
) {
  
  active <-
    portfolio -
    benchmark
  
  
  annualized_active <-
    mean(
      active,
      na.rm = TRUE
    ) *
    12
  
  
  te <-
    sd(
      active,
      na.rm = TRUE
    ) *
    sqrt(12)
  
  
  annualized_active / te
}


max_drawdown_manual <- function(r) {
  
  wealth <-
    cumprod(
      1 +
        na.omit(r)
    )
  
  
  running_max <-
    cummax(
      wealth
    )
  
  
  drawdown <-
    wealth /
    running_max - 1
  
  
  min(
    drawdown,
    na.rm = TRUE
  )
}

# ============================================================
# 29. CORRECTED PERFORMANCE SUMMARY
# ============================================================

performance_summary <- portfolio_returns |>
  group_by(te_target) |>
  summarise(
    
    benchmark_annual_return =
      annualized_return(benchmark_return),
    
    ml_annual_return =
      annualized_return(ml_return_net),
    
    traditional_annual_return =
      annualized_return(traditional_return_net),
    
    ml_active_return =
      mean(
        ml_return_net - benchmark_return,
        na.rm = TRUE
      ) * 12,
    
    traditional_active_return =
      mean(
        traditional_return_net - benchmark_return,
        na.rm = TRUE
      ) * 12,
    
    realized_ml_te =
      tracking_error(
        ml_return_net,
        benchmark_return
      ),
    
    realized_traditional_te =
      tracking_error(
        traditional_return_net,
        benchmark_return
      ),
    
    ml_information_ratio =
      information_ratio(
        ml_return_net,
        benchmark_return
      ),
    
    traditional_information_ratio =
      information_ratio(
        traditional_return_net,
        benchmark_return
      ),
    
    ml_turnover =
      mean(ml_turnover, na.rm = TRUE),
    
    traditional_turnover =
      mean(traditional_turnover, na.rm = TRUE),
    
    .groups = "drop"
  )


# Display all results

performance_summary |>
  mutate(
    across(
      c(
        te_target,
        benchmark_annual_return,
        ml_annual_return,
        traditional_annual_return,
        ml_active_return,
        traditional_active_return,
        realized_ml_te,
        realized_traditional_te,
        ml_turnover,
        traditional_turnover
      ),
      ~ round(.x * 100, 3)
    ),
    across(
      c(
        ml_information_ratio,
        traditional_information_ratio
      ),
      ~ round(.x, 3)
    )
  ) |>
  print(width = Inf)

# ============================================================
# 30. IC ANALYSIS
# ============================================================

monthly_ic <- predictions |>
  group_by(month) |>
  summarise(
    
    IC =
      cor(
        ml_prediction,
        future_excess_return,
        use =
          "complete.obs"
      ),
    
    rank_IC =
      cor(
        ml_prediction,
        future_excess_return,
        method =
          "spearman",
        use =
          "complete.obs"
      ),
    
    .groups = "drop"
    
  )


mean_ic <-
  mean(
    monthly_ic$IC,
    na.rm = TRUE
  )


sd_ic <-
  sd(
    monthly_ic$IC,
    na.rm = TRUE
  )


n_ic <-
  sum(
    !is.na(
      monthly_ic$IC
    )
  )


ic_t_stat <-
  mean_ic /
  (
    sd_ic /
      sqrt(
        n_ic
      )
  )


ic_hit_rate <-
  mean(
    monthly_ic$IC > 0,
    na.rm = TRUE
  )


cat(
  "\nMean IC:",
  round(
    mean_ic,
    4
  ),
  "\n"
)


cat(
  "IC t-statistic:",
  round(
    ic_t_stat,
    2
  ),
  "\n"
)


cat(
  "IC hit rate:",
  round(
    ic_hit_rate *
      100,
    1
  ),
  "%\n"
)


cat(
  "Mean rank IC:",
  round(
    mean(
      monthly_ic$rank_IC,
      na.rm = TRUE
    ),
    4
  ),
  "\n"
)


# ============================================================
# 31. IC PLOT
# ============================================================

ggplot(
  monthly_ic,
  aes(
    x = month,
    y = IC
  )
) +
  geom_col() +
  geom_hline(
    yintercept = 0,
    linetype = "dashed"
  ) +
  labs(
    title =
      "Out-of-Sample Information Coefficient",
    subtitle =
      "Monthly correlation between ML predictions and realized excess returns",
    x = NULL,
    y = "IC"
  ) +
  theme_minimal()


# ============================================================
# 32. CUMULATIVE PERFORMANCE
# ============================================================

cumulative_performance <- portfolio_returns |>
  group_by(
    te_target
  ) |>
  arrange(
    month,
    .by_group = TRUE
  ) |>
  mutate(
    
    benchmark_value =
      100 *
      cumprod(
        1 +
          benchmark_return
      ),
    
    ml_value =
      100 *
      cumprod(
        1 +
          ml_return_net
      ),
    
    traditional_value =
      100 *
      cumprod(
        1 +
          traditional_return_net
      )
    
  ) |>
  ungroup()


# ============================================================
# 33. PERFORMANCE PLOT — 1% TE TARGET
# ============================================================

plot_data <- cumulative_performance |>
  filter(
    te_target ==
      0.01
  ) |>
  select(
    month,
    Benchmark =
      benchmark_value,
    `Traditional Enhanced` =
      traditional_value,
    `ML Enhanced` =
      ml_value
  ) |>
  pivot_longer(
    -month,
    names_to =
      "Portfolio",
    values_to =
      "Value"
  )


ggplot(
  plot_data,
  aes(
    x = month,
    y = Value,
    color = Portfolio
  )
) +
  geom_line(
    linewidth = 1
  ) +
  labs(
    title =
      "AI-Enhanced Indexing Backtest",
    subtitle =
      "1% ex-ante tracking-error target; growth of $100",
    x = NULL,
    y =
      "Portfolio Value",
    color = NULL
  ) +
  theme_minimal()


# ============================================================
# 34. ACTIVE PERFORMANCE PLOT
# ============================================================

active_performance <- portfolio_returns |>
  group_by(
    te_target
  ) |>
  arrange(
    month,
    .by_group = TRUE
  ) |>
  mutate(
    
    cumulative_ml_active =
      cumprod(
        1 +
          ml_return_net
      ) /
      cumprod(
        1 +
          benchmark_return
      ) - 1
    
  ) |>
  ungroup()


ggplot(
  active_performance,
  aes(
    x = month,
    y =
      cumulative_ml_active,
    color =
      factor(
        te_target
      )
  )
) +
  geom_line(
    linewidth = 1
  ) +
  labs(
    title =
      "Cumulative ML Active Performance",
    subtitle =
      "Comparison across tracking-error targets",
    x = NULL,
    y =
      "Cumulative Active Performance",
    color =
      "TE Target"
  ) +
  theme_minimal()


# ============================================================
# 35. YEARLY ACTIVE RETURNS
# ============================================================

yearly_active <- portfolio_returns |>
  mutate(
    year =
      year(
        month
      )
  ) |>
  group_by(
    year,
    te_target
  ) |>
  summarise(
    
    ml_active =
      prod(
        1 +
          ml_return_net
      ) /
      prod(
        1 +
          benchmark_return
      ) - 1,
    
    traditional_active =
      prod(
        1 +
          traditional_return_net
      ) /
      prod(
        1 +
          benchmark_return
      ) - 1,
    
    .groups = "drop"
    
  )


print(
  yearly_active
)


# ============================================================
# 36. FEATURE IMPORTANCE
# ============================================================

final_train <- model_data |>
  filter(
    year(month) <
      2025
  )


X_final <- as.matrix(
  final_train |>
    select(
      all_of(
        feature_names
      )
    )
)


y_final <-
  final_train$future_excess_return


final_model <- xgboost(
  
  x = X_final,
  
  y = y_final,
  
  objective =
    "reg:squarederror",
  
  nrounds = 150,
  
  max_depth = 3,
  
  learning_rate = 0.05,
  
  subsample = 0.8,
  
  colsample_bytree = 0.8,
  
  min_child_weight = 10
  
)


importance <- xgb.importance(
  feature_names =
    feature_names,
  model =
    final_model
)


print(
  importance
)


xgb.plot.importance(
  importance,
  measure = "Gain"
)


# ============================================================
# 37. FINAL SUMMARY
# ============================================================

cat(
  "\n",
  "====================================================\n",
  "AI-ENHANCED INDEXING — VERSION 2\n",
  "====================================================\n"
)


cat(
  "Test period:",
  as.character(
    min(
      portfolio_returns$month
    )
  ),
  "to",
  as.character(
    max(
      portfolio_returns$month
    )
  ),
  "\n\n"
)


cat(
  "Mean monthly IC:",
  round(
    mean_ic,
    4
  ),
  "\n"
)


cat(
  "IC t-statistic:",
  round(
    ic_t_stat,
    2
  ),
  "\n"
)


cat(
  "IC hit rate:",
  round(
    ic_hit_rate *
      100,
    1
  ),
  "%\n"
)


cat(
  "Mean rank IC:",
  round(
    mean(
      monthly_ic$rank_IC,
      na.rm = TRUE
    ),
    4
  ),
  "\n\n"
)


print(
  performance_summary
)


cat(
  "\nIMPORTANT LIMITATION:\n"
)


cat(
  paste0(
    "Historical membership is approximated using current ",
    "S&P 500 constituents. Results therefore contain ",
    "survivorship bias and should not be interpreted as ",
    "a production investment strategy.\n"
  )
)


cat(
  "====================================================\n"
)
