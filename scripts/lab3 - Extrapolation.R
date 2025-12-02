# setwd()
# install.packages(c("tidyverse","lubridate","forecast","caret","e1071","zoo","FNN","scales"))

library(tidyverse) # объединяет ggplot2, dplyr, tidyr
library(lubridate) # as_datetime
library(forecast) # arima
library(caret) # knn
library(e1071) # svm
library(zoo) # na.fill, na.spline
library(FNN)
library(scales)

set.seed(123) # для knn и других методов, завязанных на случайных данных

# --- 1. Построение временного ряда по времени партии ----

# https://www.kaggle.com/datasets/datasnaek/chess: 
# От автора датасета "This is a set of just over 20,000 games collected from a selection of users on the site Lichess.org
# I collected this data using the Lichess API, which enables collection of any given users game history".
games_data = read.csv("games.csv", header = TRUE, stringsAsFactors = FALSE, fileEncoding = "UTF-8")
View(games_data)
games_data = games_data %>%
  mutate(
    created_at = as.numeric(created_at),
    last_move_at = as.numeric(last_move_at),
    duration_s = (last_move_at - created_at) / 1000  # в eloундах
  )

rated_games_data_clean = games_data %>% # Убираем NA и отрицательные длительности
  filter(!is.na(duration_s), duration_s >= 0, rated == "TRUE") %>%
  arrange(created_at)
# View(games_data_clean)

# --- 1.5. Выбор параметров для модели ----
find_best_acf = function(x, max_lag = 20) {
  x = na.omit(x)
  # Если данных мало — возвращаем NA
  if (length(x) < 10) return(tibble(optimal_lag = NA, max_acf = NA))
  
  acf_values = acf(x, lag.max = max_lag, plot = FALSE)$acf[-1]  # убираем лаг 0
  lag_num = which.max(abs(acf_values)) # Находим лаг с макс. |cor|
  tibble(optimal_lag = lag_num, max_acf = acf_values[lag_num])
}

plot_autocorrelation = function(df, variable, max_lag = 20, ci = 0.95) {
  if (!is.numeric(df[[variable]])) {
    cat("Переменная", variable, "не является числовой\n")
    return()
  }
  
  x_clean = na.omit(df[[variable]])
  if (length(x_clean) <= 10) {
    cat("Недостаточно данных для", variable, "\n")
    return()
  }
  
  # Встроенный график ACF с доверительными интервалами
  acf(x_clean, 
      lag.max = max_lag,
      main = paste("Автокорреляция:", variable),
      xlab = "Лаг", 
      ylab = "ACF",
      ci = ci,           # уровень доверия
      ci.col = "blue",
      ci.type = "white") # тип интервалов
}

numeric_cols = select_if(rated_games_data_clean, is.numeric)
acf_summary = map_dfr(names(numeric_cols), function(col) {
  stats = find_best_acf(numeric_cols[[col]], max_lag = 20)
  tibble(variable = col, optimal_lag = stats$optimal_lag, max_acf = stats$max_acf)
}) %>%
  arrange(desc(abs(max_acf)))
acf_summary

plot_autocorrelation(rated_games_data_clean, "black_rating")

# Берём подряд 100 значений времени партии
n_total = 100
if(nrow(rated_games_data_clean) < n_total){
  stop("В файле меньше 100 игр для построения временного ряда. Нужно >=100.")
}

first_100games = rated_games_data_clean %>% slice(1:n_total) %>%
  mutate(index = 1:n())

# Быстрая статистика
cat("Выбрано", n_total, "игр. Длительность (elo) — min/mean/median/max:\n")
print(summary(first_100games$black_rating))
# View(first_100games)


# --- 2. Разбиение 80/20 ----------------------------------------------------
n_train = 80
n_test = 20
train_y = head(first_100games$black_rating, n_train)
test_y  = tail(first_100games$black_rating, n_test)

train = head(first_100games, n_train)
test  = tail(first_100games, n_test)

# --- 2a. ARIMA --------------------------------------------------------------
# Построим модель auto.arima на тренировке и спрогнозируем 20 шагов вперед
ts_train = ts(train_y)  # простая ts-структура
arima_fit = auto.arima(ts_train, seasonal = FALSE, stepwise = FALSE, approximation = FALSE, trace = FALSE)
arima_fc = forecast(arima_fit, h = n_test)
arima_pred = as.numeric(arima_fc$mean)

# RMSE ARIMA
rmse_arima = sqrt(mean((test_y - arima_pred)^2, na.rm = TRUE))
cat(sprintf("ARIMA RMSE = %.3f (elo)\n", rmse_arima))

find_best_arima = function(train_y, test_y, 
                            max_p = 3, max_d = 2, max_q = 3,
                            seasonal = FALSE) {
  best_rmse = Inf
  best_model = NULL
  best_order = NULL
  
  for (p in 0:max_p) {
    for (d in 0:max_d) {
      for (q in 0:max_q) {
        try({
          # Строим модель
          fit = arima(train_y, order = c(p, d, q))
          fc = forecast(fit, h = length(test_y))
          pred = as.numeric(fc$mean)
          
          rmse = sqrt(mean((test_y - pred)^2, na.rm = TRUE))
          cat(sprintf("ARIMA(%d,%d,%d): RMSE = %.4f\n", p, d, q, rmse))
          
          if (rmse < best_rmse) {
            best_rmse = rmse
            best_model = fit
            best_order = c(p, d, q)
          }
        }, silent = TRUE)
      }
    }
  }
  
  cat("\nЛучшая модель ARIMA: (", 
      paste(best_order, collapse = ","), 
      ") с RMSE =", round(best_rmse, 4), "\n")
  
  return(list(model = best_model, order = best_order, rmse = best_rmse))
}

result = find_best_arima(train_y, test_y, max_p = 3, max_d = 2, max_q = 3)
print(result)

arima_pred = forecast(result$model, h = length(test_y))
arima_pred = as.numeric(arima_pred$mean)    

# --- 2b) k-NN и скользящее среднее -----------------------
# k-NN
# install.packages("tsfknn")
library(tsfknn)

knn_forecast_pred = function(train_series, test_length, max_lag = 20, k_values = 3:25) {
  val_length = min(20, length(train_series) %/% 5)
  val_train = head(train_series, length(train_series) - val_length)
  val_true  = tail(train_series, val_length)
  
  rmses = sapply(k_values, function(k) {
    val_pred = knn_forecasting(val_train, h = val_length, lags = 1:max_lag, k = k, msas = "MIMO")$prediction
    sqrt(mean((val_true - val_pred)^2))
  })
  
  best_k = k_values[which.min(rmses)]
  cat("Оптимальное k:", best_k, "(RMSE валидации:", round(min(rmses),3), ")\n")
  # финальный прогноз
  pred = knn_forecasting(train_series, h = test_length, lags = 1:max_lag, k = best_k, msas = "MIMO")$prediction
  return(list(prediction = pred, k = best_k))
}

knn_result = knn_forecast_pred(train_y, n_test)
knn_pred = knn_result$prediction
rmse_knn = sqrt(mean((test_y - knn_pred)^2))

cat(sprintf("k-NN (автоподбор) RMSE = %.3f (elo)\n", rmse_knn))

# Скользящее среднее (MA): предсказание = среднее последних w значений
min_rmse_ma = Inf
min_rmse_w = 5
min_rmse_pred = numeric(n_test)
for(w in 5:80) {
  ma_pred = numeric(n_test)
  buffer = train_y
  for(i in 1:n_test){
    ma_pred[i] = mean(tail(buffer, w))
    buffer = c(buffer, ma_pred[i])
  }
  rmse_ma = sqrt(mean((test_y - ma_pred)^2))
  if(rmse_ma < min_rmse_ma) {
    min_rmse_ma = rmse_ma
    min_rmse_w = w
    min_rmse_pred = ma_pred
  }
}

cat(sprintf("Moving Average (w=%d) RMSE = %.3f (elo)\n", min_rmse_w, min_rmse_ma))
ma_pred = min_rmse_pred

# --- 3. МНК и SVM: модель рейтинга белых (white_rating) -----------------
cor_matrix = cor(select_if(rated_games_data_clean, is.numeric), use = "pairwise.complete.obs")
cor_df_unique = as.data.frame(as.table(cor_matrix)) %>%
  # убираем диагональ
  filter(as.numeric(Var1) < as.numeric(Var2)) %>%
  mutate(pair = paste(Var1, Var2, sep = "_")) %>%
  arrange(desc(abs(Freq)))
head(cor_df_unique, 10)

plot_relation = function(data, x_var, y_var) {
  ggplot(data, aes_string(x = x_var, y = y_var)) +
    geom_point(alpha = 0.6) +
    theme_minimal() +
    labs(title = paste("Зависимость", y_var, "от", x_var),
         x = x_var, y = y_var)
}

plot_relation(rated_games_data_clean, "black_rating", "white_rating")
plot_relation(rated_games_data_clean, "opening_ply", "white_rating")

# Подготовим датасет
model_df = first_100games %>%
  mutate(black_rating = as.numeric(black_rating),
         white_rating = as.numeric(white_rating)) %>%
  select(index, opening_ply, black_rating, white_rating, turns)

model_df = model_df %>% filter(!is.na(black_rating), !is.na(white_rating))
# Проверка, не сократился ли размер до < 100 после фильтрации:
if(nrow(model_df) < n_total){
  warning("После удаления NA в рейтингах длина ряда уменьшилась.")
}

train_m = model_df %>% slice(1:n_train)
test_m  = model_df %>% slice((n_train+1):(n_train+n_test))

# 3a) МНК: y = c0 + c1*x + c2*x^2, где x = black_rating
lm_quad_black = lm(white_rating ~ black_rating + I(black_rating^2), data = train_m)
pred_lm_black = predict(lm_quad_black, newdata = test_m)
rmse_lm_black = sqrt(mean((test_m$white_rating - pred_lm_black)^2))
cat(sprintf("OLS (quadratic on black_rating) RMSE = %.3f (elo)\n", rmse_lm_black))

# 3b) SVM: используем black_rating и I(black_rating^2) как регрессоры
train_m_svm = train_m %>% mutate(black_sq = black_rating^2)
test_m_svm  = test_m  %>% mutate(black_sq = black_rating^2)

svm_black = e1071::svm(white_rating ~ black_rating + black_sq, data = train_m_svm, type = "eps-regression")
pred_svm_black = predict(svm_black, test_m_svm)
rmse_svm_black = sqrt(mean((test_m$white_rating - pred_svm_black)^2))
cat(sprintf("SVM (black_rating, black^2) RMSE = %.3f (elo)\n", rmse_svm_black))

# --- 4. Усложнённые модели: добавляем opening_ply -------------------------
# 4a) OLS: y = c0 + c1*x1 + c2*x1^2 + c3*x2  (x1 = black_rating, x2 = opening_ply)'
lm_quad_two = lm(white_rating ~ black_rating + I(black_rating^2) + opening_ply, data = train_m)
pred_lm_two = predict(lm_quad_two, newdata = test_m)
rmse_lm_two = sqrt(mean((test_m$white_rating - pred_lm_two)^2))
cat(sprintf("OLS (black, black^2, opening_ply) RMSE = %.3f (elo)\n", rmse_lm_two))

# 4b) SVM: используем black_rating, black^2 и opening_ply
train_m_svm2 = train_m %>% mutate(black_sq = black_rating^2)
test_m_svm2  = test_m  %>% mutate(black_sq = black_rating^2)

svm_two = e1071::svm(white_rating ~ black_rating + black_sq + opening_ply, data = train_m_svm2, type = "eps-regression")
pred_svm_two = predict(svm_two, test_m_svm2)
rmse_svm_two = sqrt(mean((test_m$white_rating - pred_svm_two)^2))
cat(sprintf("SVM (black, black^2, opening_ply) RMSE = %.3f (elo)\n", rmse_svm_two))

# 4c) OLS: y = c0 + c1*x1 + c2*x1^2 + c3*x2 + c4*x3 (x1 = black_rating, x2 = opening_ply, x3 = turns)'
lm_quad_3 = lm(white_rating ~ black_rating + I(black_rating^2) + opening_ply + turns, data = train_m)
pred_lm_3 = predict(lm_quad_3, newdata = test_m)
rmse_lm_3 = sqrt(mean((test_m$white_rating - pred_lm_3)^2))
cat(sprintf("OLS (black, black^2, opening_ply, turns) RMSE = %.3f (elo)\n", rmse_lm_3))

# 4d) SVM: используем black_rating, black^2, opening_ply + turns
svm_3 = e1071::svm(white_rating ~ black_rating + black_sq + opening_ply + turns, data = train_m_svm2, type = "eps-regression")
pred_svm_3 = predict(svm_3, test_m_svm2)
rmse_svm_3 = sqrt(mean((test_m$white_rating - pred_svm_3)^2))
cat(sprintf("SVM (black, black^2, opening_ply, turns) RMSE = %.3f (elo)\n", rmse_svm_3))

# --- 5. Построение графиков -----------------------------------------------
# Соберём всё в один data.frame для удобства визуализации
plot_df = bind_rows(
  tibble(index = train$index, y = train$white_rating, set = "train"),
  tibble(index = test$index,  y = test$white_rating,  set = "test")
)

preds = tibble(
  index = test$index,
  arima = arima_pred,
  knn   = knn_pred,
  ma    = ma_pred,
  lm_black = pred_lm_black,
  svm_black = pred_svm_black,
  lm_two = pred_lm_two,
  svm_two = pred_svm_two,
  lm_3 = pred_lm_3,
  svm_3 = pred_svm_3
)
preds_long = preds %>%
  pivot_longer(-index, names_to = "model", values_to = "pred")

# График 1: фактические значения + ARIMA + k-NN + MA
p1 = ggplot() +
  geom_line(data = plot_df, aes(x = index, y = y, linetype = set), size = 0.8) +
  geom_point(data = plot_df %>% filter(set == "test"), aes(x = index, y = y), color = "black", size = 1.8) +
  geom_line(data = preds_long %>% filter(model %in% c("arima","knn","ma")), 
            aes(x = index, y = pred, color = model), size = 1) +
  scale_color_manual(values = c("arima" = "blue", "knn" = "green", "ma" = "purple")) +
  labs(title = "Временной ряд: факт (train/test) и прогнозы ARIMA, k-NN, MA",
       x = "Индекс (последовательность игр)", y = "Длительность, elo",
       color = "Модель", linetype = "Набор") +
  theme_bw()

# График 2: сравнение регрессионных моделей OLS / SVM
p2 = ggplot() +
  geom_line(data = plot_df, aes(x = index, y = y, linetype = set), size = 0.8) +
  geom_line(data = preds_long %>% filter(model %in% c("lm_black","svm_black","lm_two","svm_two")),
            aes(x = index, y = pred, color = model), size = 1, alpha = 0.5) +
  scale_color_manual(values = c("lm_black"="orange","svm_black"="red","lm_two"="darkgreen","svm_two"="darkblue")) +
  labs(title = "Регрессионные модели: OLS (black^2), SVM (black^2), OLS(+opening_ply), SVM(+opening_ply)",
       x = "Индекс (последовательность игр)", y = "Рейтинг, elo",
       color = "Модель", linetype = "Набор") +
  theme_bw()

print(p1)
print(p2)

library(Metrics)  # для mae, rmse
library(caret)    # для R2


# Функция для вычисления всех метрик по одной модели
get_metrics <- function(true, pred) {
  mae_val <- mae(true, pred)
  rmse_val <- rmse(true, pred)
  r2_val <- R2(pred, true)
  tibble(MAE = mae_val, RMSE = rmse_val, R2 = r2_val)
}

metrics_df <- bind_rows(
  get_metrics(test_y, arima_pred) %>%
    mutate(Model = "ARIMA"),
  get_metrics(test_y, knn_pred) %>%
    mutate(Model = "kNN"),
  get_metrics(test_y, ma_pred) %>%
    mutate(Model = "Moving Average"),
  get_metrics(test_y, pred_lm_black) %>%
    mutate(Model = "LM (black_rating)"),
  get_metrics(test_y, pred_svm_black) %>%
    mutate(Model = "SVM (black_rating)"),
  get_metrics(test_y, pred_lm_two) %>%
    mutate(Model = "LM (2 vars)"),
  get_metrics(test_y, pred_svm_two) %>%
    mutate(Model = "SVM (2 vars)"),
  get_metrics(test_y, pred_lm_3) %>%
    mutate(Model = "LM (3 vars)"),
  get_metrics(test_y, pred_svm_3) %>%
    mutate(Model = "SVM (3 vars)")
) %>%
  select(Model, MAE, RMSE, R2) %>%
  arrange(RMSE)

print(metrics_df)
