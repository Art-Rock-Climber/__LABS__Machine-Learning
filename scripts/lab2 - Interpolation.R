# setwd()

install.packages("ggplot2")
install.packages("dplyr") # для манипуляций с данными: filter, mutate, arrange, select
install.packages("tidyr") # для преобразования данных: pivot_longer

install.packages("zoo")
install.packages("https://cran.r-project.org/bin/windows/contrib/4.4/pracma_2.4.4.zip", 
                 repos = NULL, type = "binary")

library(ggplot2)
library(dplyr)
library(tidyr)

library(zoo)
library(pracma)

# https://www.kaggle.com/datasets/datasnaek/chess: 
# От автора датасета "This is a set of just over 20,000 games collected from a selection of users on the site Lichess.org
# I collected this data using the Lichess API, which enables collection of any given users game history".
games_data = read.csv("lab1 - games.csv", header =T, sep = ",", encoding = "UTF-8")
View(games_data)

# 1) Подготовка данных, прореживание
rated_QP = games_data %>%
  filter(rated == "TRUE", opening_name == "Queen's Pawn Game") %>%
  mutate(datetime = as.POSIXct(created_at / 1000, origin = "1970-01-01")) %>%
  arrange(datetime) %>%
  head(30) %>%
  mutate(
    time_index = 1:30,
    original = white_rating,
    thinned = ifelse(time_index %% 2 == 1, white_rating, NA)
  )
View(rated_QP)

# 2) Заполнение разными методами
rated_QP_filled = rated_QP %>%
  mutate(
    mean_filled = coalesce(thinned, mean(thinned, na.rm = TRUE)),
    median_filled = coalesce(thinned, median(thinned, na.rm = TRUE)),
    mean_3_previous = zoo::na.fill(
      zoo::rollmeanr(zoo::na.fill(thinned, "extend"), 3, fill = NA), 
      "extend"
    ),
    lagrange = {
      known = which(!is.na(thinned))
      missing = which(is.na(thinned))
      valid_missing = missing[missing >= min(known) & missing <= max(known)]
      if(length(known) >= 2 & length(valid_missing) > 0) {
        result = thinned
        result[valid_missing] = barylag(known, thinned[known], valid_missing)
        result
      } else thinned
    },
    spline = na.spline(thinned),
    
    # Непрерывная функция Лагранжа для всех точек
    lagrange_continuous = {
      # Используем уже заполненный ряд lagrange как основу
      if(all(is.na(lagrange))) return(rep(NA, length(thinned)))
      known_points <- which(!is.na(lagrange))
      known_values <- lagrange[known_points]
      
      if(length(known_points) >= 2) {
        continuous_points <- seq(min(known_points), max(known_points), length.out = length(thinned))
        barylag(known_points, known_values, continuous_points)
      } else {
        rep(NA, length(thinned))
      }
    },
    # Непрерывная функция сплайнов для всех точек
    spline_continuous = {
      if(all(is.na(spline))) return(rep(NA, length(thinned)))
      known_points <- which(!is.na(spline))
      known_values <- spline[known_points]
      if(length(known_points) >= 2) {
        continuous_points <- seq(min(known_points), max(known_points), length.out = length(thinned))
        splinefun(known_points, known_values)(continuous_points)
      } else {
        rep(NA, length(thinned))
      }
    }
  )
# View(rated_QP_filled)

# Функция для построения графиков отдельных методов
plot_fill = function(df, method_name, method_label, show_continuous = FALSE) {
  # Определяем, нужно ли показывать непрерывную функцию
  continuous_col <- if(show_continuous) {
    if(method_name == "lagrange") "lagrange_continuous"
    else if(method_name == "spline") "spline_continuous"
    else NULL
  } else NULL
  
  p <- ggplot(df, aes(x = time_index)) +
    # Исходный ряд
    geom_line(aes(y = original), color = "grey", size = 1.5, alpha = 0.8) +
    # Известные значения
    geom_point(aes(y = thinned), color = "forestgreen", size = 3, shape = 16) +
    # Пропущенные реальные значения
    geom_point(aes(y = ifelse(is.na(thinned), original, NA)), 
               color = "darkred", size = 2, shape = 4, stroke = 1.5) +
    # Заполненные значения
    geom_point(aes(y = ifelse(is.na(thinned), .data[[method_name]], NA)), 
               color = "blue", size = 2, shape = 1, stroke = 1.2) +
    labs(title = paste("Метод:", method_label),
         subtitle = "Зеленые точки - известные значения, Крестики - пропуски, Синие точки - заполнения",
         x = "Временной индекс", y = "Рейтинг") +
    theme_bw() +
    theme(plot.title = element_text(hjust = 0.5))
  
  # Добавляем непрерывную функцию если нужно
  if(!is.null(continuous_col)) {
    p <- p + 
      geom_line(aes(y = .data[[continuous_col]]), 
                color = "purple", size = 0.8, alpha = 0.7, linetype = "solid")
  }
  
  return(p)
}

print(plot_fill(rated_QP_filled, "mean_filled", "Заполнение средним"))
print(plot_fill(rated_QP_filled, "median_filled", "Заполнение медианой"))
print(plot_fill(rated_QP_filled, "mean_3_previous", "Среднее 3-х предыдущих"))
print(plot_fill(rated_QP_filled, "lagrange", "Лагранж", show_continuous = TRUE))
print(plot_fill(rated_QP_filled, "spline", "Кубические сплайны", show_continuous = TRUE))


# Проверка на выбросы
analyze_outliers = function(data, n_sigmas) {
  values = na.omit(data)
  
  bp_stats = boxplot.stats(values)
  bp_lower = bp_stats$stats[1]  # нижний ус
  bp_upper = bp_stats$stats[5]  # верхний ус
  
  mean_val = mean(values)
  sd_val = sd(values)
  z_lower = mean_val - n_sigmas * sd_val
  z_upper = mean_val + n_sigmas * sd_val
  
  # Выбросы
  bp_out = bp_stats$out
  z_out = values[abs(scale(values)) > n_sigmas]
  
  cat("Анализ выбросов\n")
  cat("Всего точек:", length(values), "\n")
  cat("Box-plot границы:", round(bp_lower, 1), "-", round(bp_upper, 1), "\n")
  cat("Box-plot выбросы:", length(bp_out), "- Значения:", bp_out, "\n")
  cat(n_sigmas, "сигма границы:", round(z_lower, 1), "-", round(z_upper, 1), "\n")
  cat(n_sigmas, "сигма выбросы:", length(z_out), "- Значения:", z_out, "\n")
  
  return(list(
    boxplot = bp_out, 
    zscore = z_out,
    bounds = list(
      boxplot = c(bp_lower, bp_upper),
      zscore = c(z_lower, z_upper)
    )
  ))
}

outliers = analyze_outliers(rated_QP$original, 2)

ggplot(rated_QP, aes(x = time_index, y = original)) +
  # Области
  geom_ribbon(aes(ymin = outliers$bounds$boxplot[1], 
                  ymax = outliers$bounds$boxplot[2]),
              fill = "lightblue", alpha = 0.4) +
  geom_ribbon(aes(ymin = outliers$bounds$zscore[1], 
                  ymax = outliers$bounds$zscore[2]),
              fill = "pink1", alpha = 0.2) +
  
  # Точки данных
  geom_point(aes(color = case_when(
    original %in% outliers$boxplot ~ "Box-plot выброс",
    original %in% outliers$zscore ~ "n-σ выброс",
    TRUE ~ "Норма"
  )), size = 3) +
  
  # Границы
  geom_hline(yintercept = outliers$bounds$boxplot, 
             linetype = "dashed", color = "blue", size = 1) +
  geom_hline(yintercept = outliers$bounds$zscore, 
             linetype = "dotted", color = "red", size = 1) +
  
  scale_color_manual(values = c(
    "Box-plot выброс" = "red",
    "n-σ выброс" = "orange",
    "Норма" = "grey50"
  )) +
  labs(title = "Визуализация выбросов и допустимых границ",
       subtitle = "Розовая область - n-σ, Синяя - box-plot",
       x = "Временной индекс", y = "Рейтинг",
       color = "Классификация") +
  theme_bw()
colors()


# Очистка выбросов
cleaned_QP = rated_QP %>%
  filter(!(original %in% outliers$boxplot)) # можно использовать объединение boxplot и zscore, если хочешь строже
View(cleaned_QP)

# Создадим пропуски по схеме: с 10 по 15 индекс (имитация сбоя работы api)
missing_range = 10:15

cleaned_QP_with_gaps = cleaned_QP %>%
  mutate(
    thinned = ifelse(time_index %in% missing_range, NA, original)
  )
View(cleaned_QP_with_gaps)

cleaned_QP_filled = cleaned_QP_with_gaps %>%
  mutate(
    mean_filled = coalesce(thinned, mean(thinned, na.rm = TRUE)),
    median_filled = coalesce(thinned, median(thinned, na.rm = TRUE)),
    mean_3_previous = zoo::na.fill(
      zoo::rollmeanr(zoo::na.fill(thinned, "extend"), 3, fill = NA), 
      "extend"
    ),
    lagrange = {
      known = which(!is.na(thinned))
      missing = which(is.na(thinned))
      valid_missing = missing[missing >= min(known) & missing <= max(known)]
      if(length(known) >= 2 & length(valid_missing) > 0) {
        result = thinned
        result[valid_missing] = barylag(known, thinned[known], valid_missing)
        result
      } else thinned
    },
    spline = na.spline(thinned)
  )
View(cleaned_QP_filled)

library(Metrics)
# Истинные и предсказанные значения
true_vals = cleaned_QP_with_gaps$original[missing_range]

mean_pred = cleaned_QP_filled$mean_filled[missing_range]
median_pred = cleaned_QP_filled$median_filled[missing_range]
mean3_pred = cleaned_QP_filled$mean_3_previous[missing_range]
lagrange_pred = cleaned_QP_filled$lagrange[missing_range]
spline_pred = cleaned_QP_filled$spline[missing_range]

# Вспомогательная функция для одной строки метрик
get_metrics <- function(true, pred, method) {
  data.frame(
    Method = method,
    MAE = mae(true, pred),
    RMSE = rmse(true, pred),
    MAPE = mape(true, pred) * 100, # умножаем на 100 для %
    MASE = mae(true, pred) / mean(abs(diff(true)), na.rm = TRUE)
  )
}

# Объединяем в таблицу
metrics_df <- bind_rows(
  get_metrics(true_vals, mean_pred, "Mean"),
  get_metrics(true_vals, median_pred, "Median"),
  get_metrics(true_vals, mean3_pred, "Mean 3 prev"),
  get_metrics(true_vals, lagrange_pred, "Lagrange"),
  get_metrics(true_vals, spline_pred, "Spline")
)

print(metrics_df)


print(plot_fill(cleaned_QP_filled, "mean_filled", "Среднее"))
print(plot_fill(cleaned_QP_filled, "median_filled", "Медиана"))
print(plot_fill(cleaned_QP_filled, "mean_3_previous", "Среднее 3 предыдущих"))
print(plot_fill(cleaned_QP_filled, "lagrange", "Лагранж"))
print(plot_fill(cleaned_QP_filled, "spline", "Сплайн"))


library(knitr)
kable(metrics_df, caption = "Сравнение метрик восстановления пропусков")
