# setwd()
graphics.off()

install.packages("ggplot2")
install.packages("dplyr") # для манипуляций с данными: filter, mutate, arrange, select
install.packages("tidyr") # для преобразования данных: pivot_longer

library(ggplot2)
library(dplyr)
library(tidyr)

# https://www.kaggle.com/datasets/datasnaek/chess: 
# От автора датасета "This is a set of just over 20,000 games collected from a selection of users on the site Lichess.org
# I collected this data using the Lichess API, which enables collection of any given users game history".
games_data = read.csv("lab1 - games.csv", header =T, sep = ",", encoding = "UTF-8")
View(games_data)
summary(games_data)

# Линейная диаграмма
help(plot)
plot(games_data$white_rating, games_data$black_rating, type="p", col="blue", panel.first=grid(), 
  xlab = "Рейтинг белых", ylab = "Рейтинг чёрных",
  main="Рейтинг чёрных в зависимости от рейтинга белых",) # axes = FALSE

sicilian_def_games = games_data[games_data$opening_name == "Sicilian Defense",]
plot(sicilian_def_games$white_rating, sicilian_def_games$black_rating, type="p", col="blue", panel.first=grid(), 
  xlab = "Рейтинг белых", ylab = "Рейтинг чёрных", 
  main="Рейтинг чёрных в зависимости от рейтинга белых (сицилийская защита)",)

# с помощью ggplot
help("ggplot2")
ggplot(sicilian_def_games, aes(x = white_rating, y = black_rating)) +
  geom_point(alpha = 0.3, color = "blue") +
  geom_smooth(method = "lm", color = "red", se = TRUE) + # se = TRUE - доверительный интервал
  labs(x = "Рейтинг белых", y = "Рейтинг чёрных", 
       title = "Рейтинг чёрных в зависимости от рейтинга белых (сицилийская защита)") +
  theme_bw()
help("theme_bw")

# Сравнение рейтингов белых и чёрных игроков
# Преобразуем данные в длинный формат
games_long = games_data %>%
  pivot_longer(cols = c(white_rating, black_rating),
               names_to = "color", 
               values_to = "rating") %>% 
  mutate(color = ifelse(color == "white_rating", "Белые", "Чёрные"))

# Графики плотности с наложением
ggplot(games_long, aes(x = rating, color = color)) +
  geom_density(alpha = 0.5) +
  scale_color_manual(values = c("Белые" = "blue", "Чёрные" = "red")) +
  labs(x = "Рейтинг", y = "Плотность", 
       title = "Плотность распределения рейтингов белых и чёрных игроков",
       fill = "Цвет фигур", color = "Цвет фигур") +
  theme_bw()

# Две гистограммы
ggplot(games_long, aes(x = rating, fill = color)) +
  geom_histogram(alpha = 0.7, bins = 50) +
  facet_wrap(~ color, ncol = 2) +
  scale_fill_manual(values = c("Белые" = "gray", "Чёрные" = "black")) +
  labs(x = "Рейтинг", y = "Количество игр", 
       title = "Распределение рейтингов белых и чёрных игроков") +
  theme_bw() +
  theme(legend.position = "none") # убираем легенду, т.к. есть заголовки

# Два violin + boxplot
ggplot(games_long, aes(x = color, y = rating, fill = color)) +
  geom_violin(alpha = 0.5, width = 0.8) +
  geom_boxplot(width = 0.2, alpha = 0.7) +
  scale_fill_manual(values = c("Белые" = "gray", "Чёрные" = "black")) +
  labs(x = "Цвет фигур", y = "Рейтинг", 
       title = "Распределение рейтингов: Violin + Boxplot") +
  theme_bw() +
  theme(legend.position = "none")

# Сравнение итогов партий
# Круговая диаграмма
status_counts = games_data %>% # Количество игр по статусу
  count(victory_status) %>%
  mutate(percentage = n / sum(n) * 100)

ggplot(status_counts, aes(x = "", y = n, fill = victory_status)) +
  geom_bar(stat = "identity", width = 1) +
  geom_text(aes(label = paste0(round(percentage, 1), "%")), 
            position = position_stack(vjust = 0.5)) +
  coord_polar("y", start = 0) +
  labs(title = "Распределение статусов игр",
       fill = "Статус игры") +
  theme_void() 
help("position_stack")

# Столбчатая диаграмма
ggplot(status_counts, aes(x = victory_status, y = n, fill = victory_status)) +
  geom_col() +
  labs(title = "Распределение статусов игр",
       x = "Статус", y = "Количество игр") +
  theme_bw() +
  theme(legend.position = "none")


# Area chart для min, max, avg рейтинга
rapid_resign_games = games_data[games_data$increment_code == "15+10" & games_data$victory_status == "resign",]
min_max_avg = rapid_resign_games %>%
  mutate(date = as.Date(as.POSIXct(created_at/1000, origin = "1970-01-01"))) %>%
  group_by(date) %>%
  summarise(
    min_rating = min(white_rating),
    max_rating = max(white_rating),
    avg_rating = mean(white_rating)
  )

ggplot(min_max_avg, aes(x = date)) +
  geom_ribbon(aes(ymin = min_rating, ymax = max_rating), 
              fill = "red", alpha = 0.3) +
  geom_line(aes(y = avg_rating), color = "blue") +
  labs(title = "Min/Max рейтинг белых (рапид 15+10, resign)", x = "Дата", y = "Рейтинг белых") +
  theme_minimal()

