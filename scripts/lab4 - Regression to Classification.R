# setwd("")
# install.packages(c("tidyverse","lubridate","caret","e1071","Metrics","MLmetrics","pROC","scales","glmnet","FNN","recipes"))
library(tidyverse)
library(lubridate)
library(caret)
library(e1071)
library(Metrics)
library(MLmetrics)
library(pROC)
library(scales)
library(recipes)
set.seed(123)

# https://www.kaggle.com/datasets/anassarfraz13/student-success-factors-and-insights

# От автора датасета:
# This dataset contains information of about 6,590 students and the factors that may affect their academic performance. 
# It includes variables such as study habits, attendance, parental involvement, access to resources, 
# extracurricular activities, sleep hours, motivation, and socio-economic background. 
# Academic results are measured through previous and final exam scores
students_data = read.csv("lab4 - StudentPerformanceFactors.csv", stringsAsFactors = FALSE, fileEncoding = "UTF-8")
glimpse(students_data)
summary(students_data)

# Приведём типы
students_data = students_data %>%
  mutate(
    Parental_Involvement = factor(Parental_Involvement),
    Access_to_Resources = factor(Access_to_Resources),
    Extracurricular_Activities = factor(Extracurricular_Activities),
    Motivation_Level = factor(Motivation_Level),
    Internet_Access = factor(Internet_Access),
    Family_Income = factor(Family_Income),
    Teacher_Quality = factor(Teacher_Quality),
    School_Type = factor(School_Type),
    Peer_Influence = factor(Peer_Influence),
    Learning_Disabilities = factor(Learning_Disabilities),
    Parental_Education_Level = factor(Parental_Education_Level),
    Distance_from_Home = factor(Distance_from_Home),
    Gender = factor(Gender),
  )
glimpse(students_data)


# --- 0. Выбор наиболее коррелирующих параметров ----
# Сначала оставим только числовые предикторы
num_students_data = students_data %>% select(Hours_Studied, Attendance, Sleep_Hours, Previous_Scores, 
                                 Tutoring_Sessions, Physical_Activity,
                                 Exam_Score)
glimpse(num_students_data)

# install.packages(c('Hmisc', 'corrplot'))
library(Hmisc)
library(corrplot)
corr_results = rcorr(as.matrix(num_students_data))
corr_results$r
corr_results$P

corrplot(
  corr_results$r,
  method = "color",
  type = "upper", # только верхний треугольник
  order = "hclust", # упорядочить по кластеру
  tl.col = "black", # цвет подписей переменных
  tl.srt = 45, # угол текста
  addCoef.col = "black", # показывать r
  p.mat = corr_results$P,
  sig.level = 0.05, # значимость
  insig = "pch" # или "pch", "n", "blank"
)


# --- 1.1 Построение моделей МНК ----
model_students_data = students_data %>%
  select(Exam_Score, 
         Tutoring_Sessions, Attendance, Hours_Studied, Previous_Scores, Physical_Activity,
         Access_to_Resources, Internet_Access, School_Type, Learning_Disabilities, Parental_Education_Level)

# Линейная модель
lm_model = lm(Exam_Score ~ Tutoring_Sessions + Attendance + Hours_Studied + Previous_Scores, data = model_students_data)
summary(lm_model)

# Квадратичная модель
lm_quad_model = lm(Exam_Score ~ Tutoring_Sessions + Attendance + Hours_Studied + Previous_Scores + Physical_Activity +
                      I(Previous_Scores^2), data = model_students_data)
summary(lm_quad_model)

# Полиномы
library(splines)
lm_spline = lm(Exam_Score ~ bs(Tutoring_Sessions, df = 4) +
                  bs(Attendance, df = 4) +
                  bs(Hours_Studied, df = 4),
                data = model_students_data)
summary(lm_spline)

# Категориальные переменные
lm_dummies_model = lm(
  Exam_Score ~ Tutoring_Sessions + Attendance + Hours_Studied + Previous_Scores + 
    Access_to_Resources + Internet_Access + School_Type + Learning_Disabilities + Parental_Education_Level,
  data = model_students_data
)
summary(lm_dummies_model)

# --- 1.2 Построение моделей SVM ----
svm_model = svm(Exam_Score ~ Tutoring_Sessions + Attendance + Hours_Studied + Previous_Scores + 
                    Access_to_Resources + Internet_Access + School_Type + Learning_Disabilities + Parental_Education_Level,
                data = model_students_data,
                type = "eps-regression", kernel = "radial")  # rbf kernel
pred = predict(svm_model, model_students_data)
rmse(model_students_data$Exam_Score, pred)

train_control <- trainControl(method="cv", number=3, search="random", verboseIter = TRUE)
svm_caret <- train(
  Exam_Score ~ Tutoring_Sessions + Attendance + Hours_Studied + Previous_Scores + 
    Access_to_Resources + Internet_Access + School_Type + Learning_Disabilities + Parental_Education_Level,
  data = model_students_data,
  method = "svmRadial",
  trControl = train_control,
  tuneLength = 10,
  preProcess = c("center","scale")
)
pred = predict(svm_caret, model_students_data)
rmse(model_students_data$Exam_Score, pred)


# --- 2. Оценка моделей ----
library(ggplot2)
library(dplyr)
library(tidyr)
predictions_df <- model_students_data %>%
  select(Exam_Score) %>%
  mutate(
    LM_linear = predict(lm_model, model_students_data),
    LM_quad = predict(lm_quad_model, model_students_data),
    LM_spline = predict(lm_spline, model_students_data),
    LM_dummies = predict(lm_dummies_model, model_students_data),
    SVM_basic = predict(svm_model, model_students_data),
    SVM_caret = predict(svm_caret, model_students_data)
  )

MASE_custom <- function(true, pred) {
  n <- length(true)
  mae_model <- mean(abs(true - pred))
  mae_naive <- mean(abs(diff(true)))
  return(mae_model / mae_naive)
}

calc_metrics <- function(true, pred){
  tibble(
    MAE  = mae(true, pred),
    RMSE = rmse(true, pred),
    MAPE = MAPE(pred, true),
    MASE = MASE_custom(true, pred)
  )
}

metrics_list <- lapply(names(predictions_df)[-1], function(col){
  calc_metrics(predictions_df$Exam_Score, predictions_df[[col]]) %>%
    mutate(Model = col)
})

metrics_df <- bind_rows(metrics_list) %>% select(Model, everything())
print(metrics_df %>% arrange(RMSE))

# График фактических vs предсказанных значений
pred_long <- predictions_df %>%
  pivot_longer(-Exam_Score, names_to = "Model", values_to = "Predicted")

ggplot(pred_long, aes(x = Exam_Score, y = Predicted, color = Model)) +
  geom_point(alpha = 0.6, size = 1.5) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed") +
  facet_wrap(~Model) +
  labs(title = "Фактические vs Предсказанные значения Exam_Score",
       x = "Фактическое Exam_Score",
       y = "Предсказанное") +
  theme_minimal()

# Скрипичные диаграммы (Violin plot)
pred_long2 <- pred_long %>%
  pivot_longer(cols = c(Exam_Score, Predicted), names_to = "Type", values_to = "Value")

ggplot(pred_long2, aes(x = Model, y = Value, fill = Type)) +
  geom_violin(alpha = 0.5, position = position_dodge(width = 0.8)) +
  geom_boxplot(width = 0.1, position = position_dodge(width = 0.8)) +
  labs(title = "Сравнение распределений: фактические vs предсказанные",
       x = "Модель", y = "Exam_Score") +
  theme_minimal() +
  scale_fill_manual(values = c("Exam_Score"="skyblue","Predicted"="orange"))


# --- 3. Сведение к классификации ----
model_students_data <- model_students_data %>%
  mutate(
    Exam_Score_Class = ifelse(Exam_Score >= 70, "High", "Low"),
    Exam_Score_Class = factor(Exam_Score_Class, levels = c("Low", "High"))
  )
table(model_students_data$Exam_Score_Class)

# Выбираем признаки для классификации
features <- c("Hours_Studied", "Previous_Scores", "Attendance")
X <- model_students_data[, features]
y <- model_students_data$Exam_Score_Class

# Масштабирование (для kNN и SVM)
preproc <- preProcess(X, method = c("center", "scale"))
X_scaled <- predict(preproc, X)
data_class <- cbind(X_scaled, Exam_Score_Class = y)


# --- 4. Классификация ----
# --- 4a. kNN ----
train_index <- createDataPartition(data_class$Exam_Score_Class, p = 0.8, list = FALSE)
train_data <- data_class[train_index,]
test_data  <- data_class[-train_index,]

ctrl <- trainControl(
  method = "cv",
  number = 3,
  classProbs = TRUE,
  summaryFunction = twoClassSummary,
  verboseIter = TRUE
)
knn_caret_auto <- train(
  Exam_Score_Class ~ ., 
  data = train_data, 
  method = "knn",
  trControl = ctrl,
  tuneLength = 15,  # caret выберет 15 значений k
  metric = "ROC",
  preProcess = c("center", "scale")
)

print(knn_caret_auto)

knn_pred_prob <- predict(knn_caret_auto, test_data, type = "prob")[,"High"]
knn_pred <- predict(knn_caret_auto, test_data)

# --- 4b. SVM ----
svm_model_class <- svm(Exam_Score_Class ~ ., data = train_data, kernel = "radial", probability = TRUE)
svm_pred_prob <- attr(predict(svm_model_class, test_data, probability = TRUE), "probabilities")[,"High"]
svm_pred <- predict(svm_model_class, test_data)


# --- 5. Оценка качества классификации ----
# Функция для метрик
eval_class <- function(true, pred, prob){
  true <- factor(true, levels = c("Low", "High"))
  pred <- factor(pred, levels = c("Low", "High"))
  
  cm <- confusionMatrix(pred, true, positive = "High")
  true_numeric <- ifelse(true == "High", 1, 0)
  
  roc_obj <- roc(true_numeric, as.numeric(prob))
  auc_val <- as.numeric(auc(roc_obj))
  
  tibble(
    Accuracy = cm$overall["Accuracy"],
    F1 = cm$byClass["F1"],
    LogLoss = -mean(ifelse(true_numeric == 1, log(prob), log(1 - prob))),
    ROC_AUC = auc_val
  )
}

metrics_knn <- eval_class(test_data$Exam_Score_Class, knn_pred, knn_pred_prob) %>% mutate(Model="kNN")
metrics_svm <- eval_class(test_data$Exam_Score_Class, svm_pred, svm_pred_prob) %>% mutate(Model="SVM")

metrics_all <- bind_rows(metrics_knn, metrics_svm)
print(metrics_all)


# --- ROC кривые ----
roc_knn <- roc(test_data$Exam_Score_Class, knn_pred_prob)
roc_svm <- roc(test_data$Exam_Score_Class, svm_pred_prob)

roc_df <- bind_rows(
  tibble(FPR = 1 - roc_knn$specificities, TPR = roc_knn$sensitivities, Model="kNN"),
  tibble(FPR = 1 - roc_svm$specificities, TPR = roc_svm$sensitivities, Model="SVM")
)

ggplot(roc_df, aes(x=FPR, y=TPR, color=Model)) +
  geom_line(size=1.2) +
  geom_abline(intercept=0, slope=1, linetype="dashed", color="gray") +
  labs(title="ROC Curve Comparison: kNN vs SVM", x="False Positive Rate", y="True Positive Rate") +
  theme_bw() +
  scale_color_manual(values=c("kNN"="blue", "SVM"="red"))

# --- 5a. Перевод регрессии в классификацию ----
pred_class_from_reg <- factor(ifelse(predictions_df$SVM_basic >= 70, "High", "Low"), 
                             levels = c("Low", "High"))
conf_reg <- confusionMatrix(pred_class_from_reg, model_students_data$Exam_Score_Class, positive = "High")
print(conf_reg)

# Для ROC кривой регрессионной модели нам нужны вероятности
# Поскольку у регрессии нет вероятностей, создадим их на основе расстояния до порога
reg_probabilities <- predictions_df$SVM_basic / 100  # Нормализуем от 0 до 100 в [0,1]
reg_probabilities <- pmin(pmax(reg_probabilities, 0.01), 0.99)  # Ограничиваем диапазон

# Создаем ROC объект для регрессионной модели
true_numeric <- ifelse(model_students_data$Exam_Score_Class == "High", 1, 0)
roc_reg <- roc(true_numeric, reg_probabilities)

# Обновляем ROC dataframe
roc_df <- bind_rows(
  tibble(FPR = 1 - roc_knn$specificities, TPR = roc_knn$sensitivities, Model = "kNN"),
  tibble(FPR = 1 - roc_svm$specificities, TPR = roc_svm$sensitivities, Model = "SVM"),
  tibble(FPR = 1 - roc_reg$specificities, TPR = roc_reg$sensitivities, Model = "Reg->Class")
)

ggplot(roc_df, aes(x = FPR, y = TPR, color = Model)) +
  geom_line(size = 1.2) +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "gray") +
  labs(title = "ROC Curve Comparison: kNN vs SVM vs Reg->Class", 
       x = "False Positive Rate", 
       y = "True Positive Rate") +
  theme_bw() +
  scale_color_manual(values = c("kNN" = "blue", "SVM" = "red", "Reg->Class" = "green"))


# Функция для оценки регрессионной модели как классификатора
eval_reg_model <- function(true_score, pred_score) {
  # Классы на основе порога 70
  true_class <- factor(ifelse(true_score >= 70, "High", "Low"), levels = c("Low", "High"))
  pred_class <- factor(ifelse(pred_score >= 70, "High", "Low"), levels = c("Low", "High"))
  
  # "Вероятности" для ROC - нормализуем предсказанные значения в [0,1]
  prob <- (pred_score - min(pred_score)) / (max(pred_score) - min(pred_score))
  prob <- pmin(pmax(prob, 0.001), 0.999)  # Ограничиваем для избежания log(0)
  
  cm <- confusionMatrix(pred_class, true_class, positive = "High")
  true_num <- as.numeric(true_class == "High")
  
  # Проверяем, что есть оба класса
  if (length(unique(true_num)) < 2) {
    roc_auc <- NA
    warning("Only one class present, ROC AUC cannot be calculated")
  } else {
    roc_obj <- roc(true_num, prob)
    roc_auc <- as.numeric(auc(roc_obj))
  }
  
  # Вычисляем LogLoss с защитой от log(0)
  epsilon <- 1e-15
  prob_clipped <- pmin(pmax(prob, epsilon), 1 - epsilon)
  logloss <- -mean(ifelse(true_num == 1, log(prob_clipped), log(1 - prob_clipped)))
  
  tibble(
    Accuracy = round(cm$overall["Accuracy"], 4),
    F1 = round(cm$byClass["F1"], 4),
    LogLoss = round(logloss, 4),
    ROC_AUC = round(roc_auc, 4)
  )
}

metrics_all <- bind_rows(
  eval_class(true = test_data$Exam_Score_Class, pred = knn_pred, prob = knn_pred_prob) %>%
    mutate(Model = "kNN"),
  
  eval_class(true = test_data$Exam_Score_Class, pred = svm_pred, prob = svm_pred_prob) %>%
    mutate(Model = "SVM"),
  
  # Используем исходные числовые баллы и предсказания регрессионной модели
  eval_reg_model(
    true_score = model_students_data$Exam_Score, 
    pred_score = predictions_df$SVM_basic
  ) %>%
    mutate(Model = "SVM_Reg")
)

print(metrics_all)

