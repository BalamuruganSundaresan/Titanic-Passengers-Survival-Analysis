# ============================================================
# WEEK 1 PROJECT: Titanic Data Cleaning & Preliminary Analysis
# Author: <Your Name>
# Tool: R / RStudio
# Dataset: Kaggle Titanic - Machine Learning from Disaster
# ============================================================

# 0. Install packages once if needed
packages <- c("tidyverse", "janitor", "skimr", "naniar", "scales")
new_packages <- packages[!(packages %in% installed.packages()[, "Package"])]
if(length(new_packages)) install.packages(new_packages)

library(tidyverse)
library(janitor)
library(skimr)
library(naniar)
library(scales)

# 1. Project folders
dir.create("data", showWarnings = FALSE)
dir.create("outputs", showWarnings = FALSE)

# Download train.csv from Kaggle and save it as:
# data/train.csv
# Source: https://www.kaggle.com/c/titanic/data

# 2. Import
raw <- read_csv("data/train.csv", show_col_types = FALSE) %>%
  clean_names()

# 3. Initial profiling
dim(raw)
names(raw)
str(raw)
summary(raw)
skim(raw)

# Missing-value profile
missing_tbl <- raw %>%
  summarise(across(everything(), ~sum(is.na(.)))) %>%
  pivot_longer(everything(), names_to = "variable", values_to = "missing") %>%
  mutate(missing_pct = round(100 * missing / nrow(raw), 2)) %>%
  arrange(desc(missing))

print(missing_tbl)
write_csv(missing_tbl, "outputs/missing_value_summary.csv")

# 4. Duplicate checks
duplicate_rows <- raw %>% duplicated() %>% sum()
duplicate_passenger_ids <- raw %>%
  count(passenger_id) %>%
  filter(n > 1)

cat("Duplicate rows:", duplicate_rows, "\n")
print(duplicate_passenger_ids)

# 5. Data cleaning
clean <- raw %>%
  mutate(
    survived = factor(survived, levels = c(0, 1),
                     labels = c("No", "Yes")),
    pclass = factor(pclass, levels = c(1, 2, 3),
                    labels = c("1st", "2nd", "3rd")),
    sex = factor(sex),
    embarked = factor(embarked)
  )

# Extract useful information before removing high-missing raw columns
clean <- clean %>%
  mutate(
    title = str_extract(name, "(?<=, )[^.]+(?=\\.)") %>%
      str_squish(),
    title = case_when(
      title %in% c("Mlle", "Ms") ~ "Miss",
      title == "Mme" ~ "Mrs",
      title %in% c("Lady", "Countess", "Capt", "Col", "Don",
                   "Dr", "Major", "Rev", "Sir", "Jonkheer", "Dona") ~ "Rare",
      TRUE ~ title
    ),
    family_size = sib_sp + parch + 1,
    is_alone = if_else(family_size == 1, "Yes", "No"),
    deck = case_when(
      is.na(cabin) ~ "Unknown",
      TRUE ~ str_sub(cabin, 1, 1)
    )
  )

# Age: median imputation within passenger class + sex
clean <- clean %>%
  group_by(pclass, sex) %>%
  mutate(age = if_else(
    is.na(age),
    median(age, na.rm = TRUE),
    age
  )) %>%
  ungroup()

# Embarked: replace missing values with the mode
embarked_mode <- clean %>%
  filter(!is.na(embarked)) %>%
  count(embarked, sort = TRUE) %>%
  slice(1) %>%
  pull(embarked) %>%
  as.character()

clean <- clean %>%
  mutate(
    embarked = fct_explicit_na(embarked, na_level = embarked_mode),
    fare = if_else(is.na(fare), median(fare, na.rm = TRUE), fare)
  )

# Cabin has very high missingness; retain useful Deck feature and drop raw Cabin
clean <- clean %>%
  select(
    passenger_id, survived, pclass, sex, age, sib_sp, parch,
    family_size, is_alone, fare, embarked, title, deck
  )

# 6. Post-cleaning validation
print(colSums(is.na(clean)))
print(dim(clean))
print(summary(clean))

# 7. Outlier detection using IQR
iqr_limits <- function(x) {
  q1 <- quantile(x, 0.25, na.rm = TRUE)
  q3 <- quantile(x, 0.75, na.rm = TRUE)
  iqr <- q3 - q1
  c(lower = q1 - 1.5 * iqr, upper = q3 + 1.5 * iqr)
}

age_limits <- iqr_limits(clean$age)
fare_limits <- iqr_limits(clean$fare)

clean <- clean %>%
  mutate(
    age_outlier = age < age_limits["lower"] | age > age_limits["upper"],
    fare_outlier = fare < fare_limits["lower"] | fare > fare_limits["upper"]
  )

cat("Age outliers:", sum(clean$age_outlier), "\n")
cat("Fare outliers:", sum(clean$fare_outlier), "\n")

# Do NOT automatically delete these observations.
# In this project, outliers are flagged and retained because high fares
# can represent legitimate first-class passengers.

# 8. Normalization / standardization
clean <- clean %>%
  mutate(
    age_scaled = as.numeric(scale(age)),
    fare_scaled = as.numeric(scale(fare)),
    family_size_scaled = as.numeric(scale(family_size))
  )

# 9. Encoding categorical variables
# Factors are kept for EDA. One-hot encoding is demonstrated separately.
model_matrix <- model.matrix(
  ~ pclass + sex + embarked + title + deck + is_alone - 1,
  data = clean
)

encoded <- bind_cols(
  clean %>% select(passenger_id, survived, age_scaled,
                   fare_scaled, family_size_scaled),
  as.data.frame(model_matrix)
)

write_csv(encoded, "outputs/titanic_encoded.csv")
write_csv(clean, "outputs/titanic_cleaned.csv")

# 10. Descriptive statistics
numeric_summary <- clean %>%
  summarise(
    age_mean = mean(age),
    age_median = median(age),
    age_sd = sd(age),
    fare_mean = mean(fare),
    fare_median = median(fare),
    fare_sd = sd(fare),
    family_size_mean = mean(family_size),
    family_size_median = median(family_size)
  )

print(numeric_summary)
write_csv(numeric_summary, "outputs/numeric_summary.csv")

# Survival rate overall
survival_rate <- clean %>%
  count(survived) %>%
  mutate(rate = n / sum(n))

print(survival_rate)

# Survival by sex
survival_by_sex <- clean %>%
  group_by(sex) %>%
  summarise(
    passengers = n(),
    survivors = sum(survived == "Yes"),
    survival_rate = mean(survived == "Yes")
  )

print(survival_by_sex)

# Survival by class
survival_by_class <- clean %>%
  group_by(pclass) %>%
  summarise(
    passengers = n(),
    survivors = sum(survived == "Yes"),
    survival_rate = mean(survived == "Yes")
  )

print(survival_by_class)

# Sex + class interaction
survival_sex_class <- clean %>%
  group_by(pclass, sex) %>%
  summarise(
    passengers = n(),
    survival_rate = mean(survived == "Yes"),
    .groups = "drop"
  )

print(survival_sex_class)

# 11. Correlation among numeric variables
numeric_data <- clean %>%
  select(age, sib_sp, parch, family_size, fare)

cor_matrix <- cor(numeric_data, use = "pairwise.complete.obs")
print(round(cor_matrix, 3))

write.csv(round(cor_matrix, 3),
          "outputs/correlation_matrix.csv")

# 12. Visualizations

# Plot 1: Missing values before cleaning
p_missing <- gg_miss_var(raw, show_pct = TRUE) +
  labs(title = "Missing Values by Variable",
       x = "Variable", y = "Number Missing") +
  theme_minimal()

ggsave("outputs/01_missing_values.png",
       p_missing, width = 8, height = 5, dpi = 300)

# Plot 2: Survival distribution
p_survival <- ggplot(clean, aes(x = survived)) +
  geom_bar() +
  labs(title = "Passenger Survival Distribution",
       x = "Survived", y = "Passengers") +
  theme_minimal()

ggsave("outputs/02_survival_distribution.png",
       p_survival, width = 7, height = 5, dpi = 300)

# Plot 3: Survival by sex
p_sex <- ggplot(clean, aes(x = sex, fill = survived)) +
  geom_bar(position = "fill") +
  scale_y_continuous(labels = percent) +
  labs(title = "Survival Proportion by Sex",
       x = "Sex", y = "Proportion") +
  theme_minimal()

ggsave("outputs/03_survival_by_sex.png",
       p_sex, width = 7, height = 5, dpi = 300)

# Plot 4: Survival by passenger class
p_class <- ggplot(clean, aes(x = pclass, fill = survived)) +
  geom_bar(position = "fill") +
  scale_y_continuous(labels = percent) +
  labs(title = "Survival Proportion by Passenger Class",
       x = "Passenger Class", y = "Proportion") +
  theme_minimal()

ggsave("outputs/04_survival_by_class.png",
       p_class, width = 7, height = 5, dpi = 300)

# Plot 5: Age distribution
p_age <- ggplot(clean, aes(x = age, fill = survived)) +
  geom_histogram(bins = 30, alpha = 0.65, position = "identity") +
  labs(title = "Age Distribution by Survival Status",
       x = "Age", y = "Count") +
  theme_minimal()

ggsave("outputs/05_age_distribution.png",
       p_age, width = 8, height = 5, dpi = 300)

# Plot 6: Fare boxplot
p_fare <- ggplot(clean, aes(x = pclass, y = fare)) +
  geom_boxplot() +
  labs(title = "Fare Distribution and Potential Outliers",
       x = "Passenger Class", y = "Fare") +
  theme_minimal()

ggsave("outputs/06_fare_boxplot.png",
       p_fare, width = 8, height = 5, dpi = 300)

# Plot 7: Family size vs survival
p_family <- ggplot(clean, aes(x = family_size, fill = survived)) +
  geom_bar(position = "dodge") +
  labs(title = "Family Size and Survival",
       x = "Family Size", y = "Passengers") +
  theme_minimal()

ggsave("outputs/07_family_size.png",
       p_family, width = 8, height = 5, dpi = 300)

# Plot 8: Fare vs Age
p_scatter <- ggplot(clean, aes(x = age, y = fare, color = survived)) +
  geom_point(alpha = 0.55) +
  labs(title = "Age vs Fare by Survival Status",
       x = "Age", y = "Fare") +
  theme_minimal()

ggsave("outputs/08_age_vs_fare.png",
       p_scatter, width = 8, height = 5, dpi = 300)

# 13. Export final report-ready tables
write_csv(survival_by_sex, "outputs/survival_by_sex.csv")
write_csv(survival_by_class, "outputs/survival_by_class.csv")
write_csv(survival_sex_class, "outputs/survival_by_sex_and_class.csv")

# 14. Final data quality checks
stopifnot(nrow(clean) == nrow(raw))
stopifnot(sum(is.na(clean$age)) == 0)
stopifnot(sum(is.na(clean$fare)) == 0)

cat("\nPROJECT COMPLETE\n")
cat("Cleaned rows:", nrow(clean), "\n")
cat("Cleaned columns:", ncol(clean), "\n")
cat("All required output files are in the outputs/ folder.\n")
