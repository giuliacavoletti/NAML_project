
## - Logistic linear regression  

### Librerie ############################################################

library(rms)
library(arm)
library(pROC)
library(PRROC)
library(ResourceSelection)
library(mgcv)


# Load the data
lw = read.csv("train.csv", header = TRUE)

# Cleaning the data
# We create a clean dataset removing NAs, we remove outliers and data which we don't need
lw_clean = na.omit(lw) 
#  Fix "Age 0" (Impossible for credit)
# We usually filter for age >= 18 or 21
lw_clean <- subset(lw_clean, age >= 18)

#  Cap 'RevolvingUtilization' 
# Any value > 10 (1000% utilization) is likely an error or extreme outlier. 
lw_clean$RevolvingUtilizationOfUnsecuredLines <- pmin(lw_clean$RevolvingUtilizationOfUnsecuredLines, 10)


# Income spans huge orders of magnitude. Log compresses it so 10k vs 100k is comparable to 100k vs 1M.
# We add +1 to avoid log(0).
lw_clean$LogIncome <- log(lw_clean$MonthlyIncome + 1)

# Cap DebtRatio
# Ratios > 5 or 10 usually mean "Divided by Zero income". Cap at a reasonable high number.
lw_clean$DebtRatio <- pmin(lw_clean$DebtRatio, 10)

# Fit the model using the CLEAN data

#mod.low = glm(SeriousDlqin2yrs ~ RevolvingUtilizationOfUnsecuredLines + age + MonthlyIncome + DebtRatio, 
              #family = binomial(link = logit), 
              #data = lw_clean)
mod.low = glm(SeriousDlqin2yrs ~ RevolvingUtilizationOfUnsecuredLines + 
                age + I(age^2) +  # Added squared term for better fit
                NumberOfTime30.59DaysPastDueNotWorse + 
                DebtRatio + 
                LogIncome +       
                NumberOfOpenCreditLinesAndLoans + 
                NumberOfTimes90DaysLate + 
                NumberRealEstateLoansOrLines + 
                NumberOfTime60.89DaysPastDueNotWorse + 
                NumberOfDependents, 
              family = binomial(link = "logit"), 
              data = lw_clean)


summary(mod.low)

# Check Goodness of Fit (GOF)
hoslem.test(mod.low$y, fitted(mod.low), g = 6)


# (Mis-)Classification Tables

# Find the threshold that maximizes Sensitivity + Specificity
best_coords <- coords(roc_obj, "best", ret = "threshold")
print(best_coords)

# Use this new threshold
soglia <- best_coords$threshold

# We use mod.low$y to ensure we represent exactly what the model used
valori.reali    = mod.low$y  
valori.predetti = as.numeric(fitted(mod.low) > soglia)
# 1 if > threshold, 0 if <= threshold

# Create the Confusion Matrix
tab = table(Reali = valori.reali, Predetti = valori.predetti)
print(tab)

# % di casi classificati correttamente:
round( sum( diag( tab ) ) / sum( tab ), 2 )

# % di casi misclassificati:
round( ( tab [ 1, 2 ] + tab [ 2, 1 ] ) / sum( tab ), 2 )

# Sensitivity
sensitivita =  tab [ 2, 2 ] /( tab [ 2, 1 ] + tab [ 2, 2 ] ) 
sensitivita

# Specificity 
specificita = tab[ 1, 1 ] /( tab [ 1, 2 ] + tab [ 1, 1 ] )
specificita

#ora testo il modello sul dataset "test" fornito da kaggle

test_df <- read.csv("test.csv", header = TRUE)

#prima modifico il dataset nello stesso modo di train.csv

# Cap Revolving Utilization
test_df$RevolvingUtilizationOfUnsecuredLines <- pmin(test_df$RevolvingUtilizationOfUnsecuredLines, 10)

# Create LogIncome (The model looks for 'LogIncome', not 'MonthlyIncome')
test_df$LogIncome <- log(test_df$MonthlyIncome + 1)

# Handle Missing Values
test_clean <- na.omit(test_df)

# Cap DebtRatio (if you did this in training)
test_clean$DebtRatio <- pmin(test_clean$DebtRatio, 10)

# Predict Probabilities
test_probabilities <- predict(mod.low, newdata = test_clean, type = "response")

# Apply the Threshold 
test_predictions <- ifelse(test_probabilities > soglia, 1, 0)

# View the distribution the predictions
table(test_predictions)









