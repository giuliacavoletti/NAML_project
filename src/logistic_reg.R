
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


#split the data between test and validate 

set.seed(123) 
train_index <- sample(1:nrow(lw_clean), 0.8 * nrow(lw_clean))
train_set <- lw_clean[train_index, ]  # The 80% used to teach the model
val_set   <- lw_clean[-train_index, ] # The 20% used to test the model

# Fit the model using the CLEAN train data

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
              data = train_set)


summary(mod.low)

# Check Goodness of Fit (GOF)
hoslem.test(mod.low$y, fitted(mod.low), g = 6)


#testiamo sul validate set
val_probs <- predict(mod.low, newdata = val_set, type = "response")

#Calculate the AUC (Area Under Curve)
# 1.0 is perfect, 0.5 is guessing
roc_val <- roc(val_set$SeriousDlqin2yrs, val_probs)

# Re-run the plot with visual enhancements
plot(roc_val, 
     col = "#0073C2FF",             
     lwd = 3,                      
     main = "ROC Curve (Validation Set)", 
     print.auc = TRUE,              # Print the AUC score directly on the plot
     auc.polygon = TRUE,            
     auc.polygon.col = "#d9e9f7",   
     grid = TRUE,                   
     print.thres = "best",          
     print.thres.pch = 19,         
     print.thres.col = "red"       
)

# Calculate Confusion Matrix using the best threshold
best_coords <- coords(roc_val, "best", ret = "threshold")
best_thresh <- best_coords$threshold

val_preds <- ifelse(val_probs > best_thresh, 1, 0)
tab2=table(Actual = val_set$SeriousDlqin2yrs, Predicted = val_preds)
round( sum( diag( tab2 ) ) / sum( tab2 ), 2 ) #percentuale di casi classificati nel modo corretto 

#ora applico sul dataset "test" fornito da kaggle

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
test_predictions <- ifelse(test_probabilities > best_thresh, 1, 0)

# View the distribution the predictions
table(test_predictions)









