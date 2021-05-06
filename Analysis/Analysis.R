library(dplyr)
library(data.table)
library(tidyr)
library(rstatix)

zeta = 0.5
CI = .95
efficacy_df = './efficacy_score.csv'
output_file = './identified_treatments.csv'


efficacyEstimation<- function(csv_file_name)
{
  X = read.csv(file = csv_file_name , header = T , sep = ',' , stringsAsFactors = F, na.strings = '')
  X$treatment = sapply(X$treatment , simpleCap)
  X = X %>% mutate(trcr = paste(treatment,concentration,sep = '_'))
  meds = apply(as.matrix(unique(X$trcr)), 1,
               function(x) {
                 ee = sign_test(data = X[X$trcr == x[1],],Score ~ 1,
                                conf.level = CI,alternative = "two.sided",detailed = T)
                 data.frame(trcr = x[[1]],est = ee$estimate, LB = ee$conf.low , UB = ee$conf.high)
               }
  )
  meds_df = rbindlist(meds)
  X_merged = inner_join(x = X , y = meds_df , by = 'trcr')
  X_trcr = X_merged %>% group_by(treatment,concentration) %>% summarise(eff=median(UB),n=n())
  X_tr = X_trcr %>% group_by(treatment) %>% summarise(successful_con = length(eff[eff < zeta]), 
                                                success_score = x1 <- if (any(eff < zeta)) median(eff[eff < zeta]) else median(eff))
  X_tr = X_tr[order(X_tr$success_score,decreasing = F),]
  return (list(X_merged,X_trcr,X_tr))
}

X = efficacyEstimation(efficacy_df)
write.table(x = X[[3]],file = output_file,append = F , sep = ',' , col.names = T)

