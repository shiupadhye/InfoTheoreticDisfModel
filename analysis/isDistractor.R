library(lme4)
library(MASS)
library(stats)
library(ppcor)
library(corrplot)
library(reshape2)
library(lmtest)
library(tidyverse)
library(GLMMadaptive)
library(effectsize)
library(MuMIn)
library(latex2exp)
library(phonics)
library(stringdist)
library(qwraps2)

get_lower_tri<-function(cormat){
  cormat[upper.tri(cormat)] <- NA
  return(cormat)
}

filename = "SWBD_scores_combined.csv"
scores <- read_csv(filename)
head(scores)


scores <- scores %>%
  mutate(log_pReprn = log2(pReprn),
         log_pRepair= log2(pRepair),
         log_pReprnFw = log2(pReprnFw),
         log_pRepairFw = log2(pRepairFw),
         log_pReprnBw = log2(pReprnBw),
         log_pRepairBw = log2(pRepairBw),
         log_pReprnMasked = log2(pReprnMasked),
         log_pRepairMasked = log2(pRepairMasked))


scores <- scores %>%
  mutate(reprn_pmiP = log_pReprnFw - log_pReprn,
         repair_pmiP = log_pRepairFw - log_pRepair,
         reprn_pmiF = log_pReprnBw - log_pReprn,
         repair_pmiP = log_pRepairBw - log_pRepair,
         reprn_pmiFP = log_pReprnMasked - log_pReprnFw,
         repair_pmiFP = log_pRepairMasked - log_pRepairFw)

scores <- scores %>%
  mutate(sim_score = phonREL_dist * semREL_dist)

# single-predictor models
# baseline
m0 <- mixed_model(fixed = isDistractor ~ 1, random ~ 1|tID,data=scores,family=binomial)
# fixed effect: log unigram probability
m1.1 <- mixed_model(fixed = isDistractor ~ log_pReprn, random ~ 1|tID,data=scores,family=binomial)
summary(m1.1)
# fixed effect: log forward probability
m1.2 <- mixed_model(fixed = isDistractor ~ log_pReprnFw, random ~ 1|tID,data=scores,family=binomial)
summary(m1.2)
# fixed effect: log backward probability
m1.3 <- mixed_model(fixed = isDistractor ~ log_pReprnBw, random ~ 1|tID,data=scores,family=binomial)
summary(m1.3)
# fixed effect: pmiFP
m1.4 <- mixed_model(fixed = isDistractor ~ reprn_pmiFP, random ~ 1|tID,data=scores,family=binomial)
summary(m1.4)
# fixed effect: semantic and phonological distance
m1.5 <- mixed_model(fixed = isDistractor ~ semREL_dist + phonREL_dist, random ~ 1|tID,data=scores,family=binomial)
summary(m1.5)

# log likelihood ratio test
lrtest(m0,m1.1,m1.2,m1.3,m1.4,m1.5)


# incremental isDistractor model
m0 <- mixed_model(fixed = isDistractor ~ 1, random ~ 1|tID,data=scores,family=binomial)
# fixed effect: log unigram probability
m1.1 <- mixed_model(fixed = isDistractor ~ log_pReprn, random ~ 1|tID,data=scores,family=binomial)
summary(m1.1)
# fixed effects: log unigram probability, distances
m2 <- mixed_model(fixed = isDistractor ~ log_pReprn + semREL_dist + phonREL_dist, random ~ 1|tID,data=scores,family=binomial)
summary(m2)
# fixed effects: log unigram probability, distances, log forward probability
m3 <- mixed_model(fixed = isDistractor ~ log_pReprn  + phonREL_dist + semREL_dist + log_pReprnFw, random ~ 1|tID,data=scores,family=binomial)
summary(m3)
# fixed effects: log unigram probability, distances, log forward probability, pmiFP
m4 <- mixed_model(fixed = isDistractor ~ log_pReprn + log_pReprnFw + phonREL_dist + semREL_dist + reprn_pmiFP, random ~ 1|tID,data=scores,family=binomial)
summary(m4)
# fixed effects: log unigram probability, distances, log forward probability, log backward probability
m5 <- mixed_model(fixed = isDistractor ~ log_pReprn + phonREL_dist + semREL_dist + log_pReprnFw + log_pReprnBw, random ~ 1|tID,data=scores,family=binomial)
summary(m5)
# fixed effects: log unigram probability, distances, log forward probability, log backward probability, pmiFP
m6 <- mixed_model(fixed = isDistractor ~ log_pReprn + log_pReprnFw + phonREL_dist + semREL_dist + reprn_pmiFP + log_pReprnBw, random ~ 1|tID,data=scores,family=binomial)
summary(m6)

# log likelihood ratio test
lrtest(m0,m1.1,m2,m3,m4,m5,m6)


