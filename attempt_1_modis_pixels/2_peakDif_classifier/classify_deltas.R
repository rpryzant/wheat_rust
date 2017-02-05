setwd('~/Dropbox/school/ermon_lab/2_peakDif_classifier')


data = read.csv("threshold_2_stemStripe.csv")
# shuffle rows
data = data[sample(nrow(data)),]


N = nrow(data)
train = data[1:(N - N/10),]
test = data[(N - N/10):N,]

model <- glm(diseased ~ sur_refl_b01+sur_refl_b02+sur_refl_b03+sur_refl_b04+sur_refl_b05+sur_refl_b06+sur_refl_b07,family=binomial(link='logit'),data=train)

summary(model)

anova(model, test="Chisq")

# everything's predicted as diseased!!
fitted.results <- predict(model,newdata=subset(test,select=c(2,3,4,5,6,7,8)),type='response')
fitted.results <- ifelse(fitted.results > 0.5,1,0)




