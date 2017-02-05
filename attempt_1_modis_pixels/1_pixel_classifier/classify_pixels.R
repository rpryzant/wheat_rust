setwd('~/Dropbox/school/ermon_lab/1_pixel_classifier')


data = read.csv("pixels.csv")


train = data[1:7000,]
test = data[7001:7061,]

model <- glm(diseased ~ sr_b01+sr_b02+sr_b03+sr_b04+sr_b05+sr_b06,family=binomial(link='logit'),data=train)

summary(model)

anova(model, test="Chisq")

# everything's predicted as diseased!!
fitted.results <- predict(model,newdata=subset(test,select=c(9,10,11,12,13,14)),type='response')
fitted.results <- ifelse(fitted.results > 0.5,1,0)



#misClasificError <- mean(fitted.results != test$diseased)
#print(paste('Accuracy',1-misClasificError))

