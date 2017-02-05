library("ggplot2")


setwd("~/Dropbox/school/ermon_lab/data/wheat_rust")


data = read.csv("ET_RustSurvey_Published.csv")



######################################################################################
####################### PLOT 1: SEVERITY AGAINST GROWTH STAGE 
######################################################################################


#################### STEM RUST 
means = c(
  mean(subset(data, GrowthStageID == -9 & Severity > -1)$Severity),
  mean(subset(data, GrowthStageID == 1 & Severity > -1)$Severity),
  mean(subset(data, GrowthStageID == 2 & Severity > -1)$Severity),
  mean(subset(data, GrowthStageID == 3 & Severity > -1)$Severity),
  mean(subset(data, GrowthStageID == 4 & Severity > -1)$Severity),
  mean(subset(data, GrowthStageID == 5 & Severity > -1)$Severity),
  mean(subset(data, GrowthStageID == 6 & Severity > -1)$Severity),
  mean(subset(data, GrowthStageID == 7 & Severity > -1)$Severity)
)
sds = c(
  sd(subset(data, GrowthStageID == -9 & Severity > -1)$Severity),
  sd(subset(data, GrowthStageID == 1 & Severity > -1)$Severity),
  sd(subset(data, GrowthStageID == 2 & Severity > -1)$Severity),
  sd(subset(data, GrowthStageID == 3 & Severity > -1)$Severity),
  sd(subset(data, GrowthStageID == 4 & Severity > -1)$Severity),
  sd(subset(data, GrowthStageID == 5 & Severity > -1)$Severity),
  sd(subset(data, GrowthStageID == 6 & Severity > -1)$Severity),
  sd(subset(data, GrowthStageID == 7 & Severity > -1)$Severity)
)
ns = c(
  nrow(subset(data, GrowthStageID == -9 & Severity > -1)),
  nrow(subset(data, GrowthStageID == 1 & Severity > -1)),
  nrow(subset(data, GrowthStageID == 2 & Severity > -1)),
  nrow(subset(data, GrowthStageID == 3 & Severity > -1)),
  nrow(subset(data, GrowthStageID == 4 & Severity > -1)),
  nrow(subset(data, GrowthStageID == 5 & Severity > -1)),
  nrow(subset(data, GrowthStageID == 6 & Severity > -1)),
  nrow(subset(data, GrowthStageID == 7 & Severity > -1))
)
CIs = (1.96 * sds) / sqrt(ns)

################# LEAF RUST
means.1 = c(
  mean(subset(data, GrowthStageID == -9 & Severity.1 > -1)$Severity.1),
  mean(subset(data, GrowthStageID == 1 & Severity.1 > -1)$Severity.1),
  mean(subset(data, GrowthStageID == 2 & Severity.1 > -1)$Severity.1),
  mean(subset(data, GrowthStageID == 3 & Severity.1 > -1)$Severity.1),
  mean(subset(data, GrowthStageID == 4 & Severity.1 > -1)$Severity.1),
  mean(subset(data, GrowthStageID == 5 & Severity.1 > -1)$Severity.1),
  mean(subset(data, GrowthStageID == 6 & Severity.1 > -1)$Severity.1),
  mean(subset(data, GrowthStageID == 7 & Severity.1 > -1)$Severity.1)
)
sds.1 = c(
  sd(subset(data, GrowthStageID == -9 & Severity.1 > -1)$Severity.1),
  sd(subset(data, GrowthStageID == 1 & Severity.1 > -1)$Severity.1),
  sd(subset(data, GrowthStageID == 2 & Severity.1 > -1)$Severity.1),
  sd(subset(data, GrowthStageID == 3 & Severity.1 > -1)$Severity.1),
  sd(subset(data, GrowthStageID == 4 & Severity.1 > -1)$Severity.1),
  sd(subset(data, GrowthStageID == 5 & Severity.1 > -1)$Severity.1),
  sd(subset(data, GrowthStageID == 6 & Severity.1 > -1)$Severity.1),
  sd(subset(data, GrowthStageID == 7 & Severity.1 > -1)$Severity.1)
)
ns.1 = c(
  nrow(subset(data, GrowthStageID == -9 & Severity.1 > -1)),
  nrow(subset(data, GrowthStageID == 1 & Severity.1 > -1)),
  nrow(subset(data, GrowthStageID == 2 & Severity.1 > -1)),
  nrow(subset(data, GrowthStageID == 3 & Severity.1 > -1)),
  nrow(subset(data, GrowthStageID == 4 & Severity.1 > -1)),
  nrow(subset(data, GrowthStageID == 5 & Severity.1 > -1)),
  nrow(subset(data, GrowthStageID == 6 & Severity.1 > -1)),
  nrow(subset(data, GrowthStageID == 7 & Severity.1 > -1))
)
CIs.1 = (1.96 * sds.1) / sqrt(ns.1)


################# STRIPE RUST
means.2 = c(
  mean(subset(data, GrowthStageID == -9 & Severity.2 > -1)$Severity.2),
  mean(subset(data, GrowthStageID == 1 & Severity.2 > -1)$Severity.2),
  mean(subset(data, GrowthStageID == 2 & Severity.2 > -1)$Severity.2),
  mean(subset(data, GrowthStageID == 3 & Severity.2 > -1)$Severity.2),
  mean(subset(data, GrowthStageID == 4 & Severity.2 > -1)$Severity.2),
  mean(subset(data, GrowthStageID == 5 & Severity.2 > -1)$Severity.2),
  mean(subset(data, GrowthStageID == 6 & Severity.2 > -1)$Severity.2),
  mean(subset(data, GrowthStageID == 7 & Severity.2 > -1)$Severity.2)
)
sds.2 = c(
  sd(subset(data, GrowthStageID == -9 & Severity.2 > -1)$Severity.2),
  sd(subset(data, GrowthStageID == 1 & Severity.2 > -1)$Severity.2),
  sd(subset(data, GrowthStageID == 2 & Severity.2 > -1)$Severity.2),
  sd(subset(data, GrowthStageID == 3 & Severity.2 > -1)$Severity.2),
  sd(subset(data, GrowthStageID == 4 & Severity.2 > -1)$Severity.2),
  sd(subset(data, GrowthStageID == 5 & Severity.2 > -1)$Severity.2),
  sd(subset(data, GrowthStageID == 6 & Severity.2 > -1)$Severity.2),
  sd(subset(data, GrowthStageID == 7 & Severity.2 > -1)$Severity.2)
)
ns.2 = c(
  nrow(subset(data, GrowthStageID == -9 & Severity.2 > -1)),
  nrow(subset(data, GrowthStageID == 1 & Severity.2 > -1)),
  nrow(subset(data, GrowthStageID == 2 & Severity.2 > -1)),
  nrow(subset(data, GrowthStageID == 3 & Severity.2 > -1)),
  nrow(subset(data, GrowthStageID == 4 & Severity.2 > -1)),
  nrow(subset(data, GrowthStageID == 5 & Severity.2 > -1)),
  nrow(subset(data, GrowthStageID == 6 & Severity.2 > -1)),
  nrow(subset(data, GrowthStageID == 7 & Severity.2 > -1))
)
CIs.2 = (1.96 * sds.2) / sqrt(ns.2)


######################### PLOTTING
library(calibrate)
x = 1:8

# start with stem rust
plot(means~x, 
     cex=1.5, 
     ylim=c(0,1), 
     xlab='Growth Stage',
     xaxt = "n",
     ylab='Mean stem rust severity', 
     main='Stem Rust\nSeverity + Growth Stage (all years)',
     col='blue',
     pch=16,
     )
arrows(x,means+CIs,x,means-CIs,code=3,length=0.2,angle=90,col='blue')
axis(1, at=x, labels=c("N/A (missing)", "Tillering", "Boot", "Flowering", "Milk", "Dough", "Maturity", "Heading"))
textxy(x, means, ns, cex=1, offset=0.6)

# leaf rust
plot(means.1~x, 
       ylim=c(0,1),
       col='red',
       xaxt = "n",
       cex=1.5,
       main='Leaf Rust\nSeverity + Growth Stage (all years)',
       pch=16)
arrows(x,means.1+CIs.1,x,means.1-CIs.1,code=3,length=0.2,angle=90,col='red')
axis(1, at=x, labels=c("N/A (missing)", "Tillering", "Boot", "Flowering", "Milk", "Dough", "Maturity", "Heading"))
textxy(x, means.1, ns.1, cex=1, offset=0.6)

# stripe rust
plot(means.2~x, 
     ylim=c(0,1),
     col='purple',
     xaxt = "n",
     cex=1.5,
     main='Stripe Rust\nSeverity + Growth Stage (all years)',
     pch=16)
arrows(x,means.2+CIs.2,x,means.2-CIs.2,code=3,length=0.2,angle=90,col='purple')
axis(1, at=x, labels=c("N/A (missing)", "Tillering", "Boot", "Flowering", "Milk", "Dough", "Maturity", "Heading"))
textxy(x, means.2, ns.2, cex=1, offset=0.6)







######################################################################################
####################### PLOT 2: NUM OBSERVATIONS (and broken) BY YEAR
######################################################################################

counts = c(
  nrow(subset(data, ObsYear == "2007")),
  nrow(subset(data, ObsYear == "2008")),
  nrow(subset(data, ObsYear == "2009")),
  nrow(subset(data, ObsYear == "2010")),
  nrow(subset(data, ObsYear == "2011")),
  nrow(subset(data, ObsYear == "2012")),
  nrow(subset(data, ObsYear == "2013")),
  nrow(subset(data, ObsYear == "2014")),
  nrow(subset(data, ObsYear == "2015")),
  nrow(subset(data, ObsYear == "2016"))
)


broken.counts.1 = c(
  nrow(subset(data, (Latitude < 0 | Longitude < 0 | GrowthStageID == -9 | Severity == -9 | Severity.1 == -9 | Severity.2 == -9) & ObsYear == "2007")),
  nrow(subset(data, (Latitude < 0 | Longitude < 0 | GrowthStageID == -9 | Severity == -9 | Severity.1 == -9 | Severity.2 == -9) & ObsYear == "2008")),
  nrow(subset(data, (Latitude < 0 | Longitude < 0 | GrowthStageID == -9 | Severity == -9 | Severity.1 == -9 | Severity.2 == -9) & ObsYear == "2009")),
  nrow(subset(data, (Latitude < 0 | Longitude < 0 | GrowthStageID == -9 | Severity == -9 | Severity.1 == -9 | Severity.2 == -9) & ObsYear == "2010")),
  nrow(subset(data, (Latitude < 0 | Longitude < 0 | GrowthStageID == -9 | Severity == -9 | Severity.1 == -9 | Severity.2 == -9) & ObsYear == "2011")),
  nrow(subset(data, (Latitude < 0 | Longitude < 0 | GrowthStageID == -9 | Severity == -9 | Severity.1 == -9 | Severity.2 == -9) & ObsYear == "2012")),
  nrow(subset(data, (Latitude < 0 | Longitude < 0 | GrowthStageID == -9 | Severity == -9 | Severity.1 == -9 | Severity.2 == -9) & ObsYear == "2013")),
  nrow(subset(data, (Latitude < 0 | Longitude < 0 | GrowthStageID == -9 | Severity == -9 | Severity.1 == -9 | Severity.2 == -9) & ObsYear == "2014")),
  nrow(subset(data, (Latitude < 0 | Longitude < 0 | GrowthStageID == -9 | Severity == -9 | Severity.1 == -9 | Severity.2 == -9) & ObsYear == "2015")),
  nrow(subset(data, (Latitude < 0 | Longitude < 0 | GrowthStageID == -9 | Severity == -9 | Severity.1 == -9 | Severity.2 == -9) & ObsYear == "2016"))
)


barplot(counts,
        col="grey",
        names.arg=c("2007", "2008", "2009", "2010", "2011", "2012", "2013", "2014", "2015", "2016"),
        ylab="count",
        xlab="year",
        main="missing ANYTHING (growth stage, loc, any severity)",
        ylim=c(0, 1500)
)
par(new=TRUE)
barplot(broken.counts.1, col="red", axes=FALSE, ylim=c(0, 1500))
#legend("topleft", leg="Missing growth \n stage OR severity \n OR location", fill="red", box.lwd = 0)





broken.counts.2 = c(
  nrow(subset(data, (Latitude < 0 | Longitude < 0 | Severity == -9 | Severity.1 == -9 | Severity.2 == -9) & ObsYear == "2007")),
  nrow(subset(data, (Latitude < 0 | Longitude < 0 | Severity == -9 | Severity.1 == -9 | Severity.2 == -9) & ObsYear == "2008")),
  nrow(subset(data, (Latitude < 0 | Longitude < 0 | Severity == -9 | Severity.1 == -9 | Severity.2 == -9) & ObsYear == "2009")),
  nrow(subset(data, (Latitude < 0 | Longitude < 0 | Severity == -9 | Severity.1 == -9 | Severity.2 == -9) & ObsYear == "2010")),
  nrow(subset(data, (Latitude < 0 | Longitude < 0 | Severity == -9 | Severity.1 == -9 | Severity.2 == -9) & ObsYear == "2011")),
  nrow(subset(data, (Latitude < 0 | Longitude < 0 | Severity == -9 | Severity.1 == -9 | Severity.2 == -9) & ObsYear == "2012")),
  nrow(subset(data, (Latitude < 0 | Longitude < 0 | Severity == -9 | Severity.1 == -9 | Severity.2 == -9) & ObsYear == "2013")),
  nrow(subset(data, (Latitude < 0 | Longitude < 0 | Severity == -9 | Severity.1 == -9 | Severity.2 == -9) & ObsYear == "2014")),
  nrow(subset(data, (Latitude < 0 | Longitude < 0 | Severity == -9 | Severity.1 == -9 | Severity.2 == -9) & ObsYear == "2015")),
  nrow(subset(data, (Latitude < 0 | Longitude < 0 | Severity == -9 | Severity.1 == -9 | Severity.2 == -9) & ObsYear == "2016"))
)

barplot(counts,
        col="grey",
        names.arg=c("2007", "2008", "2009", "2010", "2011", "2012", "2013", "2014", "2015", "2016"),
        ylab="count",
        xlab="year",
        main="missing LOC or ANY severity",
        ylim=c(0, 1500)
)
par(new=TRUE)
barplot(broken.counts.2, col="red", axes=FALSE, ylim=c(0, 1500))
#legend("topleft", leg="Missing loc or stem/stripe rust", fill="red", box.lwd = 0)






broken.counts.3 = c(
  nrow(subset(data, (Latitude < 0 | Longitude < 0 | Severity == -9 | Severity.2 == -9) & ObsYear == "2007")),
  nrow(subset(data, (Latitude < 0 | Longitude < 0 | Severity == -9 | Severity.2 == -9) & ObsYear == "2008")),
  nrow(subset(data, (Latitude < 0 | Longitude < 0 | Severity == -9 | Severity.2 == -9) & ObsYear == "2009")),
  nrow(subset(data, (Latitude < 0 | Longitude < 0 | Severity == -9 | Severity.2 == -9) & ObsYear == "2010")),
  nrow(subset(data, (Latitude < 0 | Longitude < 0 | Severity == -9 | Severity.2 == -9) & ObsYear == "2011")),
  nrow(subset(data, (Latitude < 0 | Longitude < 0 | Severity == -9 | Severity.2 == -9) & ObsYear == "2012")),
  nrow(subset(data, (Latitude < 0 | Longitude < 0 | Severity == -9 | Severity.2 == -9) & ObsYear == "2013")),
  nrow(subset(data, (Latitude < 0 | Longitude < 0 | Severity == -9 | Severity.2 == -9) & ObsYear == "2014")),
  nrow(subset(data, (Latitude < 0 | Longitude < 0 | Severity == -9 | Severity.2 == -9) & ObsYear == "2015")),
  nrow(subset(data, (Latitude < 0 | Longitude < 0 | Severity == -9 | Severity.2 == -9) & ObsYear == "2016"))
)

barplot(counts,
        col="grey",
        names.arg=c("2007", "2008", "2009", "2010", "2011", "2012", "2013", "2014", "2015", "2016"),
        ylab="count",
        xlab="year",
        main="missing LOC or STEM/STRIPE severity",
        ylim=c(0, 1500)
)
par(new=TRUE)
barplot(broken.counts.3, col="red", axes=FALSE, ylim=c(0, 1500))
#legend("topleft", leg="Missing loc or stem/stripe rust", fill="red", box.lwd = 0)






broken.counts.4 = c(
  nrow(subset(data, (Latitude < 0 | Longitude < 0) & ObsYear == "2007")),
  nrow(subset(data, (Latitude < 0 | Longitude < 0) & ObsYear == "2008")),
  nrow(subset(data, (Latitude < 0 | Longitude < 0) & ObsYear == "2009")),
  nrow(subset(data, (Latitude < 0 | Longitude < 0) & ObsYear == "2010")),
  nrow(subset(data, (Latitude < 0 | Longitude < 0) & ObsYear == "2011")),
  nrow(subset(data, (Latitude < 0 | Longitude < 0) & ObsYear == "2012")),
  nrow(subset(data, (Latitude < 0 | Longitude < 0) & ObsYear == "2013")),
  nrow(subset(data, (Latitude < 0 | Longitude < 0) & ObsYear == "2014")),
  nrow(subset(data, (Latitude < 0 | Longitude < 0) & ObsYear == "2015")),
  nrow(subset(data, (Latitude < 0 | Longitude < 0) & ObsYear == "2016"))
)

barplot(counts,
        col="grey",
        names.arg=c("2007", "2008", "2009", "2010", "2011", "2012", "2013", "2014", "2015", "2016"),
        ylab="count",
        xlab="year",
        main="Missing LOC",
        ylim=c(0, 1500)
)
par(new=TRUE)
barplot(broken.counts.4, col="red", axes=FALSE, ylim=c(0, 1500))
#legend("topleft", leg="Missing location", fill="red", box.lwd = 0)





broken.counts.5 = c(
  nrow(subset(data, (Severity == -9 | Severity.2 == -9) & ObsYear == "2007")),
  nrow(subset(data, (Severity == -9 | Severity.2 == -9) & ObsYear == "2008")),
  nrow(subset(data, (Severity == -9 | Severity.2 == -9) & ObsYear == "2009")),
  nrow(subset(data, (Severity == -9 | Severity.2 == -9) & ObsYear == "2010")),
  nrow(subset(data, (Severity == -9 | Severity.2 == -9) & ObsYear == "2011")),
  nrow(subset(data, (Severity == -9 | Severity.2 == -9) & ObsYear == "2012")),
  nrow(subset(data, (Severity == -9 | Severity.2 == -9) & ObsYear == "2013")),
  nrow(subset(data, (Severity == -9 | Severity.2 == -9) & ObsYear == "2014")),
  nrow(subset(data, (Severity == -9 | Severity.2 == -9) & ObsYear == "2015")),
  nrow(subset(data, (Severity == -9 | Severity.2 == -9) & ObsYear == "2016"))
)

barplot(counts,
        col="grey",
        names.arg=c("2007", "2008", "2009", "2010", "2011", "2012", "2013", "2014", "2015", "2016"),
        ylab="count",
        xlab="year",
        main="missing STRIPE or STEM severity",
        ylim=c(0, 1500)
)
par(new=TRUE)
barplot(broken.counts.5, col="red", axes=FALSE, ylim=c(0, 1500))
#legend("topleft", leg="Missing loc or stem/stripe rust", fill="red", box.lwd = 0)





broken.counts.6 = c(
  nrow(subset(data, (Severity.2 == -9) & ObsYear == "2007")),
  nrow(subset(data, (Severity.2 == -9) & ObsYear == "2008")),
  nrow(subset(data, (Severity.2 == -9) & ObsYear == "2009")),
  nrow(subset(data, (Severity.2 == -9) & ObsYear == "2010")),
  nrow(subset(data, (Severity.2 == -9) & ObsYear == "2011")),
  nrow(subset(data, (Severity.2 == -9) & ObsYear == "2012")),
  nrow(subset(data, (Severity.2 == -9) & ObsYear == "2013")),
  nrow(subset(data, (Severity.2 == -9) & ObsYear == "2014")),
  nrow(subset(data, (Severity.2 == -9) & ObsYear == "2015")),
  nrow(subset(data, (Severity.2 == -9) & ObsYear == "2016"))
)

barplot(counts,
        col="grey",
        names.arg=c("2007", "2008", "2009", "2010", "2011", "2012", "2013", "2014", "2015", "2016"),
        ylab="count",
        xlab="year",
        main="missing LEAF severity",
        ylim=c(0, 1500)
)
par(new=TRUE)
barplot(broken.counts.6, col="red", axes=FALSE, ylim=c(0, 1500))
#legend("topleft", leg="Missing loc or stem/stripe rust", fill="red", box.lwd = 0)









######################################################################################
####################### PLOT 3: SEVERITY BY YEAR
######################################################################################


data.2007 = subset(data, ObsYear == "2007")
data.2008 = subset(data, ObsYear == "2008")
data.2009 = subset(data, ObsYear == "2009")
data.2010 = subset(data, ObsYear == "2010")
data.2011 = subset(data, ObsYear == "2011")
data.2012 = subset(data, ObsYear == "2012")
data.2013 = subset(data, ObsYear == "2013")
data.2014 = subset(data, ObsYear == "2014")
data.2015 = subset(data, ObsYear == "2015")
data.2016 = subset(data, ObsYear == "2016")


hist(data.2007$Severity)
hist(data.2008$Severity)
hist(data.2009$Severity)
hist(data.2010$Severity)
hist(data.2011$Severity)
hist(data.2012$Severity)
hist(data.2013$Severity)
hist(data.2014$Severity)
hist(data.2015$Severity)
hist(data.2016$Severity)

nrow(subset(data, Severity == 0 & Severity.1 == 0 & Severity.2 == 0))      # rust-free   2220
nrow(subset(data, Severity > 0 | Severity.1 > 0 | Severity.2 > 0))         # some rust   5931











#####################################
# # # # # #  GROWTH STAGE STUFF
#####################################

setwd("~/Dropbox/school/ermon_lab/data/")
df = read.csv("RAW_LABELS.csv")


na = subset(df, GrowthStageID == -9)
tillering = subset(df, GrowthStageID == 1)
boot = subset(df, GrowthStageID == 2)
heading = subset(df, GrowthStageID == 7)
flowering = subset(df, GrowthStageID == 3)
milk = subset(df, GrowthStageID == 4)
dough = subset(df, GrowthStageID == 5)
mature = subset(df, GrowthStageID == 6)






