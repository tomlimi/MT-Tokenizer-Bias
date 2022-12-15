library(bnlearn)

data = read.csv("~/Desktop/MT-tokenizer/data/data_analysis_opus100_es_new.csv")

x = data$"F1"
y = data$"Tokens"
# y <- as.character(y)
y <- as.numeric(y)
z = data$"log.Frequency"

x <- factor(x, order=TRUE)
y <-factor(y, order=TRUE)
z <- factor(round(z, digits=0), order=TRUE)


ci.test(x, z, test='jt')
ci.test(x, y, test='jt')
ci.test(x, y, z, test='jt')
ci.test(x, z, y, test='jt')

