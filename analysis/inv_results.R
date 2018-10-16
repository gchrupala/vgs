library(jsonlite)
library(dplyr)
library(ggplot2)
data <- fromJSON("~/valid_runs.json")

data %>% mutate(s2i=factor(s2i)) %>%   ggplot(aes(x=`recall@10`, y=speaker_id, color=s2i)) + geom_point() + geom_smooth(method="lm") +
  theme(aspect.ratio=1, text=element_text(size=25)) +
  ylab("Accuracy of speaker ID") +
  xlab("Recall@10")
ggsave("~/repos/Perceptual-and-symbolic-correlates-of-spoken-language/spkrinv.png")
summary(lm(speaker_id ~ `recall@10`, data=data))
summary(lm(speaker_id ~  `recall@10` + s2i, data=data))
summary(lm(speaker_id ~ ., data=data))
