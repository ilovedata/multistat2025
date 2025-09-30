

url <- "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data"
wine <- read.csv(url, header = FALSE)

# Add column names from UCI description
colnames(wine) <- c("Class","Alcohol","Malic_Acid","Ash","Alcalinity_Ash","Magnesium",
                    "Total_Phenols","Flavanoids","Nonflav_Phenols","Proanthocyanins",
                    "Color_Intensity","Hue","OD280_OD315","Proline")

df_1 <- wine %>% select(-Class)

pca_wine <- princomp(df_1, cor = TRUE)
summary(pca_wine, loadings = TRUE)



# Scree plot
plot(pca_wine, type = "l")


df_pca <- as.data.frame(pca_wine$scores)

# PC1–PC2 scatter
df_pca$Class <- factor(wine$Class)

library(ggplot2)
ggplot(df_pca, aes(Comp.1, Comp.2, color = Class)) +
  geom_point(size = 2) +
  theme_minimal() +
  labs(title = "PCA on Wine dataset")
