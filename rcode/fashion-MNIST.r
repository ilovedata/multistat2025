# ---- Packages ----
install.packages(c("keras3","ggplot2","dplyr","tidyr","forcats","gridExtra"))
library(keras3)
library(ggplot2)
library(dplyr)
library(tidyr)
library(forcats)
library(gridExtra)

# ---- 1) Load Fashion-MNIST ----
# This works even if you haven't installed TensorFlow yet; it just downloads the data.
fmnist <- dataset_fashion_mnist()
x_train <- fmnist$train$x  # array: 60000 x 28 x 28
y_train <- fmnist$train$y  # labels 0..9
x_test  <- fmnist$test$x
y_test  <- fmnist$test$y

# Class names (from Zalando paper)
classes <- c("T-shirt/top","Trouser","Pullover","Dress","Coat",
             "Sandal","Shirt","Sneaker","Bag","Ankle boot")

# ---- 2) Build feature matrix (images -> rows; pixels -> columns) ----
# Combine train & test (optional; feel free to use train only)
x_all <- abind::abind(x_train, x_test, along = 1) # needs abind
y_all <- c(y_train, y_test)

# If abind isn't installed, use this fallback:
# x_all <- array(c(x_train, x_test), dim = c(dim(x_train)[1] + dim(x_test)[1], 28, 28))

n <- dim(x_all)[1]
p <- 28 * 28

# reshape to n x 784, scale to [0,1]
X <- matrix(as.numeric(x_all), nrow = n, ncol = p)
X <- X / 255

labels <- factor(y_all, levels = 0:9, labels = classes)

# ---- (Optional) subsample to speed up PCA in class ----
set.seed(42)
n_sub <- 10000  # try 5000–15000 for a quick demo
idx <- sample.int(n, n_sub)
X_sub <- X[idx, ]
labels_sub <- labels[idx]

# ---- 3) PCA (center & scale features). Use rank. to compute first k PCs quickly. ----
k <- 50  # number of principal components to keep
pca <- prcomp(X_sub, center = TRUE, scale. = TRUE, rank. = k)

# ---- 4) Variance explained plot ----
var_explained <- pca$sdev^2
pve <- var_explained / sum(var_explained)
df_pve <- tibble(
  PC = factor(seq_along(pve)),
  PVE = pve,
  CumPVE = cumsum(pve)
)

g1 <- ggplot(df_pve, aes(as.integer(PC), PVE)) +
  geom_col() +
  geom_line(aes(y = CumPVE)) +
  geom_point(aes(y = CumPVE)) +
  labs(x = "Principal component", y = "Proportion of variance explained",
       title = "PCA on Fashion-MNIST: variance explained",
       subtitle = paste0("Using ", n_sub, " images · ", k, " PCs")) +
  theme_minimal()

print(g1)

# ---- 5) PC1–PC2 scatter by class ----
scores <- as.data.frame(pca$x[, 1:2])
scores$label <- labels_sub

g2 <- ggplot(scores, aes(PC1, PC2, color = label)) +
  geom_point(alpha = 0.5, size = 1) +
  guides(color = guide_legend(override.aes = list(size = 3, alpha = 1))) +
  labs(title = "Fashion-MNIST in PC1–PC2", color = NULL) +
  theme_minimal()

print(g2)

# ---- 6) “Eigen-clothes”: visualize first few PCs as 28×28 images ----
viz_pc_images <- function(pca_obj, pcs = 1:16, nrow = 4) {
  loadings <- pca_obj$rotation[, pcs, drop = FALSE]  # 784 x |pcs|
  imgs <- lapply(seq_along(pcs), function(i) {
    m <- matrix(loadings[, i], nrow = 28, byrow = TRUE)
    # center around 0 for diverging look; use base image with ggplot raster
    df <- expand.grid(x = 1:28, y = 1:28)
    df$val <- as.vector(m)
    ggplot(df, aes(x, y, fill = val)) +
      geom_raster() +
      scale_y_reverse() +
      labs(title = paste0("PC", pcs[i])) +
      theme_void() +
      theme(legend.position = "none")
  })
  do.call(gridExtra::grid.arrange, c(imgs, nrow = nrow))
}

viz_pc_images(pca, pcs = 1:16, nrow = 4)

# ---- 7) Reconstruction demo: compress & reconstruct with K PCs ----
reconstruct_with_k <- function(pca_obj, X_scaled, K, center, scale) {
  # pca_obj$x = scores for rank.=k; but if we want different K<=k, truncate
  scoresK <- pca_obj$x[, 1:K, drop = FALSE]
  VK <- pca_obj$rotation[, 1:K, drop = FALSE]
  Xk <- scoresK %*% t(VK)  # in scaled, centered space
  # undo scaling/centering that prcomp applied
  if (!is.null(scale)) Xk <- sweep(Xk, 2, scale, `*`)
  if (!is.null(center)) Xk <- sweep(Xk, 2, center, `+`)
  Xk
}

# Note: prcomp(X, center=TRUE, scale.=TRUE) stores 'center' and 'scale'
X_recon_10  <- reconstruct_with_k(pca, X_sub, 10,  pca$center, pca$scale)
X_recon_50  <- reconstruct_with_k(pca, X_sub, 50,  pca$center, pca$scale)

# ---- Helper to plot original vs reconstructed for a few images ----
plot_triplet <- function(X_orig, X10, X50, idx_show) {
  one <- function(v, ttl) {
    m <- matrix(v, nrow = 28, byrow = TRUE)
    df <- expand.grid(x = 1:28, y = 1:28)
    df$val <- as.vector(m)
    ggplot(df, aes(x, y, fill = val)) +
      geom_raster() +
      scale_y_reverse() +
      theme_void() +
      theme(legend.position = "none") +
      labs(title = ttl)
  }
  g_list <- list(
    one(X_orig[idx_show, ], "Original"),
    one(X10[idx_show, ],    "Recon (10 PCs)"),
    one(X50[idx_show, ],    "Recon (50 PCs)")
  )
  gridExtra::grid.arrange(grobs = g_list, nrow = 1)
}

# Show three random examples
set.seed(1)
show_ids <- sample.int(n_sub, 3)
for (i in show_ids) {
  plot_triplet(X_sub, X_recon_10, X_recon_50, i)
}

# ---- 8) (Optional) Quick downstream ML demo: kNN on first d PCs ----
# install.packages("class")
library(class)
d <- 30
train_ids <- 1:round(0.8 * n_sub)
test_ids  <- setdiff(1:n_sub, train_ids)

Xpc <- as.data.frame(pca$x[, 1:d])
pred <- knn(train = Xpc[train_ids, ], test = Xpc[test_ids, ],
            cl = labels_sub[train_ids], k = 5)
acc <- mean(pred == labels_sub[test_ids])
message(sprintf("kNN on %d PCs, k=5 → accuracy = %.3f", d, acc))
