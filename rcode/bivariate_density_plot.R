# 필요한 패키지
install.packages("mvtnorm")
install.packages("plotly")

library(mvtnorm)
library(plotly)

# 예시 데이터 생성 (임의)
set.seed(123)
n <- 200
x <- rnorm(n, mean = 5, sd = 2)
y <- 0.5 * x + rnorm(n, mean = 3, sd = 1)
df <- data.frame(x = x, y = y)

# 평균과 공분산 추정
mu <- colMeans(df)       # 평균 벡터
sigma <- cov(df)         # 공분산 행렬

# x, y 그리드 생성
x_seq <- seq(min(df$x) - 1, max(df$x) + 1, length.out = 50)
y_seq <- seq(min(df$y) - 1, max(df$y) + 1, length.out = 50)
grid <- expand.grid(x = x_seq, y = y_seq)

# 이변량 정규분포 PDF 계산
grid$z <- dmvnorm(grid, mean = mu, sigma = sigma)

# z를 행렬로 변환 (surface plot용)
z_matrix <- matrix(grid$z, nrow = length(x_seq), ncol = length(y_seq))

# 3D Surface Plot
plot_ly(
  x = x_seq, 
  y = y_seq, 
  z = z_matrix
) %>% add_surface() %>%
  layout(
    title = "Bivariate Normal PDF (Surface)",
    scene = list(
      xaxis = list(title = "X"),
      yaxis = list(title = "Y"),
      zaxis = list(title = "Density")
    )
  )


plot_ly(
  x = x_seq,
  y = y_seq,
  z = z_matrix
) %>%
  add_contour(
    contours = list(
      coloring = "heatmap",
      showlabels = TRUE
    )
  ) %>%
  layout(
    title = "Bivariate Normal PDF (Contour)",
    xaxis = list(title = "X"),
    yaxis = list(title = "Y")
  )

