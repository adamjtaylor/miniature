


#' Prepare tiff for dimensionality reduction
#'
#' @param path A path to a tif/tiff file 
#' @param output_width Width of the minature output in px (A positive integer, defaults to 200) 
#'
#' @return A list containing the tissue_matrix, tissue_coords, tissue_px, pseudocount, and small_image
#' @export
#'
#' @examples
prepare_tiff = function(path, output_max_dim, remove_bg){
  
  tiff_image <- EBImage::readImage(path)
  
  full_size_x <- tiff_image %>% dim() %>% .[1]
  full_size_y <- tiff_image %>% dim() %>% .[2]
  
  small_image = if (full_size_x > full_size_y) {
    resize(tiff_image, w = output_max_dim)
  } else {
    resize(tiff_image, h = output_max_dim)
  }
  
  
  
  size_x <- small_image %>% dim() %>% .[1]
  size_y <- small_image %>% dim() %>% .[2]
  size_c <- small_image %>% dim() %>% .[3]
  n_px <- size_x * size_y
  
  image_matrix <- small_image %>% imageData() %>% apply(3, as.vector)
  mean_intensity <- rowMeans(image_matrix)
  

  pseudocount <- quantile(mean_intensity,0.01)
  #tissue_px <- which(mean_intensity > quantile(mean_intensity,0.35))
  
  sum_image <- small_image %>% apply(c(1,2), sum) 
  log_sum_image <- log(sum_image+quantile(sum_image,0.01))
  otsu_image <-log_sum_image > otsu(log_sum_image, range = range(log_sum_image))
  tissue_px <- which(otsu_image == TRUE)
  
  tissue_matrix <- if(remove_bg == TRUE) image_matrix[tissue_px,] else image_matrix
  all_coords <- which(!is.na(otsu_image), arr.ind = TRUE) 
  tissue_coords <- if (remove_bg == TRUE) all_coords[tissue_px,] else all_coords
  return(
    list(
      tissue_matrix = tissue_matrix, 
      tissue_coords = tissue_coords, 
      tissue_px = tissue_px, 
      pseudocount = pseudocount, 
      small_image = small_image,
      mean_intensity = mean_intensity,
      size_x = size_x,
      size_y = size_y
    )
  )
}


#' Perform UMAP dim red
#'
#' @param input
#' @param pseudocount
#'
#' @return
#' @export
#'
#' @examples
run_umap <- function(input, pseudocount, n_neighbours, metric, a, b, min_dist, spread){
  #out <- log(input+pseudocount) #%>% apply(2, rescale)
  out <- input
  dim_red <- umap(out, verbose = TRUE, fast_sgd = TRUE, n_threads = parallel::detectCores()-1, metric = "correlation", n_components =3)
  return(dim_red)
}



plot_minature <- function(dim_red, coords, output_name){
  data <- coords %>% 
    as_tibble() %>%
    bind_cols(
      dim_red %>% as_tibble()
    ) %>%
    mutate(L = rescale(dynutils::scale_quantile(V3, 0.01), c(10,90)),
           a = (rescale(dynutils::scale_quantile(V1, 0.01),c(0,1)) - 0.5) *200,
           b = (rescale(dynutils::scale_quantile(V2, 0.01),c(0,1)) - 0.5) *200,
           LabHexUMAP = colorspace::LAB(L,a,b) %>% colorspace::hex(fixup = TRUE))
  
  plot <- data %>%
    ggplot(aes(row, col, fill = LabHexUMAP)) +
    geom_tile() +
    scale_fill_identity() +
    scale_y_reverse() +
    coord_equal() +
    theme_void() +
    theme(plot.background = element_rect(fill = "black"))
  
  return(plot)
}



paint_miniature <- function(path, output_name, output_max_dim = 720, remove_bg = TRUE){
  s1 <- prepare_tiff(path, output_max_dim = output_max_dim, remove_bg)
  s2 <- run_umap(s1$tissue_matrix, s1$pseudocount)
  plot <- plot_minature(s2, s1$tissue_coords)
  
  ggsave(output_name, plot, width = s1$size_x, height = s1$size_y, units = "px")
}