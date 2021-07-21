library(tidyverse)
library(EBImage)
library(uwot)
library(scales)
library(dynutils)
library(colorspace)


#' Prepare tiff for dimensionality reduction
#'
#' @param path A path to a tif/tiff file 
#' @param output_width Width of the minature output in px (A positive integer, defaults to 200) 
#'
#' @return A list containing the tissue_matrix, tissue_coords, tissue_px, pseudocount, and small_image
#' @export
#'
#' @examples
prepare_tiff = function(input, output_max_dim, remove_bg){
  
  print(paste("Reading image:", input))
  
  tiff_image <- EBImage::readImage(input)
  
  full_size_x <- tiff_image %>% dim() %>% .[1]
  full_size_y <- tiff_image %>% dim() %>% .[2]
  
  print(paste("Resizing image to", output_max_dim, "px in largest dimension"))
  
  small_image = if (full_size_x > full_size_y) {
    resize(tiff_image, w = output_max_dim)
  } else {
    resize(tiff_image, h = output_max_dim)
  }
  
  
  
  size_x <- small_image %>% dim() %>% .[1]
  size_y <- small_image %>% dim() %>% .[2]
  size_c <- small_image %>% dim() %>% .[3]
  n_px <- size_x * size_y
  
  print("Making datacube")
  
  image_matrix <- small_image %>% imageData() %>% apply(3, as.vector)
  mean_intensity <- rowMeans(image_matrix)
  

  pseudocount <- quantile(mean_intensity,0.01)
  #tissue_px <- which(mean_intensity > quantile(mean_intensity,0.35))
  
  print("Finding background with Otsu's threshold")
  
  sum_image <- small_image %>% apply(c(1,2), sum) 
  log_sum_image <- log(sum_image+quantile(sum_image,0.01))
  otsu_image <-log_sum_image > otsu(log_sum_image, range = range(log_sum_image))
  tissue_px <- which(otsu_image == TRUE)
  
  tissue_matrix <- if (remove_bg == TRUE) {image_matrix[tissue_px,]} else {image_matrix}
  all_coords <- which(!is.na(sum_image), arr.ind = TRUE) 
  colnames(all_coords) = c("x","y")
  tissue_coords <- if (remove_bg == TRUE) {all_coords[tissue_px,]} else {all_coords}
  print("Preparation complete!")
  return(
    list(
      tissue_matrix = tissue_matrix, 
      tissue_coords = tissue_coords, 
      tissue_px = tissue_px, 
      pseudocount = pseudocount, 
      small_image = small_image,
      otsu_image = otsu_image,
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
run_umap <- function(input){
  print("Starting dimensionality reduction")
  dim_red <- umap(input, verbose = TRUE, 
                  fast_sgd = TRUE, n_threads = parallel::detectCores()-1, 
                  metric = "correlation", n_components =3)
  print("Dimensionality reduction complete")
  return(dim_red)
}



plot_minature <- function(dim_red, coords){
  print("Generating colours")
  data <- coords %>% 
    as_tibble() %>%
    bind_cols(
      dim_red %>% as_tibble()
    ) %>%
    mutate(L = rescale(dynutils::scale_quantile(V3, 0.01), c(10,90)),
           a = (rescale(dynutils::scale_quantile(V1, 0.01),c(0,1)) - 0.5) *200,
           b = (rescale(dynutils::scale_quantile(V2, 0.01),c(0,1)) - 0.5) *200,
           LabHexUMAP = colorspace::LAB(L,a,b) %>% colorspace::hex(fixup = TRUE))
  
  print("Painting miniature")
  
  plot <- data %>%
    ggplot(aes(x, y, fill = LabHexUMAP)) +
    geom_tile() +
    scale_fill_identity() +
    scale_y_reverse() +
    coord_equal() +
    theme_void() +
    theme(plot.background = element_rect(fill = "black"))
  
  return(list(plot = plot, umap_colours = data))
}



paint_miniature <- function(input, output = tempfile(fileext = ".png"), output_max_dim = 512, remove_bg = TRUE, return_objects = FALSE){
  s1 <- prepare_tiff(input, output_max_dim = output_max_dim, remove_bg)
  s2 <- run_umap(s1$tissue_matrix)
  plot <- plot_minature(s2, s1$tissue_coords)
  print(paste("Saving miniature to", output))
  ggsave(output, plot$plot, width = s1$size_x, height = s1$size_y, units = "px")
  if (return_objects == TRUE) return(list(input = input, output= output, small_image = s1, umap = s2,plot = plot)) else return(output)
}