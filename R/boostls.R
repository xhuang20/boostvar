#' @title Least-squares boosting method for cross-section regression
#'
#' @description This function computes the estimates, standard errors, and p-values
#'   in a cross-section regression.
#'
#' @param y A n by 1 vector of dependent variable.
#' 
#' @param x An n by p matrix of variables with n observations and p variables.
#' 
#' @param bstop The number of boosting steps. Defaults to 50.
#'   
#' @param nu The learning rate. Defaults to 0.1.
#' 
#' @details The return value of this function always includes an intercept estimate.
#' 
#' @return \code{boostls} returns a list with the following components:
#'   \item{y}{The n by 1 vector of the dependent variable.}
#'   \item{x}{The n by p matrix of the independent variables.}
#'   \item{n}{The sample size.}
#'   \item{p}{The number of variables in the linear regression model.}
#'   \item{bstop}{The number of boosting steps.}
#'   \item{beta}{The p by bstop matrix of parameter estimates in all boosting steps.}
#'   \item{beta0}{The bstop by 1 vector of intercept estimates in all boosting steps.}
#'   \item{se}{The p by bstop matrix of standard errors in all boosting steps.}
#'   \item{pval}{The p by bstop matrix of p-values for all parameter estimates in all
#'         boosting steps. If a p-value is equal to 1, it means the corresponding parameter has
#'         not been updated up to that boosting step.}
#'   \item{step_mat}{A p by bstop matrix that records which variable is updated at each
#'         boosting step.}
#'   \item{aic}{A sequence of AIC values of length \code{bstop}.}
#'
#' @usage boostls(y, x,bstop = 50, nu = 0.1)
#' 
#' @export
boostls <- function(y, x,
                    bstop = 50,
                    nu = 0.1) {
  
  # Check column names
  if(is.null(colnames(x))) {
    var.names = NULL
  } else {
    var.names = colnames(x)
  }
  
  n = dim(x)[1]
  p = dim(x)[2]
  
  xdm = scale(x, center = TRUE, scale = FALSE)
  ydm = scale(y, center = TRUE, scale = FALSE)
  
  result = boostlscpp(y = as.matrix(y),
                      x = as.matrix(x),
                      bstop = bstop,
                      nu = nu)
  
  row.names(result$beta_mat) = var.names
  row.names(result$se_mat)   = var.names
  row.names(result$p_mat)    = var.names
  row.names(result$step_mat) = var.names
  
  obj = list()

  obj$y = y  
  obj$x = x  
  obj$n = n
  obj$p = p
  obj$bstop = bstop
  obj$beta  = result$beta_mat
  obj$beta0 = result$beta0_vec
  obj$se    = result$se_mat
  obj$pval  = result$p_mat
  obj$step_mat = result$step_mat
  obj$aic = result$aic
  
  return(obj)
  
}