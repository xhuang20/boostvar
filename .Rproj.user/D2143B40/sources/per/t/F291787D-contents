#' @title p-Values for least-squares boosting in high-dimensional VARs
#'
#' @description This function computes the p-value for the two LS-Boost algorithms
#'   discussed in Huang (2022).
#'
#' @param datay An n by d matrix of variables with n observations and d variables.
#'   n is the number of time periods in the data, where the first row gives the
#'   first observation and the last row includes the most recent observation.
#'   
#' @param p A positive integer for the lag of VAR. Defaults to 1.
#' 
#' @param bstop The number of boosting steps. Defaults to 50.
#' 
#' @param n_lambda The number of equally spaced boosting steps on the sequence from \code{1}
#'   to \code{bstop}. This number needs to be smaller than or equal to \code{bstop}. One can
#'   simply set it equal to \code{bstop}. Defaults to 50.
#'   
#' @param nu The learning rate. Defaults to 0.1.
#' 
#' @param group.type Specify the type of LS-Boost procedure. Options \code{1} and \code{2}
#'   refer to the LS-Boost1 and LS-Boost2 methods in Huang (2022), respectively. Defaults 
#'   to 1.
#'   
#' @param intercept A logical value for whether to add intercept in the prediction function.
#'   The two LS-Boost methods always estimate the intercept. The logical value of 
#'   \code{intercept} only affects the prediction function. Defaults to TRUE.
#'   
#' @details This is the main function of the package. It returns parameter estimates, 
#'   standard errors, and p-values for all selected variables at every boosting step.
#'  
#' @return \code{boostvar} returns a S3 class "boostvar" with the following components:
#'   \item{y}{the n by d matrix of y on the l.h.s. of the
#'   equation in the multivariate regression format.} \item{x}{the \eqn{n*pd} matrix of x on
#'   the r.h.s. of the equation in the multivariate regression format. It consists of p
#'   blocks of \eqn{n * d} matrices. Each \eqn{n * d} matrix includes the d variables at lag p.}
#'   \item{n}{the sample size} \item{d}{the number of variables in the VAR} \item{p}{the 
#'   lag of the VAR model} \item{bstop}{the number of boosting steps} \item{nlam}{the 
#'   value of \code{n_lambda}} \item{beta}{an array of dimension \eqn{d * d * p * bstop} that 
#'   stores, for each boosting step, the \eqn{d*d} coefficient matrix for each of the lag. If
#'   the value of a parameter estimate is 0 at a given boosting step, it means that parameter
#'   has not been updated up to that step.} 
#'   \item{beta0}{a \eqn{d * bstop} matrix that stores the \eqn{d * 1} vector of intercept for 
#'   each boosting step} \item{se}{an array of dimension \eqn{d * d * p * bstop} that 
#'   stores, for each boosting step, the \eqn{d*d} standard error matrix for each of the lag.
#'   If the value of a se is 0 at a given boosting step, it means the corresponding parameter 
#'   has not been updated up to that step.}
#'   \item{pval}{an array of dimension \eqn{d * d * p * bstop} that 
#'   stores, for each boosting step, the \eqn{d*d} p-value matrix for each of the lag. If the
#'   p-value is 1 at a given step, the corresponding parameter has not been updated up to
#'   that boosting step.} \item{pval_stacked}{a \eqn{pd * d} matrix of p-values, rearranging
#'   the values in \code{pval}. The shape of this matrix matches the \eqn{pd * d} parameter
#'   matrix in the multivariate regression format of a VAR. Each element of this matrix
#'   gives the p-value for the corresponding parameter in the following equation:
#'   \deqn{Y = X\phi+u.}} \item{intercept}{the \code{intercept} logical variable} 
#'   \item{step_mat}{the matrix that records which variable is updated at each boosting step.
#'   For the method with \code{group.type=1}, it is a \eqn{d * bstop} matrix. For the method
#'   with \code{group.type=2}, it is a \eqn{d^2 * bstop} matrix.} \item{aic}{a sequence of 
#'   Akaike information criterion value for each boosting step}
#'   
#'   
#' @examples  
#'   \dontrun{
#'   # An example of bivariate vector autoregression
#'   library(boostvar)
#'   set.seed(123)
#'   p = 2
#'   d = 2
#'   nobs = 100
#'   bstop = 50
#'   y = matrix(0, nobs, d)
#'   b0 = matrix(c(0.02,0.03),2,1)
#'   b1 = matrix(c(0.5,0.4,0.1,0.5),2,2)
#'   b2 = matrix(c(0,0.25,0,0),2,2)
#'   for (i in 3:dim(y)[1]) {
#'     y[i,] = t(b0) + t(b1 %*%  y[i-1,]) + t(b2 %*% y[i-2,]) +
#'       c(rnorm(1,0,0.3), rnorm(1,0,0.2))
#'   }
#'   y = tail(y,-p)
#'   result1 = boostvar(y, p = p, bstop = bstop, nu = 0.1, group.type = 1, intercept = TRUE)
#'   result2 = boostvar(y, p = p, bstop = bstop, nu = 0.1, group.type = 2, intercept = TRUE)
#'   }
#' 
#' @export boostvar
#' @export

# This is the main function.
boostvar <- function(datay, 
                     p = 1, 
                     bstop = 50,
                     n_lambda = 50,
                     nu = 0.1,
                     group.type = 1,
                     intercept = FALSE) { 
  
  if(n_lambda > bstop) {
    stop("n_lambda can not be larger than bstop.")
  }
  
  if (group.type == 1) {
    
    result = lsboost1(datay = datay,
                      p = p,
                      bstop = bstop,
                      nu = nu)
  } else if (group.type == 2) {
    
    result = lsboost2(datay = datay,
                      p = p,
                      bstop = bstop,
                      nu = nu)
  }
  
  d = dim(datay)[2]
  n = dim(datay)[1] - p
  
  if (group.type == 1) {
    
    beta_mat = array(t(result$beta_mat), dim = c(d,p,d,bstop))
    beta_mat = aperm(beta_mat, c(2,1,3,4))
    p_mat    = array(t(result$p_mat), dim = c(d,p,d,bstop))
    p_mat    = aperm(p_mat, c(2,1,3,4))
    se_mat   = array(t(result$se_mat), dim = c(d,p,d,bstop))
    se_mat   = aperm(se_mat, c(2,1,3,4))
    
    # Reshape the parameters, s.e., and p-value matrices.
    beta = array(0, dim = c(d,d,p,bstop))
    pval = array(0, dim = c(d,d,p,bstop))
    se   = array(0, dim = c(d,d,p,bstop))
    
    for (k in 1:bstop) {
      for (s in 1:p) {
        for (j in 1:d) {   
          beta[,j,s,k] = beta_mat[s,,j,k]
          pval[,j,s,k] = p_mat[s,,j,k]
          se[,j,s,k]   = se_mat[s,,j,k]
        }
      }
    }
  } else if (group.type == 2) {
    
    beta_mat = array(t(result$beta_mat), dim = c(d,d,p,bstop))
    beta_mat = aperm(beta_mat, c(2,1,3,4))
    p_mat    = array(t(result$p_mat), dim = c(d,d,p,bstop))
    p_mat    = aperm(p_mat, c(2,1,3,4))
    se_mat   = array(t(result$se_mat), dim = c(d,d,p,bstop))
    se_mat   = aperm(se_mat, c(2,1,3,4))

    beta = beta_mat
    pval = p_mat
    se   = se_mat
    
  }

  # Compute the n*pd x matrix.
  x = matrix(0,n,p*d)
  for (s in 1:p) {
    x[,((s-1)*d + 1):((s-1)*d + d)] = datay[((p-s)+1):((p-s)+n),]
  }
  y = tail(datay,-p)
  
  # Select n_lambda number of beta and p values to return.
  boost.seq    = as.integer(seq(1, bstop, length = n_lambda))
  beta.return  = array(beta[,,,boost.seq],dim = c(d,d,p,length(boost.seq)))
  beta0.return = (result$beta0_mat)[,boost.seq]
  pval.return  = array(pval[,,,boost.seq],dim = c(d,d,p,length(boost.seq)))
  se.return    = array(se[,,,boost.seq],dim = c(d,d,p,length(boost.seq)))
  
  
  # Stack the p values.
  pval.return.stacked = array(0,c(p*d,d,n_lambda))
  for (k in 1:n_lambda) {
    for (s in 1:p) {
      pval.return.stacked[((s-1)*d + 1):((s-1)*d + d),,k] = t(pval.return[,,s,k])
    }
  }
  
  # Export the step_mat.
  step_mat = (result$step_mat)[,boost.seq]
  
  obj = list()
  
  obj$y = y  
  obj$x = x  
  obj$n = n
  obj$d = d
  obj$p = p
  obj$bstop = bstop
  obj$nlam  = n_lambda
  obj$beta  = beta.return
  obj$beta0 = beta0.return
  obj$se    = se.return
  obj$pval  = pval.return
  obj$pval.stacked = pval.return.stacked
  obj$intercept = intercept
  obj$step_mat = step_mat
  obj$aic = result$aic
  
  class(obj) = "boostvar"
  return(obj)
}

#' @title Coefficient function for boostvar object
#' 
#' @description This function returns the coefficient of an estimated VAR model.
#'   It can return the coefficient estimates in a stacked matrix format.
#'   
#' @param object The S3 object returned by the \code{boostvar} function.
#' 
#' @param stacked A logical variable to indicate the format of the returned coefficients. If
#'   the value is \code{TRUE}, the function will return a \eqn{pd * d} matrix of 
#'   all coefficients, matching the shape of the parameter in the equation 
#'   \deqn{Y = X \phi + u.} If the value is \code{FALSE}, this function simply return the 
#'   the same coefficient as the function \code{boostvar} does.
#'   
#' 
#' @export coef.boostvar
#' @export
coef.boostvar <- function(object, stacked = FALSE) {
  p = object$p
  d = object$d
  bstop = object$bstop
  
  bstop = object$nlam
  
  if (stacked) {
      # Stack the coefficient so that it is a pd*d matrix.
      beta.stacked = array(0,c(p*d,d,bstop))
      for (k in 1:bstop) {
        for (s in 1:p) {
          beta.stacked[((s-1)*d + 1):((s-1)*d + d),,k] = t((object$beta)[,,s,k])
        }
      }
    return(beta.stacked)
  } else {
    return(object$beta)
  }
  
}

#' @title Prediction function for the boostvar object
#' 
#' @description This function can return the prediction for the entire sequence of parameter
#'   estimates or for parameter estimates at a selected boosting step.
#'   
#' @param object The S3 object returned by the \code{boostvar} function.
#' 
#' @param newx The data at which the prediction is made. \code{newx} must be a \eqn{n * pd}
#'   matrix, same as the data matrix X in equations (3) and (4) in Huang (2022).
#' 
#' @param k a boosting step, whose corresponding parameter estimates are used in prediction.
#' @export predict.boostvar
#' @export
predict.boostvar <- function(object, newx, k=NULL) {
  
  if(missing(newx)) newx = object$x
  n = dim(newx)[1]
  d = object$d
  p = object$p
  #bstop = object$bstop
  bstop = object$nlam
  
  if (is.null(k)) {
    # Return an array of predictions for each boosting step.
    pred = apply(coef.boostvar(object, stacked = TRUE), 3, 
                 function(x) {newx %*% x })
    
    if (object$intercept) {
      pred = pred + (object$beta0)[rep(seq(d),each=n),]  # Add intercept.
    }
    
    dim(pred) = c(n,d,bstop)
    return(pred)
  } else {
    
    if (length(k) > 1) stop("The input k should be a single number.")
    
    pred = newx %*% (coef.boostvar(object, stacked = TRUE))[,,k] +
      t(replicate(n,(object$beta0)[,k]))  # Add intercept.
    return(pred)
  }
  
}
