// -*- mode: C++; c-indent-level: 4; c-basic-offset: 4; indent-tabs-mode: nil; -*-

#include <RcppEigen.h>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <cmath>

// [[Rcpp::depends(RcppEigen)]]

using Eigen::Map;                       
using Eigen::MatrixXd;                  
using Eigen::VectorXd;        

//' @title CPP function for the first group boosting method in Huang (2022)
//' 
//' @description This function computes the LS-Boost1 estimates and their 
//'   p-values. 
//' 
//' @param datay An n by d matrix of variables with n observations and d variables.
//'   n is the number of time periods in the data, where the first row gives the
//'   first observation and the last row includes the most recent observation.
//' @param p A positive integer for the lag of VAR. Defaults to 1.
//' @param bstop The number of boosting steps. Defaults to 50.
//' @param nu The learning rate. Defaults to 0.1.
//' @return \code{lsboost1} returns a list with the following components: \itemize{
//' \item y: The (n-p)*d matrix of the dependent variable on the left hand side of the 
//'   regression equation.
//' \item x: The d(n-p)*p matrix of the X matrix. Each (n-p)*p block includes the
//'   p lags of a variable. Each lag is an (n-p)*1 column.
//' \item aic: A sequence of AIC values of length \code{bstop}.
//' \item df: A sequence of degree of freedom estimates of length \code{bstop}.
//' \item beta_mat: A \eqn{bstop*d*p} by d matrix of coefficient estimates. It stacks
//'   all \code{bstop} \eqn{\phi_{g}} coefficient matrices in equation (7) in Huang (2022).
//' \item se_mat: A matrix of standard errors for \code{beta_mat}.
//' \item pval_mat: A matrix of p-values for \code{beta_mat}.
//' \item beta0_mat: A \eqn{d*bstop} matrix of intercept estimates for each boosting step.
//' \item step_mat: A \eqn{d*bstop} matrix that records which variable is updated at 
//'   each boosting step.
//' }
//' 
//' @examples
//' \dontrun{
//'   # An example of bivariate vector autoregression
//'   library(boostvar)
//'   set.seed(123)
//'   p = 2
//'   d = 2
//'   nobs = 100
//'   bstop = 50
//'   y = matrix(0, nobs, d)
//'   b0 = matrix(c(0.02,0.03),2,1)
//'   b1 = matrix(c(0.5,0.4,0.1,0.5),2,2)
//'   b2 = matrix(c(0,0.25,0,0),2,2)
//'   for (i in 3:dim(y)[1]) {
//'     y[i,] = t(b0) + t(b1 %*%  y[i-1,]) + t(b2 %*% y[i-2,]) +
//'       c(rnorm(1,0,0.3), rnorm(1,0,0.2))
//'   }
//'   y = tail(y,-p)
//'   result = lsboost1(y, p = p, bstop = bstop, nu = 0.1)
//'   }
// [[Rcpp::export]]
Rcpp::List lsboost1(const Eigen::MatrixXd & datay, 
                    int p = 1,
                    int bstop = 50,
                    double nu = 0.1) {
  
  int n = datay.rows() - p;
  int d = datay.cols();
  
  // Create the y matrix. y0 is used to compute the intercept.
  MatrixXd y0 = datay.bottomRows(n);
  MatrixXd y  = y0.rowwise() - y0.colwise().mean();
  
  // Create a matrix for the rearranged x. 
  MatrixXd x(d*n,p);
  MatrixXd x0(d*n,p);

  for (int i = 0; i < d; i++) {
    for (int j = 0; j < p; j++) {
      
      x0.block(i*n,p-1-j,n,1) = datay.block(j,i,n,1);
      x.block(i*n,p-1-j,n,1)  = datay.block(j,i,n,1).rowwise() - 
        datay.block(j,i,n,1).colwise().mean();
      
    }
  }
  
  MatrixXd I_mat = MatrixXd::Identity(n,n);

  MatrixXd xtxinv(d*p,p);
  for (int j = 0; j < d; j++) {
    xtxinv.block(j*p,0,p,p) = (x.block(j*n,0,n,p).transpose() * x.block(j*n,0,n,p)).llt().solve(MatrixXd::Identity(p,p));  
  }

  MatrixXd A_mat(d*p,n);
  for (int j = 0; j < d; j++) {
    A_mat.block(j*p,0,p,n) = xtxinv.block(j*p,0,p,p) * x.block(j*n,0,n,p).transpose();
  }
  
  MatrixXd A_array = MatrixXd::Zero(bstop*d*p,n);
  
  MatrixXd H_mat(d*n,n);
  for (int j = 0; j < d; j++) {
    H_mat.block(j*n,0,n,n) = x.block(j*n,0,n,p) * A_mat.block(j*p,0,p,n);
  }
  
  VectorXd df = VectorXd::Zero(bstop);
  VectorXd aic = VectorXd::Zero(bstop);
  MatrixXd step_mat = MatrixXd::Zero(d,bstop);
  
  // It will be better to use a sparse matrix...
  MatrixXd beta_mat = MatrixXd::Zero(bstop*d*p,d);
  
  MatrixXd sig_mat   = MatrixXd::Zero(bstop*d*p,d);
  MatrixXd p_mat   = MatrixXd::Constant(bstop*d*p,d,1.0); 
  
  MatrixXd InuH_mat = MatrixXd::Identity(n,n).replicate(bstop,1);
  
  VectorXd min_list = VectorXd::Zero(bstop);
  
  MatrixXd B_mat = MatrixXd::Zero(n,n);
  MatrixXd B_mat_old = B_mat;
  
  MatrixXd c_cumsum = MatrixXd::Zero(d*p,n);
  
  MatrixXd res(n,d);
  int min_ind;
  MatrixXd sig2;
  MatrixXd Q(p,p);
  MatrixXd c_coef(p,n);
  VectorXd coef_se(d*p);
  
  // Start the boosting iteration.
  for (int k = 0; k < bstop; k++) {
    
    if (k == 0) {
      res = y;
    }
    
    MatrixXd beta = MatrixXd::Zero(d*p,d);
    VectorXd resid2_vec = VectorXd::Zero(d);
    
    for (int j = 0; j < d; j++) {
      beta.block(j*p,0,p,d) = A_mat.block(j*p,0,p,n) * res;
      auto res_j = res - x.block(j*n,0,n,p) * beta.block(j*p,0,p,d);
      resid2_vec(j) = res_j.squaredNorm();
    }
    
    // Select the variable that gives the smallest RSS.
    double min_of_resid2_vec = resid2_vec.minCoeff(&min_ind); 
    
    step_mat(min_ind, k) = 1;

    min_list(k) = min_ind;
    
    if (k == 0) {
      beta_mat.block((k*d+min_ind)*p,0,p,d) = nu * beta.block(min_ind*p,0,p,d);
    } else {
      
      for (int j = 0; j < d; j++) {
        if (j == min_ind) {
          beta_mat.block((k*d+j)*p,0,p,d) = beta_mat.block(((k-1)*d+j)*p,0,p,d) +
            nu * beta.block(j*p,0,p,d);
        } else {
          beta_mat.block((k*d+j)*p,0,p,d) = beta_mat.block(((k-1)*d+j)*p,0,p,d);
        }
      }
      
    }
    
    res = res - nu * x.block(min_ind*n,0,n,p) * beta.block(min_ind*p,0,p,d);

    B_mat = B_mat_old + nu * H_mat.block(min_ind*n,0,n,n) * (I_mat - B_mat_old);
    
    df(k) = B_mat.diagonal().sum();

    sig2 = (y - B_mat * y).transpose() * (y - B_mat * y) / (n - df(k));
    
    // Compute the AIC.
    if (n > d) {
      aic(k) = log(sig2.determinant()) + 2 * df(k) / n;
    } else {
      // Compute the Cp statistic.
      double s2 = (y - B_mat * y).squaredNorm() / ((n - df(k))*d);
      aic(k) = (y - B_mat * y).squaredNorm() / (n - df(k)) + 2 * df(k) / (n - df(k)) * s2;
    }
    A_array.block((k*d + min_ind)*p,0,p,n) = A_mat.block(min_ind*p,0,p,n);
    
    if (k > 0) {
      InuH_mat.block(k*n,0,n,n) = I_mat - B_mat_old;
    }
    
    B_mat_old = B_mat;
    
    c_coef = nu * A_array.block((k*d + min_ind)*p,0,p,n) * InuH_mat.block(k*n,0,n,n);
    Q = (c_cumsum.block(min_ind*p,0,p,n) + c_coef) * (c_cumsum.block(min_ind*p,0,p,n) + c_coef).transpose();
    
    c_cumsum.block(min_ind*p,0,p,n) = c_cumsum.block(min_ind*p,0,p,n) + c_coef;
    
    // Update the sig_mat and p_mat first.
    if (k > 0) {
      sig_mat.block(k*d*p,0,d*p,d) = sig_mat.block((k-1)*d*p,0,d*p,d);
      p_mat.block(k*d*p,0,d*p,d) = p_mat.block((k-1)*d*p,0,d*p,d);
    }
    
    // Compute the standard errors.
    for (int j = 0; j < d; j++) {
      coef_se.segment(j*p,p) = (sig2.diagonal()(j) * Q.diagonal()).cwiseSqrt();
    }
    
    Map<MatrixXd> se_seg(coef_se.data(), p, d); 
    
    sig_mat.block((k*d + min_ind)*p,0,p,d) = se_seg;
    
    // Compute the p value matrix. Use a loop since the pnorm function in
    // Rmath is not vectorized.
    for (int j = 0; j < d; j++) {
      for (int i = (k*d + min_ind)*p; i < (k*d + min_ind)*p + p; i++) {
        p_mat(i,j) = 2 * R::pnorm5(-std::abs(beta_mat(i,j) / sig_mat(i,j)), 0.0, 1, 1, 0);
      }
    }
    
  } 
  // Compute the intercept.
  MatrixXd b0 = MatrixXd::Zero(d,bstop);
  
  for (int k = 0; k < bstop; k++) {
    MatrixXd xbarhat = MatrixXd::Zero(1,d);
    for (int j = 0; j < d; j++) {
      xbarhat = xbarhat + (x0.block(j*n,0,n,p)).colwise().mean() * beta_mat.block((k*d+j)*p,0,p,d);
    }
    b0.col(k) = (y0.colwise().mean() - xbarhat).transpose();
  }
  
  return Rcpp::List::create(Rcpp::Named("y") = y0,
                            Rcpp::Named("x") = x0,
                            Rcpp::Named("aic") = aic,
                            Rcpp::Named("df") = df,
                            Rcpp::Named("beta_mat") = beta_mat,
                            Rcpp::Named("se_mat") = sig_mat,
                            Rcpp::Named("p_mat") = p_mat,
                            Rcpp::Named("beta0_mat") = b0,
                            Rcpp::Named("step_mat") = step_mat);
  
}