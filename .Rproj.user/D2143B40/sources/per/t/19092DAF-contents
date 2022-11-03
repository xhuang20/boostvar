// -*- mode: C++; c-indent-level: 4; c-basic-offset: 4; indent-tabs-mode: nil; -*-

#include <RcppEigen.h>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <cmath>

// [[Rcpp::depends(RcppEigen)]]

using Eigen::Map;                       
using Eigen::MatrixXd;                  
using Eigen::VectorXd;                  

//' @title CPP function for the second group boosting method in Huang (2022)
//' 
//' @description This function computes the LS-Boost2 estimates and their 
//'   p-values. 
//' 
//' @param datay An n by d matrix of variables with n observations and d variables.
//'   n is the number of time periods in the data, where the first row gives the
//'   first observation and the last row includes the most recent observation.
//' @param p A positive integer for the lag of VAR. Defaults to 1.
//' @param bstop The number of boosting steps. Defaults to 50.
//' @param nu The learning rate. Defaults to 0.1.
//' @return \code{lsboost2} returns a list with the following components: \itemize{
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
//' \item step_mat: A \eqn{d^2*bstop} matrix that records which variable is updated at 
//'   each boosting step.
//' }
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
//'   result = lsboost2(y, p = p, bstop = bstop, nu = 0.1)
//'   }
// [[Rcpp::export]]
Rcpp::List lsboost2(const Eigen::MatrixXd & datay, 
                    int p = 1,
                    int bstop = 50,
                    double nu = 0.1) {
  
  int n = datay.rows() - p;
  int d = datay.cols();
  
  // Create the y matrix. y0 is used to compute the intercept.
  MatrixXd y0 = datay.bottomRows(n);
  MatrixXd y  = y0.rowwise() - y0.colwise().mean();
  
  MatrixXd x(p*n,d);
  MatrixXd x0(p*n,d);
  
  for (int s = 0; s < p; s++) {
    x0.block(s*n,0,n,d) = datay.block(p-1-s,0,n,d);
    x.block(s*n,0,n,d)  = datay.block(p-1-s,0,n,d).rowwise() -
      datay.block(p-1-s,0,n,d).colwise().mean();
  }
  
  MatrixXd I_mat = MatrixXd::Identity(n,n);
  
  MatrixXd A_mat = MatrixXd::Zero(p*n,d);
  
  for (int s = 0; s < p; s++) {
    for (int j = 0; j < d; j++) {
      A_mat.block(s*n,j,n,1) = x.block(s*n,j,n,1) / x.block(s*n,j,n,1).squaredNorm();
    }
  }
  
  MatrixXd A_array = MatrixXd::Zero(bstop*p*n,d);
  
  MatrixXd H_mat = MatrixXd::Zero(p*d*n,n);
  for (int s = 0; s < p; s++) {
    for (int j = 0; j < d; j++) {
      H_mat.block((s*d+j)*n,0,n,n) = x.block(s*n,j,n,1) * x.block(s*n,j,n,1).transpose() / x.block(s*n,j,n,1).squaredNorm();
    }
  }
  
  VectorXd df = VectorXd::Zero(bstop);
  VectorXd aic = VectorXd::Zero(bstop);
  MatrixXd step_mat = MatrixXd::Zero(d*p,bstop);
  
  MatrixXd beta_mat = MatrixXd::Zero(bstop*p*d,d);
  MatrixXd sig_mat  = MatrixXd::Zero(bstop*p*d,d);
  MatrixXd p_mat    = MatrixXd::Constant(bstop*p*d,d,1.0);
  
  MatrixXd InuH_mat = MatrixXd::Identity(n,n).replicate(bstop,1);
  
  VectorXd min_list = VectorXd::Zero(bstop);
  
  MatrixXd B_mat = MatrixXd::Zero(n,n);
  MatrixXd B_mat_old = B_mat;
  
  MatrixXd c_cumsum = MatrixXd::Zero(p*n,d);
  
  MatrixXd res(n,d);
  int min_ind;
  MatrixXd sig2(d,d);
  MatrixXd c_coef(1,n);
  double Q;
  
  // Start the boosting iteration.
  for (int k = 0; k < bstop; k++) {
    
    if (k == 0) {
      res = y;
    }
    
    // This beta matrix stores the beta for each variable (each data column).
    // Each d*1 column is the transpose of 1*n row vector, phi_js'.
    MatrixXd beta = MatrixXd::Zero(p*d,d);
    MatrixXd resid2_mat = MatrixXd::Zero(d,p);
    
    for (int s = 0; s < p; s++) {
      for (int j = 0; j < d; j++) {
        beta.block(s*d,j,d,1) = res.transpose() * A_mat.block(s*n,j,n,1);
        auto res_js = res - x.block(s*n,j,n,1) * beta.block(s*d,j,d,1).transpose();
        resid2_mat(j,s) = res_js.squaredNorm();
      }
    }
    
    // Select the variable that gives the smalest RSS.
    Eigen::Index j_ind, s_ind;
    float min_val = resid2_mat.minCoeff(&j_ind, &s_ind);
    int var_id = s_ind * d + j_ind;
    
    step_mat(var_id,k) = 1;
    
    min_list(k) = var_id;
    
    if (k == 0) {
      beta_mat.block((k*p+s_ind)*d,j_ind,d,1) = nu * beta.block(s_ind*d,j_ind,d,1);
    } else {
      
      for (int s = 0; s < p; s++) {
        for (int j = 0; j < d; j++) {
          if (j == j_ind && s == s_ind) {
            beta_mat.block((k*p + s)*d,j,d,1) = beta_mat.block(((k-1)*p + s)*d,j,d,1) + nu * beta.block(s_ind*d,j_ind,d,1);
          } else {
            beta_mat.block((k*p + s)*d,j,d,1) = beta_mat.block(((k-1)*p + s)*d,j,d,1);
          }
        }
      }
      
    }
    
    res = res - nu * x.block(s_ind*n,j_ind,n,1) * beta.block(s_ind*d,j_ind,d,1).transpose();
    
    B_mat = B_mat_old + nu * H_mat.block((s_ind*d+j_ind)*n,0,n,n) * (I_mat - B_mat_old);
    
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
    
    A_array.block((k*p + s_ind)*n,j_ind,n,1) = A_mat.block(s_ind*n,j_ind,n,1);
    
    if (k > 0) {
      InuH_mat.block(k*n,0,n,n) = I_mat - B_mat_old;
    }
    
    B_mat_old = B_mat;
    
    c_coef = nu * A_array.block((k*p + s_ind)*n,j_ind,n,1).transpose() * InuH_mat.block(k*n,0,n,n);
    Q = (c_cumsum.block(s_ind*n,j_ind,n,1).transpose() + c_coef).squaredNorm();
    
    c_cumsum.block(s_ind*n,j_ind,n,1) = c_cumsum.block(s_ind*n,j_ind,n,1) + c_coef.transpose();
    
    // Update sig_mat and p_mat first.
    if (k > 0) {
      sig_mat.block(k*p*d,0,p*d,d) = sig_mat.block((k-1)*p*d,0,p*d,d);
      p_mat.block(k*p*d,0,p*d,d) = p_mat.block((k-1)*p*d,0,p*d,d);
    }
    
    // Compute the standard error for the d*1 column in beta[,j_ind,s_ind].
    sig_mat.block((k*p+s_ind)*d,j_ind,d,1) = (Q * sig2.diagonal()).cwiseSqrt();
    
    // Compute the d*1 column of p values. 
    // Use a loop since the pnorm5 function is not vectorized.
    for (int i = (k*p+s_ind)*d; i < (k*p+s_ind)*d+d; i++) {
      p_mat(i,j_ind) = 2 * R::pnorm5(-std::abs(beta_mat(i,j_ind) / sig_mat(i,j_ind)), 0.0, 1, 1, 0);
    }
    
  }
    
  // Compute the intercept.
  MatrixXd b0(d,bstop);
  
  for (int k = 0; k < bstop; k++) {
    MatrixXd xbarhat = MatrixXd::Zero(1,d);
    for (int s = 0; s < p; s++) {
      xbarhat = xbarhat + x0.block(s*n,0,n,d).colwise().mean() * beta_mat.block((k*p + s)*d,0,d,d).transpose();
    }
    b0.col(k) = y0.colwise().mean() - xbarhat;
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

