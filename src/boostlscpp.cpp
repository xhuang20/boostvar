// -*- mode: C++; c-indent-level: 4; c-basic-offset: 4; indent-tabs-mode: nil; -*-

#include <RcppEigen.h>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <cmath>

// [[Rcpp::depends(RcppEigen)]]

using Eigen::Map;                   
using Eigen::MatrixXd;              
using Eigen::VectorXd;                  

//' @title Function to compute least-squares boosting and the p-value
//' 
//' @description This function computes the LS-Boost estimates, the standard errors, and 
//'   the p-values for a cross-section regression model.
//'   
//' @param y The n by 1 vector of the dependent variable. 
//' @param x The n by p matrix of the independent variables.
//' @param bstop The number of boosting steps. Defaults to 50.
//' @param nu The learning rate. Defaults to 0.1.
//' @return The \code{boostlscpp} function returns a list with the following components:
//'   \item{y}{The n by 1 vector of the dependent variable.}
//'   \item{x}{The n by p matrix of the independent variables.}
//'   \item{bstop}{The number of boosting steps.}
//'   \item{beta}{The p by bstop matrix of parameter estimates in all boosting steps.}
//'   \item{beta0}{The bstop by 1 vector of intercept estimates in all boosting steps.}
//'   \item{se}{The p by bstop matrix of standard errors in all boosting steps.}
//'   \item{pval}{The p by bstop matrix of p-values for all parameter estimates in all
//'         boosting steps. If a p-value is equal to 1, it means the corresponding parameter has
//'         not been updated up to that boosting step.}
//'   \item{step_mat}{A p by bstop matrix that records which variable is updated at each
//'         boosting step.}
//'   \item{aic}{A sequence of AIC values of length \code{bstop}.}
// [[Rcpp::export]]
Rcpp::List boostlscpp(const Eigen::VectorXd & y, 
                      const Eigen::MatrixXd & x,
                      int bstop = 50,
                      double nu = 0.1) {
  
  int n = x.rows();
  int p = x.cols();
  
  MatrixXd I_mat = MatrixXd::Identity(n,n);
  VectorXd aic = VectorXd::Zero(bstop);
  
  VectorXd ydm = y.array() - y.mean();
  MatrixXd xdm = x.rowwise() - x.colwise().mean();
  
  MatrixXd H_mat(n*p,n);
  MatrixXd H_mat_sum(n*bstop,n);
  
  MatrixXd InuH_mat = MatrixXd::Identity(n,n).replicate(bstop,1);
  
  MatrixXd A_mat(p,n);
  for (int j = 0; j < p; j++) {
    H_mat.block(j*n,0,n,n) = xdm.col(j) * xdm.col(j).transpose() / xdm.col(j).dot(xdm.col(j));
    A_mat.row(j) = xdm.col(j).transpose() / xdm.col(j).dot(xdm.col(j));
  }
  
  MatrixXd A_array = MatrixXd::Zero(bstop*p,n); 
  
  MatrixXd t_nume = MatrixXd::Zero(p,bstop);
  MatrixXd t_deno = MatrixXd::Zero(p,bstop);
  MatrixXd t_mat  = MatrixXd::Zero(p,bstop);
  MatrixXd p_mat  = MatrixXd::Ones(p,bstop);
  
  VectorXd df = VectorXd::Zero(bstop);
  MatrixXd step_mat = MatrixXd::Zero(p,bstop);
  MatrixXd sig_mat  = MatrixXd::Zero(p,bstop);      
  
  MatrixXd b_delta = MatrixXd::Zero(p,bstop);
  MatrixXd s_delta = MatrixXd::Zero(p,bstop);
  VectorXd p_delta = VectorXd::Zero(bstop);
  
  VectorXd rss = VectorXd::Zero(bstop);
  
  MatrixXd beta_mat  = MatrixXd::Zero(p,bstop);
  MatrixXd B_mat     = MatrixXd::Zero(n,n);
  MatrixXd B_mat_old = MatrixXd::Zero(n,n);
  
  VectorXd res(n);
  int min_ind;
  MatrixXd c_cumsum = MatrixXd::Zero(p,n);
  MatrixXd c_coef(1,n);
  double Q;
  
  // Start the LS-Boost procedure with bstop iterations.
  for (int s = 0; s < bstop; s++) {
    
    if (s == 0) res = ydm;
    
    VectorXd beta = VectorXd::Zero(p);
    VectorXd resid2_vec = VectorXd::Zero(p);
    
    for (int j = 0; j < p; j++) {
      beta(j) = xdm.col(j).dot(res) / xdm.col(j).dot(xdm.col(j));
      auto resid = res - xdm.col(j) * beta(j);
      resid2_vec(j) = resid.dot(resid);
    }
    
    // Select the variable at step s.
    auto min_of_resid2_vec = resid2_vec.minCoeff(&min_ind);

    step_mat(min_ind,s) = 1;
    
    if (s == 0) {
      beta_mat(min_ind,s) = nu * beta(min_ind);
    }
    
    if (s > 0) {
      
      for (int j = 0; j < p; j++) {
        if (j == min_ind) {
          beta_mat(j,s) = beta_mat(j,s-1) + nu * beta(min_ind);
        } else {
          beta_mat(j,s) = beta_mat(j,s-1);
        }
      }
      
    }
    
    res = res - xdm.col(min_ind) * nu * beta(min_ind);   
    
    B_mat = B_mat_old + nu * H_mat.block(min_ind*n,0,n,n) * (I_mat - B_mat_old);
    
    
    df(s) = B_mat.diagonal().sum();
    
    auto sig2 = (ydm - B_mat * ydm).dot(ydm - B_mat * ydm) / (n - df(s));
    aic(s) = log(sig2) + (1 + df(s) / n) / (1 - (df(s) + 2) / n);
    
    A_array.row(min_ind*bstop + s) = A_mat.row(min_ind);
    
    // Update the sig_mat and p_mat matrices.
    if (s > 0) {
      sig_mat.col(s) = sig_mat.col(s-1);
      p_mat.col(s)   = p_mat.col(s-1);
    }

    if (s > 0) {
      InuH_mat.block(s*n,0,n,n) = I_mat - B_mat_old;
    }
    
    B_mat_old = B_mat;
    
    c_coef = nu * A_array.block(min_ind*bstop+s,0,1,n) * InuH_mat.block(s*n,0,n,n);
    Q = ((c_cumsum.row(min_ind) + c_coef) * (c_cumsum.row(min_ind) + c_coef).transpose())(0,0);   // Q is a scalar.
    
    c_cumsum.row(min_ind) = c_cumsum.row(min_ind) + c_coef;
    
    sig_mat(min_ind,s) = sqrt(Q * sig2);
    
    p_mat(min_ind,s) = 2 * R::pnorm5(-std::abs(beta_mat(min_ind,s) / sig_mat(min_ind,s)), 0.0, 1.0, 1, 0);
      
  }
    
  // Compute the intercept.
  VectorXd beta0_vec(bstop,1);
  for (int s = 0; s < bstop; s++) {
    beta0_vec(s) = y.mean() - x.colwise().mean() * beta_mat.col(s);
  }
  
  return Rcpp::List::create(Rcpp::Named("y") = y,
                            Rcpp::Named("x") = x,
                            Rcpp::Named("beta_mat") = beta_mat,
                            Rcpp::Named("se_mat") = sig_mat,
                            Rcpp::Named("p_mat") = p_mat,
                            Rcpp::Named("step_mat") = step_mat,
                            Rcpp::Named("aic") = aic,
                            Rcpp::Named("df") = df,
                            Rcpp::Named("beta0_vec") = beta0_vec);
}

