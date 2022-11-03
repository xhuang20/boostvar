// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <RcppEigen.h>
#include <Rcpp.h>

using namespace Rcpp;

#ifdef RCPP_USE_GLOBAL_ROSTREAM
Rcpp::Rostream<true>&  Rcpp::Rcout = Rcpp::Rcpp_cout_get();
Rcpp::Rostream<false>& Rcpp::Rcerr = Rcpp::Rcpp_cerr_get();
#endif

// boostlscpp
Rcpp::List boostlscpp(const Eigen::VectorXd& y, const Eigen::MatrixXd& x, int bstop, double nu);
RcppExport SEXP _boostvar_boostlscpp(SEXP ySEXP, SEXP xSEXP, SEXP bstopSEXP, SEXP nuSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const Eigen::VectorXd& >::type y(ySEXP);
    Rcpp::traits::input_parameter< const Eigen::MatrixXd& >::type x(xSEXP);
    Rcpp::traits::input_parameter< int >::type bstop(bstopSEXP);
    Rcpp::traits::input_parameter< double >::type nu(nuSEXP);
    rcpp_result_gen = Rcpp::wrap(boostlscpp(y, x, bstop, nu));
    return rcpp_result_gen;
END_RCPP
}
// lsboost1
Rcpp::List lsboost1(const Eigen::MatrixXd& datay, int p, int bstop, double nu);
RcppExport SEXP _boostvar_lsboost1(SEXP dataySEXP, SEXP pSEXP, SEXP bstopSEXP, SEXP nuSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const Eigen::MatrixXd& >::type datay(dataySEXP);
    Rcpp::traits::input_parameter< int >::type p(pSEXP);
    Rcpp::traits::input_parameter< int >::type bstop(bstopSEXP);
    Rcpp::traits::input_parameter< double >::type nu(nuSEXP);
    rcpp_result_gen = Rcpp::wrap(lsboost1(datay, p, bstop, nu));
    return rcpp_result_gen;
END_RCPP
}
// lsboost2
Rcpp::List lsboost2(const Eigen::MatrixXd& datay, int p, int bstop, double nu);
RcppExport SEXP _boostvar_lsboost2(SEXP dataySEXP, SEXP pSEXP, SEXP bstopSEXP, SEXP nuSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const Eigen::MatrixXd& >::type datay(dataySEXP);
    Rcpp::traits::input_parameter< int >::type p(pSEXP);
    Rcpp::traits::input_parameter< int >::type bstop(bstopSEXP);
    Rcpp::traits::input_parameter< double >::type nu(nuSEXP);
    rcpp_result_gen = Rcpp::wrap(lsboost2(datay, p, bstop, nu));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_boostvar_boostlscpp", (DL_FUNC) &_boostvar_boostlscpp, 4},
    {"_boostvar_lsboost1", (DL_FUNC) &_boostvar_lsboost1, 4},
    {"_boostvar_lsboost2", (DL_FUNC) &_boostvar_lsboost2, 4},
    {NULL, NULL, 0}
};

RcppExport void R_init_boostvar(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
