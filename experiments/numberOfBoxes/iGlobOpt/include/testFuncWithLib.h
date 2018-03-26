/*
 * testFuncWithLib.h
 *
 *  Created on: 10 Aug 2017
 *      Author: Leonid Tkachenko
 */

#ifndef INCLUDE_TESTFUNCWITHLIB_H_
#define INCLUDE_TESTFUNCWITHLIB_H_


/*
 * testFuncWithLib.cpp
 *
 *  Created on: 10 Aug 2017
 *      Author: Leonid Tkachenko
 */

/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/*
 * File:   test_func_with_lib.h
 * Author: leo
 *
 * Created on June 6, 2017, 12:43 PM
 */

#include <limits>
#include "testfuncs/testfuncs.hpp"
#include "expression/expr.hpp"
#include "expression/algorithm.hpp"
#include "descfunc/descfunc.hpp"
#include "descfunc/keys.hpp"
#include "oneobj/contboxconstr/exprfunc.hpp"


using namespace snowgoose::expression;
using namespace OPTITEST;

void calcFunc(const std::string& name, Expr<double> expr, const std::vector<double>& vars, double *outLimits)
{
	auto result = expr.calc(FuncAlg<double>(vars));
	outLimits[2] = result;
}

void calcInterval(const std::string& name, Expr<Interval<double>> expr, const std::vector<Interval<double>>& vars, double *outLimits)
{
	auto result = expr.calc(InterEvalAlg<double>(vars));
    outLimits[0] = result.lb();
    outLimits[1] = result.rb();
}

/**
*	Calculus Interval for Ackley 1 function on CPU
*	@param inbox pointer to Box
*	@param inRank number of variables
*	@param outlimits pointer to estimated function limits
*/
void calcFunBoundsAckley1WithLib(const double *inBox, const int inRank, double *outLimits)
{
    static auto expr= Ackley1<double>(2);
    calcFunc("Ackley1", expr, {(inBox[0]+inBox[1])/2, (inBox[2]+inBox[3])/2},outLimits);

    static auto exprInterval = Ackley1<Interval<double>>(2);
    calcInterval("Ackley1 estimation", exprInterval, { { inBox[0], inBox[1] }, { inBox[2], inBox[3] } }, outLimits);
}

/**
*	Calculus Interval for Ackley 2 function on CPU
*	@param inbox pointer to Box
*	@param inRank number of variables
*	@param outlimits pointer to estimated function limits
*/
void calcFunBoundsAckley2WithLib(const double *inBox, const int inRank, double *outLimits)
{
   static auto expr= Ackley2<double>(2);
    calcFunc("Ackley12", expr, {(inBox[0]+inBox[1])/2, (inBox[2]+inBox[3])/2},outLimits);

   static auto exprInterval = Ackley2<Interval<double>>(2);
    calcInterval("Ackley2 estimation", exprInterval, { { inBox[0], inBox[1] }, { inBox[2], inBox[3] } }, outLimits);
}

/**
*	Calculus Interval for Ackley 3 function on CPU
*	@param inbox pointer to Box
*	@param inRank number of variables
*	@param outlimits pointer to estimated function limits
*/
void calcFunBoundsAckley3WithLib(const double *inBox, const int inRank, double *outLimits)
{
   static auto expr= Ackley3<double>();
    calcFunc("Ackley3", expr, {(inBox[0]+inBox[1])/2, (inBox[2]+inBox[3])/2},outLimits);

   static auto exprInterval = Ackley3<Interval<double>>();
    calcInterval("Ackley3 estimation", exprInterval, { { inBox[0], inBox[1] }, { inBox[2], inBox[3] } }, outLimits);
}

/**
*	Calculus Interval for Ackley 4 function on CPU
*	@param inbox pointer to Box
*	@param inRank number of variables
*	@param outlimits pointer to estimated function limits
*/
void calcFunBoundsAckley4WithLib(const double *inBox, const int inRank, double *outLimits)
{
   static auto expr= Ackley4<double>();
    calcFunc("Ackley4", expr, {(inBox[0]+inBox[1])/2, (inBox[2]+inBox[3])/2},outLimits);

   static auto exprInterval = Ackley4<Interval<double>>();
    calcInterval("Ackley4 estimation", exprInterval, { { inBox[0], inBox[1] }, { inBox[2], inBox[3] } }, outLimits);
}

/**
*	Calculus Interval for Adjiman function on CPU
*	@param inbox pointer to Box
*	@param inRank number of variables
*	@param outlimits pointer to estimated function limits
*/
void calcFunBoundsAdjimanWithLib(const double *inBox, const int inRank, double *outLimits)
{
   static auto expr= Adjiman<double>();
    calcFunc("Adjiman", expr, {(inBox[0]+inBox[1])/2, (inBox[2]+inBox[3])/2},outLimits);

   static auto exprInterval = Adjiman<Interval<double>>();
    calcInterval("Adjiman estimation", exprInterval, { { inBox[0], inBox[1] }, { inBox[2], inBox[3] } }, outLimits);
}

/**
*	Calculus Interval for Alpine 1 function on CPU
*	@param inbox pointer to Box
*	@param inRank number of variables
*	@param outlimits pointer to estimated function limits
*/
void calcFunBoundsAlpine1WithLib(const double *inBox, const int inRank, double *outLimits)
{
    static auto expr= Alpine1<double>(2);
    calcFunc("Alpine1", expr, {(inBox[0]+inBox[1])/2, (inBox[2]+inBox[3])/2},outLimits);

    static auto exprInterval = Alpine1<Interval<double>>(2);
    calcInterval("Alpine1 estimation", exprInterval, { { inBox[0], inBox[1] }, { inBox[2], inBox[3] } }, outLimits);
}

/**
*	Calculus Interval for Alpine 2 function on CPU
*	@param inbox pointer to Box
*	@param inRank number of variables
*	@param outlimits pointer to estimated function limits
*/
void calcFunBoundsAlpine2WithLib(const double *inBox, const int inRank, double *outLimits)
{
   static auto expr= Alpine2<double>();
    calcFunc("Alpine2", expr, {(inBox[0]+inBox[1])/2, (inBox[2]+inBox[3])/2,(inBox[4]+inBox[5])/2},outLimits);

   static auto exprInterval = Alpine2<Interval<double>>();
    calcInterval("Alpine2 estimation", exprInterval, { { inBox[0], inBox[1] }, { inBox[2], inBox[3] } , { inBox[4], inBox[5] }}, outLimits);
}

/**
*	Calculus Interval for Brad function on CPU
*	@param inbox pointer to Box
*	@param inRank number of variables
*	@param outlimits pointer to estimated function limits
*/
void calcFunBoundsBradWithLib(const double *inBox, const int inRank, double *outLimits)
{
   static auto expr= Brad<double>();
    calcFunc("Brad", expr, {(inBox[0]+inBox[1])/2, (inBox[2]+inBox[3])/2,(inBox[4]+inBox[5])/2},outLimits);

   static auto exprInterval = Brad<Interval<double>>();
    calcInterval("Brad estimation", exprInterval, { { inBox[0], inBox[1] }, { inBox[2], inBox[3] } , { inBox[4], inBox[5] }}, outLimits);
}

/**
*	Calculus Interval for BartelsConn function on CPU
*	@param inbox pointer to Box
*	@param inRank number of variables
*	@param outlimits pointer to estimated function limits
*/
void calcFunBoundsBartelsConnWithLib(const double *inBox, const int inRank, double *outLimits)
{
   static auto expr= BartelsConn<double>();
    calcFunc("BartelsConn", expr, {(inBox[0]+inBox[1])/2, (inBox[2]+inBox[3])/2},outLimits);

   static auto exprInterval = BartelsConn<Interval<double>>();
    calcInterval("BartelsConn estimation", exprInterval, { { inBox[0], inBox[1] }, { inBox[2], inBox[3] }}, outLimits);
}

/**
*	Calculus Interval for Beale function on CPU
*	@param inbox pointer to Box
*	@param inRank number of variables
*	@param outlimits pointer to estimated function limits
*/
void calcFunBoundsBealeWithLib(const double *inBox, const int inRank, double *outLimits)
{
   static auto expr= Beale<double>();
    calcFunc("Beale", expr, {(inBox[0]+inBox[1])/2, (inBox[2]+inBox[3])/2},outLimits);

   static auto exprInterval = Beale<Interval<double>>();
    calcInterval("Beale estimation", exprInterval, { { inBox[0], inBox[1] }, { inBox[2], inBox[3] }}, outLimits);
}

/**
*	Calculus Interval for BiggsEXP2 function on CPU
*	@param inbox pointer to Box
*	@param inRank number of variables
*	@param outlimits pointer to estimated function limits
*/
void calcFunBoundsBiggsExpr2WithLib(const double *inBox, const int inRank, double *outLimits)
{
   static auto expr= BiggsExpr2<double>();
    calcFunc("BiggsExpr2", expr, {(inBox[0]+inBox[1])/2, (inBox[2]+inBox[3])/2},outLimits);

   static auto exprInterval = BiggsExpr2<Interval<double>>();
    calcInterval("BiggsExpr2 estimation", exprInterval, { { inBox[0], inBox[1] }, { inBox[2], inBox[3] }}, outLimits);
}

/**
*	Calculus Interval for BiggsEXP3 function on CPU
*	@param inbox pointer to Box
*	@param inRank number of variables
*	@param outlimits pointer to estimated function limits
*/
void calcFunBoundsBiggsExpr3WithLib(const double *inBox, const int inRank, double *outLimits)
{
   static auto expr= BiggsExpr3<double>();
    calcFunc("BiggsExpr3", expr, {(inBox[0]+inBox[1])/2, (inBox[2]+inBox[3])/2,(inBox[4]+inBox[5])/2},outLimits);

   static auto exprInterval = BiggsExpr3<Interval<double>>();
    calcInterval("BiggsExpr3 estimation", exprInterval, { { inBox[0], inBox[1] }, { inBox[2], inBox[3] },{ inBox[4], inBox[5] }}, outLimits);
}

/**
*	Calculus Interval for BiggsEXP4 function on CPU
*	@param inbox pointer to Box
*	@param inRank number of variables
*	@param outlimits pointer to estimated function limits
*/
void calcFunBoundsBiggsExpr4WithLib(const double *inBox, const int inRank, double *outLimits)
{
   static auto expr= BiggsExpr4<double>();
    calcFunc("BiggsExpr4", expr, {(inBox[0]+inBox[1])/2, (inBox[2]+inBox[3])/2,(inBox[4]+inBox[5])/2,(inBox[6]+inBox[7])/2},outLimits);

   static auto exprInterval = BiggsExpr4<Interval<double>>();
    calcInterval("BiggsExpr4 estimation", exprInterval, { { inBox[0], inBox[1] }, { inBox[2], inBox[3] },{ inBox[4], inBox[5] },{ inBox[6], inBox[7] }}, outLimits);
}

/**
*	Calculus Interval for BiggsEXP5 function on CPU
*	@param inbox pointer to Box
*	@param inRank number of variables
*	@param outlimits pointer to estimated function limits
*/
//auto expr = BiggsExpr5<double>();
//auto exprInterval = BiggsExpr5<Interval<double>>();
void calcFunBoundsBiggsExpr5WithLib(const double *inBox, const int inRank, double *outLimits)
{
   static auto expr= BiggsExpr5<double>();
    calcFunc("BiggsExpr5", expr, {(inBox[0]+inBox[1])/2, (inBox[2]+inBox[3])/2,(inBox[4]+inBox[5])/2,(inBox[6]+inBox[7])/2,(inBox[8]+inBox[9])/2},outLimits);

   static auto exprInterval = BiggsExpr5<Interval<double>>();
    calcInterval("BiggsExpr5 estimation", exprInterval, { { inBox[0], inBox[1] }, { inBox[2], inBox[3] },{ inBox[4], inBox[5] },{ inBox[6], inBox[7] },{ inBox[8], inBox[9] }}, outLimits);
}

/**
*	Calculus Interval for BiggsEXP6 function on CPU
*	@param inbox pointer to Box
*	@param inRank number of variables
*	@param outlimits pointer to estimated function limits
*/
void calcFunBoundsBiggsExpr6WithLib(const double *inBox, const int inRank, double *outLimits)
{
   static auto expr= BiggsExpr6<double>();
    calcFunc("BiggsExpr6", expr, {(inBox[0]+inBox[1])/2, (inBox[2]+inBox[3])/2,(inBox[4]+inBox[5])/2,(inBox[6]+inBox[7])/2,(inBox[8]+inBox[9])/2,(inBox[10]+inBox[11])/2},outLimits);

   static auto exprInterval = BiggsExpr6<Interval<double>>();
    calcInterval("BiggsExpr6 estimation", exprInterval, { { inBox[0], inBox[1] }, { inBox[2], inBox[3] },{ inBox[4], inBox[5] },{ inBox[6], inBox[7] },{ inBox[8], inBox[9] },{ inBox[10], inBox[11] }}, outLimits);
}

/**
*	Calculus Interval for Bird function on CPU
*	@param inbox pointer to Box
*	@param inRank number of variables
*	@param outlimits pointer to estimated function limits
*/
void calcFunBoundsBirdWithLib(const double *inBox, const int inRank, double *outLimits)
{
   static auto expr= Bird<double>();
    calcFunc("Bird", expr, {(inBox[0]+inBox[1])/2, (inBox[2]+inBox[3])/2},outLimits);

   static auto exprInterval = Bird<Interval<double>>();
    calcInterval("Bird estimation", exprInterval, { { inBox[0], inBox[1] }, { inBox[2], inBox[3] }}, outLimits);
}

/**
*	Calculus Interval for Bohachevsky1 function on CPU
*	@param inbox pointer to Box
*	@param inRank number of variables
*	@param outlimits pointer to estimated function limits
*/
void calcFunBoundsBohachevsky1WithLib(const double *inBox, const int inRank, double *outLimits)
{
   static auto expr= Bohachevsky1<double>();
    calcFunc("Bohachevsky1", expr, {(inBox[0]+inBox[1])/2, (inBox[2]+inBox[3])/2},outLimits);

   static auto exprInterval = Bohachevsky1<Interval<double>>();
    calcInterval("Bohachevsky1 estimation", exprInterval, { { inBox[0], inBox[1] }, { inBox[2], inBox[3] }}, outLimits);
}

/**
*	Calculus Interval for Bohachevsky2 function on CPU
*	@param inbox pointer to Box
*	@param inRank number of variables
*	@param outlimits pointer to estimated function limits
*/
void calcFunBoundsBohachevsky2WithLib(const double *inBox, const int inRank, double *outLimits)
{
   static auto expr= Bohachevsky2<double>();
    calcFunc("Bohachevsky2", expr, {(inBox[0]+inBox[1])/2, (inBox[2]+inBox[3])/2},outLimits);

   static auto exprInterval = Bohachevsky2<Interval<double>>();
    calcInterval("Bohachevsky2 estimation", exprInterval, { { inBox[0], inBox[1] }, { inBox[2], inBox[3] }}, outLimits);
}

/**
*	Calculus Interval for Booth function on CPU
*	@param inbox pointer to Box
*	@param inRank number of variables
*	@param outlimits pointer to estimated function limits
*/
void calcFunBoundsBoothWithLib(const double *inBox, const int inRank, double *outLimits)
{
   static auto expr= Booth<double>();
    calcFunc("Booth", expr, {(inBox[0]+inBox[1])/2, (inBox[2]+inBox[3])/2},outLimits);

   static auto exprInterval = Booth<Interval<double>>();
    calcInterval("Booth estimation", exprInterval, { { inBox[0], inBox[1] }, { inBox[2], inBox[3] }}, outLimits);
}

/**
*	Calculus Interval for Bohachevsky3 function on CPU
*	@param inbox pointer to Box
*	@param inRank number of variables
*	@param outlimits pointer to estimated function limits
*/
void calcFunBoundsBohachevsky3WithLib(const double *inBox, const int inRank, double *outLimits)
{
   static auto expr= Bohachevsky3<double>();
    calcFunc("Bohachevsky3", expr, {(inBox[0]+inBox[1])/2, (inBox[2]+inBox[3])/2},outLimits);

   static auto exprInterval = Bohachevsky3<Interval<double>>();
    calcInterval("Bohachevsky3 estimation", exprInterval, { { inBox[0], inBox[1] }, { inBox[2], inBox[3] }}, outLimits);
}

/**
*	Calculus Interval for BoxBettsQuadraticSum function on CPU
*	@param inbox pointer to Box
*	@param inRank number of variables
*	@param outlimits pointer to estimated function limits
*/
void calcFunBoundsBoxBettsQuadraticSumWithLib(const double *inBox, const int inRank, double *outLimits)
{
   static auto expr= BoxBettsQuadraticSum<double>();
    calcFunc("BoxBettsQuadraticSum", expr, {(inBox[0]+inBox[1])/2, (inBox[2]+inBox[3])/2,(inBox[4]+inBox[5])/2},outLimits);

   static auto exprInterval = BoxBettsQuadraticSum<Interval<double>>();
    calcInterval("BoxBettsQuadraticSum estimation", exprInterval, { { inBox[0], inBox[1] }, { inBox[2], inBox[3] },{ inBox[4], inBox[5] }}, outLimits);
}

/**
*	Calculus Interval for BraninRCOS function on CPU
*	@param inbox pointer to Box
*	@param inRank number of variables
*	@param outlimits pointer to estimated function limits
*/
void calcFunBoundsBraninRCOSWithLib(const double *inBox, const int inRank, double *outLimits)
{
   static auto expr= BraninRCOS<double>();
    calcFunc("BraninRCOS", expr, {(inBox[0]+inBox[1])/2, (inBox[2]+inBox[3])/2},outLimits);

   static auto exprInterval = BraninRCOS<Interval<double>>();
    calcInterval("BraninRCOS estimation", exprInterval, { { inBox[0], inBox[1] }, { inBox[2], inBox[3] }}, outLimits);
}

/**
*	Calculus Interval for BraninRCOS2 function on CPU
*	@param inbox pointer to Box
*	@param inRank number of variables
*	@param outlimits pointer to estimated function limits
*/
void calcFunBoundsBraninRCOS2WithLib(const double *inBox, const int inRank, double *outLimits)
{
   static auto expr= BraninRCOS2<double>();
    calcFunc("BraninRCOS2", expr, {(inBox[0]+inBox[1])/2, (inBox[2]+inBox[3])/2},outLimits);

   static auto exprInterval = BraninRCOS2<Interval<double>>();
    calcInterval("BraninRCOS2 estimation", exprInterval, { { inBox[0], inBox[1] }, { inBox[2], inBox[3] }}, outLimits);
}

/**
*	Calculus Interval for Brent function on CPU
*	@param inbox pointer to Box
*	@param inRank number of variables
*	@param outlimits pointer to estimated function limits
*/
void calcFunBoundsBrentWithLib(const double *inBox, const int inRank, double *outLimits)
{
   static auto expr= Brent<double>();
    calcFunc("Brent", expr, {(inBox[0]+inBox[1])/2, (inBox[2]+inBox[3])/2},outLimits);

   static auto exprInterval = Brent<Interval<double>>();
    calcInterval("Brent estimation", exprInterval, { { inBox[0], inBox[1] }, { inBox[2], inBox[3] }}, outLimits);
}

/**
*	Calculus Interval for Brown function on CPU
*	@param inbox pointer to Box
*	@param inRank number of variables
*	@param outlimits pointer to estimated function limits
*/
void calcFunBoundsBrownWithLib(const double *inBox, const int inRank, double *outLimits)
{
   static auto expr= Brown<double>(2);
    calcFunc("Brown", expr, {(inBox[0]+inBox[1])/2, (inBox[2]+inBox[3])/2},outLimits);

   static auto exprInterval = Brown<Interval<double>>(2);
    calcInterval("Brown estimation", exprInterval, { { inBox[0], inBox[1] }, { inBox[2], inBox[3] }}, outLimits);
}

/**
*	Calculus Interval for Bukin2 function on CPU
*	@param inbox pointer to Box
*	@param inRank number of variables
*	@param outlimits pointer to estimated function limits
*/
void calcFunBoundsBukin2WithLib(const double *inBox, const int inRank, double *outLimits)
{
   static auto expr= Bukin2<double>();
    calcFunc("Bukin2", expr, {(inBox[0]+inBox[1])/2, (inBox[2]+inBox[3])/2},outLimits);

   static auto exprInterval = Bukin2<Interval<double>>();
    calcInterval("Bukin2 estimation", exprInterval, { { inBox[0], inBox[1] }, { inBox[2], inBox[3] }}, outLimits);
}

/**
*	Calculus Interval for Bukin4 function on CPU
*	@param inbox pointer to Box
*	@param inRank number of variables
*	@param outlimits pointer to estimated function limits
*/
void calcFunBoundsBukin4WithLib(const double *inBox, const int inRank, double *outLimits)
{
   static auto expr= Bukin4<double>();
    calcFunc("Bukin4", expr, {(inBox[0]+inBox[1])/2, (inBox[2]+inBox[3])/2},outLimits);

   static auto exprInterval = Bukin4<Interval<double>>();
    calcInterval("Bukin4 estimation", exprInterval, { { inBox[0], inBox[1] }, { inBox[2], inBox[3] }}, outLimits);
}

/**
*	Calculus Interval for Bukin6 function on CPU
*	@param inbox pointer to Box
*	@param inRank number of variables
*	@param outlimits pointer to estimated function limits
*/
void calcFunBoundsBukin6WithLib(const double *inBox, const int inRank, double *outLimits)
{
   static auto expr= Bukin6<double>();
    calcFunc("Bukin6", expr, {(inBox[0]+inBox[1])/2, (inBox[2]+inBox[3])/2},outLimits);

   static auto exprInterval = Bukin6<Interval<double>>();
    calcInterval("Bukin6 estimation", exprInterval, { { inBox[0], inBox[1] }, { inBox[2], inBox[3] }}, outLimits);
}

/**
*	Calculus Interval for CamelThreeHump function on CPU
*	@param inbox pointer to Box
*	@param inRank number of variables
*	@param outlimits pointer to estimated function limits
*/
void calcFunBoundsCamelThreeHumpWithLib(const double *inBox, const int inRank, double *outLimits)
{
   static auto expr= CamelThreeHump<double>();
    calcFunc("CamelThreeHump", expr, {(inBox[0]+inBox[1])/2, (inBox[2]+inBox[3])/2},outLimits);

   static auto exprInterval = CamelThreeHump<Interval<double>>();
    calcInterval("CamelThreeHump estimation", exprInterval, { { inBox[0], inBox[1] }, { inBox[2], inBox[3] }}, outLimits);
}

/**
*	Calculus Interval for CamelSixHump function on CPU
*	@param inbox pointer to Box
*	@param inRank number of variables
*	@param outlimits pointer to estimated function limits
*/
void calcFunBoundsCamelSixHumpWithLib(const double *inBox, const int inRank, double *outLimits)
{
   static auto expr= CamelSixHump<double>();
    calcFunc("CamelSixHump", expr, {(inBox[0]+inBox[1])/2, (inBox[2]+inBox[3])/2},outLimits);

   static auto exprInterval = CamelSixHump<Interval<double>>();
    calcInterval("CamelSixHump estimation", exprInterval, { { inBox[0], inBox[1] }, { inBox[2], inBox[3] }}, outLimits);
}

/**
*	Calculus Interval for Chichinadze function on CPU
*	@param inbox pointer to Box
*	@param inRank number of variables
*	@param outlimits pointer to estimated function limits
*/
void calcFunBoundsChichinadzeWithLib(const double *inBox, const int inRank, double *outLimits)
{
   static auto expr= Chichinadze<double>();
    calcFunc("Chichinadze", expr, {(inBox[0]+inBox[1])/2, (inBox[2]+inBox[3])/2},outLimits);

   static auto exprInterval = Chichinadze<Interval<double>>();
    calcInterval("Chichinadze estimation", exprInterval, { { inBox[0], inBox[1] }, { inBox[2], inBox[3] }}, outLimits);
}

/**
*	Calculus Interval for ChungReynolds function on CPU
*	@param inbox pointer to Box
*	@param inRank number of variables
*	@param outlimits pointer to estimated function limits
*/
void calcFunBoundsChungReynoldsWithLib(const double *inBox, const int inRank, double *outLimits)
{
   static auto expr= ChungReynolds<double>(2);
    calcFunc("ChungReynolds", expr, {(inBox[0]+inBox[1])/2, (inBox[2]+inBox[3])/2},outLimits);

   static auto exprInterval = ChungReynolds<Interval<double>>(2);
    calcInterval("ChungReynolds estimation", exprInterval, { { inBox[0], inBox[1] }, { inBox[2], inBox[3] }}, outLimits);
}


/**
*	Calculus Interval for Colville function on CPU
*	@param inbox pointer to Box
*	@param inRank number of variables
*	@param outlimits pointer to estimated function limits
*/
void calcFunBoundsColvilleWithLib(const double *inBox, const int inRank, double *outLimits)
{
   static auto expr= Colville<double>();
    calcFunc("Colville", expr, {(inBox[0]+inBox[1])/2, (inBox[2]+inBox[3])/2, (inBox[4]+inBox[5])/2, (inBox[6]+inBox[7])/2},outLimits);

   static auto exprInterval = Colville<Interval<double>>();
    calcInterval("Colville estimation", exprInterval, { { inBox[0], inBox[1] }, { inBox[2], inBox[3] }, { inBox[4], inBox[5] }, { inBox[6], inBox[7] }}, outLimits);
}

/**
*	Calculus Interval for Complex function on CPU
*	@param inbox pointer to Box
*	@param inRank number of variables
*	@param outlimits pointer to estimated function limits
*/
void calcFunBoundsComplexWithLib(const double *inBox, const int inRank, double *outLimits)
{
   static auto expr= Complex<double>();
    calcFunc("Complex", expr, {(inBox[0]+inBox[1])/2, (inBox[2]+inBox[3])/2},outLimits);

   static auto exprInterval = Complex<Interval<double>>();
    calcInterval("Complex estimation", exprInterval, { { inBox[0], inBox[1] }, { inBox[2], inBox[3] }}, outLimits);
}

/**
*	Calculus Interval for CosineMixture function on CPU
*	@param inbox pointer to Box
*	@param inRank number of variables
*	@param outlimits pointer to estimated function limits
*/
void calcFunBoundsCosineMixtureWithLib(const double *inBox, const int inRank, double *outLimits)
{
   static auto expr= CosineMixture<double>();
    calcFunc("CosineMixture", expr, {(inBox[0]+inBox[1])/2, (inBox[2]+inBox[3])/2},outLimits);

   static auto exprInterval = CosineMixture<Interval<double>>();
    calcInterval("CosineMixture estimation", exprInterval, { { inBox[0], inBox[1] }, { inBox[2], inBox[3] }}, outLimits);
}

/**
*	Calculus Interval for CrossInTray function on CPU
*	@param inbox pointer to Box
*	@param inRank number of variables
*	@param outlimits pointer to estimated function limits
*/
void calcFunBoundsCrossInTrayWithLib(const double *inBox, const int inRank, double *outLimits)
{
   static auto expr= CrossInTray<double>();
    calcFunc("CrossInTray", expr, {(inBox[0]+inBox[1])/2, (inBox[2]+inBox[3])/2},outLimits);

   static auto exprInterval = CrossInTray<Interval<double>>();
    calcInterval("CrossInTray estimation", exprInterval, { { inBox[0], inBox[1] }, { inBox[2], inBox[3] }}, outLimits);
}

/**
*	Calculus Interval for CrossLeg function on CPU
*	@param inbox pointer to Box
*	@param inRank number of variables
*	@param outlimits pointer to estimated function limits
*/
void calcFunBoundsCrossLegWithLib(const double *inBox, const int inRank, double *outLimits)
{
   static auto expr= CrossLeg<double>();
    calcFunc("CrossLeg", expr, {(inBox[0]+inBox[1])/2, (inBox[2]+inBox[3])/2},outLimits);

   static auto exprInterval = CrossLeg<Interval<double>>();
    calcInterval("CrossLeg estimation", exprInterval, { { inBox[0], inBox[1] }, { inBox[2], inBox[3] }}, outLimits);
}

/**
*	Calculus Interval for Cube function on CPU
*	@param inbox pointer to Box
*	@param inRank number of variables
*	@param outlimits pointer to estimated function limits
*/
void calcFunBoundsCubeWithLib(const double *inBox, const int inRank, double *outLimits)
{
   static auto expr= Cube<double>();
    calcFunc("Cube", expr, {(inBox[0]+inBox[1])/2, (inBox[2]+inBox[3])/2},outLimits);

   static auto exprInterval = Cube<Interval<double>>();
    calcInterval("Cube estimation", exprInterval, { { inBox[0], inBox[1] }, { inBox[2], inBox[3] }}, outLimits);
}

/**
*	Calculus Interval for Deb1 function on CPU
*	@param inbox pointer to Box
*	@param inRank number of variables
*	@param outlimits pointer to estimated function limits
*/
void calcFunBoundsDeb1WithLib(const double *inBox, const int inRank, double *outLimits)
{
   static auto expr= Deb1<double>(2);
    calcFunc("Deb1", expr, {(inBox[0]+inBox[1])/2, (inBox[2]+inBox[3])/2},outLimits);

   static auto exprInterval = Deb1<Interval<double>>(2);
    calcInterval("Deb1 estimation", exprInterval, { { inBox[0], inBox[1] }, { inBox[2], inBox[3] }}, outLimits);
}

/**
*	Calculus Interval for Davis function on CPU
*	@param inbox pointer to Box
*	@param inRank number of variables
*	@param outlimits pointer to estimated function limits
*/
void calcFunBoundsDavisWithLib(const double *inBox, const int inRank, double *outLimits)
{
   static auto expr= Davis<double>();
    calcFunc("Davis", expr, {(inBox[0]+inBox[1])/2, (inBox[2]+inBox[3])/2},outLimits);

   static auto exprInterval = Davis<Interval<double>>();
    calcInterval("Davis estimation", exprInterval, { { inBox[0], inBox[1] }, { inBox[2], inBox[3] }}, outLimits);
}

/**
*	Calculus Interval for DeckkersAarts function on CPU
*	@param inbox pointer to Box
*	@param inRank number of variables
*	@param outlimits pointer to estimated function limits
*/
void calcFunBoundsDeckkersAartsWithLib(const double *inBox, const int inRank, double *outLimits)
{
   static auto expr= DeckkersAarts<double>();
    calcFunc("DeckkersAarts", expr, {(inBox[0]+inBox[1])/2, (inBox[2]+inBox[3])/2},outLimits);

   static auto exprInterval = DeckkersAarts<Interval<double>>();
    calcInterval("DeckkersAarts estimation", exprInterval, { { inBox[0], inBox[1] }, { inBox[2], inBox[3] }}, outLimits);
}

/**
*	Calculus Interval for DixonPrice function on CPU
*	@param inbox pointer to Box
*	@param inRank number of variables
*	@param outlimits pointer to estimated function limits
*/
void calcFunBoundsDixonPriceWithLib(const double *inBox, const int inRank, double *outLimits)
{
   static auto expr= DixonPrice<double>();
    calcFunc("DixonPrice", expr, {(inBox[0]+inBox[1])/2, (inBox[2]+inBox[3])/2},outLimits);

   static auto exprInterval = DixonPrice<Interval<double>>();
    calcInterval("DixonPrice estimation", exprInterval, { { inBox[0], inBox[1] }, { inBox[2], inBox[3] }}, outLimits);
}

/**
*	Calculus Interval for Dolan function on CPU
*	@param inbox pointer to Box
*	@param inRank number of variables
*	@param outlimits pointer to estimated function limits
*/
void calcFunBoundsDolanWithLib(const double *inBox, const int inRank, double *outLimits)
{
   static auto expr= Dolan<double>();
    calcFunc("Dolan", expr, {(inBox[0]+inBox[1])/2, (inBox[2]+inBox[3])/2,(inBox[4]+inBox[5])/2,(inBox[6]+inBox[7])/2,(inBox[8]+inBox[9])/2},outLimits);

   static auto exprInterval = Dolan<Interval<double>>();
    calcInterval("Dolan estimation", exprInterval, { { inBox[0], inBox[1] }, { inBox[2], inBox[3] },{ inBox[4], inBox[5] },{ inBox[6], inBox[7] },{ inBox[8], inBox[9] }}, outLimits);
}

/**
*	Calculus Interval for DropWave function on CPU
*	@param inbox pointer to Box
*	@param inRank number of variables
*	@param outlimits pointer to estimated function limits
*/
void calcFunBoundsDropWaveWithLib(const double *inBox, const int inRank, double *outLimits)
{
   static auto expr= DropWave<double>();
    calcFunc("DropWave", expr, {(inBox[0]+inBox[1])/2, (inBox[2]+inBox[3])/2},outLimits);

   static auto exprInterval = DropWave<Interval<double>>();
    calcInterval("DropWave estimation", exprInterval, { { inBox[0], inBox[1] }, { inBox[2], inBox[3] }}, outLimits);
}

/**
*	Calculus Interval for Rosenbrock function on CPU. Dimension 5
*	@param inbox pointer to Box
*	@param inRank number of variables
*	@param outlimits pointer to estimated function limits
*/
void calcFunBoundsRosenbrockWithLib5(const double *inBox, const int inRank, double *outLimits)
{
   static auto expr= Rosenbrock<double>(inRank);

    calcFunc("Rosenbrock", expr, {(inBox[0]+inBox[1])/2, (inBox[2]+inBox[3])/2, (inBox[4]+inBox[5])/2, (inBox[6]+inBox[7])/2, (inBox[8]+inBox[9])/2},outLimits);

   static auto exprInterval = Rosenbrock<Interval<double>>(inRank);
    calcInterval("Rosenbrock estimation", exprInterval, { { inBox[0], inBox[1] }, { inBox[2], inBox[3] }, { inBox[4], inBox[5] }, { inBox[6], inBox[7] }, { inBox[8], inBox[9] }}, outLimits);
}

/**
*	Calculus Interval for Rosenbrock function on CPU. Dimention 10
*	@param inbox pointer to Box
*	@param inRank number of variables
*	@param outlimits pointer to estimated function limits
*/
void calcFunBoundsRosenbrockWithLib10(const double *inBox, const int inRank, double *outLimits)
{
   static auto expr= Rosenbrock<double>(inRank);

    calcFunc("Rosenbrock", expr, {(inBox[0]+inBox[1])/2, (inBox[2]+inBox[3])/2, (inBox[4]+inBox[5])/2, (inBox[6]+inBox[7])/2, (inBox[8]+inBox[9])/2,(inBox[10]+inBox[11])/2, (inBox[12]+inBox[13])/2, (inBox[14]+inBox[15])/2, (inBox[16]+inBox[17])/2, (inBox[18]+inBox[19])/2},outLimits);

   static auto exprInterval = Rosenbrock<Interval<double>>(inRank);
    calcInterval("Rosenbrock estimation", exprInterval, { { inBox[0], inBox[1] }, { inBox[2], inBox[3] }, { inBox[4], inBox[5] }, { inBox[6], inBox[7] }, { inBox[8], inBox[9] },{ inBox[10], inBox[11] }, { inBox[12], inBox[13] }, { inBox[14], inBox[15] }, { inBox[16], inBox[17] }, { inBox[18], inBox[19] }}, outLimits);
}

/**
*	Calculus Interval for Rosenbrock function on CPU. Dimention 15
*	@param inbox pointer to Box
*	@param inRank number of variables
*	@param outlimits pointer to estimated function limits
*/
void calcFunBoundsRosenbrockWithLib15(const double *inBox, const int inRank, double *outLimits)
{
   static auto expr= Rosenbrock<double>(inRank);

    calcFunc("Rosenbrock", expr, {(inBox[0]+inBox[1])/2, (inBox[2]+inBox[3])/2, (inBox[4]+inBox[5])/2, (inBox[6]+inBox[7])/2, (inBox[8]+inBox[9])/2,(inBox[10]+inBox[11])/2, (inBox[12]+inBox[13])/2, (inBox[14]+inBox[15])/2, (inBox[16]+inBox[17])/2, (inBox[18]+inBox[19])/2, (inBox[20]+inBox[21])/2, (inBox[22]+inBox[23])/2, (inBox[24]+inBox[25])/2, (inBox[26]+inBox[27])/2, (inBox[28]+inBox[29])/2},outLimits);

   static auto exprInterval = Rosenbrock<Interval<double>>(inRank);
    calcInterval("Rosenbrock estimation", exprInterval, { { inBox[0], inBox[1] }, { inBox[2], inBox[3] }, { inBox[4], inBox[5] }, { inBox[6], inBox[7] }, { inBox[8], inBox[9] },{ inBox[10], inBox[11] }, { inBox[12], inBox[13] }, { inBox[14], inBox[15] }, { inBox[16], inBox[17] }, { inBox[18], inBox[19] },{ inBox[20], inBox[21] }, { inBox[22], inBox[23] }, { inBox[24], inBox[25] }, { inBox[26], inBox[27] }, { inBox[28], inBox[29] }}, outLimits);
}

/**
*	Calculus Interval for Rosenbrock function on CPU. Dimention 20
*	@param inbox pointer to Box
*	@param inRank number of variables
*	@param outlimits pointer to estimated function limits
*/
void calcFunBoundsRosenbrockWithLib20(const double *inBox, const int inRank, double *outLimits)
{
   static auto expr= Rosenbrock<double>(inRank);

    calcFunc("Rosenbrock", expr, {(inBox[0]+inBox[1])/2, (inBox[2]+inBox[3])/2, (inBox[4]+inBox[5])/2, (inBox[6]+inBox[7])/2, (inBox[8]+inBox[9])/2,(inBox[10]+inBox[11])/2, (inBox[12]+inBox[13])/2, (inBox[14]+inBox[15])/2, (inBox[16]+inBox[17])/2, (inBox[18]+inBox[19])/2, (inBox[20]+inBox[21])/2, (inBox[22]+inBox[23])/2, (inBox[24]+inBox[25])/2, (inBox[26]+inBox[27])/2, (inBox[28]+inBox[29])/2,(inBox[30]+inBox[31])/2, (inBox[32]+inBox[33])/2, (inBox[34]+inBox[35])/2, (inBox[36]+inBox[37])/2, (inBox[38]+inBox[39])/2},outLimits);

   static auto exprInterval = Rosenbrock<Interval<double>>(inRank);
    calcInterval("Rosenbrock estimation", exprInterval, { { inBox[0], inBox[1] }, { inBox[2], inBox[3] }, { inBox[4], inBox[5] }, { inBox[6], inBox[7] }, { inBox[8], inBox[9] },{ inBox[10], inBox[11] }, { inBox[12], inBox[13] }, { inBox[14], inBox[15] }, { inBox[16], inBox[17] }, { inBox[18], inBox[19] },{ inBox[20], inBox[21] }, { inBox[22], inBox[23] }, { inBox[24], inBox[25] }, { inBox[26], inBox[27] }, { inBox[28], inBox[29] },{ inBox[30], inBox[31] }, { inBox[32], inBox[33] }, { inBox[34], inBox[35] }, { inBox[36], inBox[37] }, { inBox[38], inBox[39] }}, outLimits);
}

/**
*	Calculus Interval for Modified Rosenbrock function on CPU
*	@param inbox pointer to Box
*	@param inRank number of variables
*	@param outlimits pointer to estimated function limits
*/
void calcFunBoundsRosenbrockModifiedWithLib(const double *inBox, const int inRank, double *outLimits)
{
   static auto expr= RosenbrockModified<double>();
    calcFunc("RosenbrockModified", expr, {(inBox[0]+inBox[1])/2, (inBox[2]+inBox[3])/2},outLimits);

   static auto exprInterval = RosenbrockModified<Interval<double>>();
    calcInterval("RosenbrockModified estimation", exprInterval, { { inBox[0], inBox[1] }, { inBox[2], inBox[3] }}, outLimits);
}
















#endif /* INCLUDE_TESTFUNCWITHLIB_H_ */
