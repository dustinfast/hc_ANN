///////////////////////////////////////////
// A data "row" of input for Sigmoid.h, in the form [label, (param_1, param_2, ... , param_n)]
//TODO: Templatize
//
/// Author: Dustin Fast, June 2017

#pragma once
#include <vector>

using namespace std;

class SigmoidDataRow
{
public:
	SigmoidDataRow(double expectedoutput, const vector<double> &params);	// Constructor

	double getExpectedResult();												// Returns the label
	vector<double>& getParams();											// Returns a ptr to the feature vector/params

private:
	double m_dblExpectedOutput;												// The correct classification.
	vector<double> m_vecParams;												// Input params
};

//Constructor
SigmoidDataRow::SigmoidDataRow(double expectedoutput, const vector<double> &params) : m_dblExpectedOutput(expectedoutput), m_vecParams(params)
{}

//Accessors
double SigmoidDataRow::getExpectedResult()
{
	return m_dblExpectedOutput;
}

vector<double>& SigmoidDataRow::getParams()
{
	return m_vecParams;
}
