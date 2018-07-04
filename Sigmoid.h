///////////////////////////////////////////
// An abstraction of a single nueral network sigmoid with a dynamic number of inputs of type double, a bias of type double, 
//	 and a single output of type double that may range from 0 to 1, non-inclusive.
//   Before output can be calculated, the input parameters must be set with setInput(params). After output calculation, output can
//	 be retrieved with getOutput() 
// See inline documentation for more info.
//
//TODO: Allow randomized pre-specified param weights. 
//		Adapt doTraining() to accept a SigmoidDataSet, rather than a vector of SigmoidDataRows
//		templatize  
//
/// Author: Dustin Fast, June 2017

#pragma once
#include <vector>
#include <numeric>
#include <iostream>
#include <cmath>
#include <ctime>
#include <ctgmath>
#include "SigmoidDataRow.h"

using namespace std;

class Sigmoid
{
public:
	Sigmoid(int paramcount, double bias, double biaswt);				//Constructor
	void setNueronDelta(double newdelta);								//Sets neuron delta.
	void updateBiasWeight(double newweight);							//Updates m_dblBiasWeight by adding the given delta to the existing bias wt
	void updateParamWeight(int index, double newweight);				//Updates m_vecWeights[i] by adding the given delta to the existing wt
	void setParams(const vector<double> &params);						//Set the inputs for the sigmoid. This will usually be the output of the previous layer
	double getNueronDelta();											//Returns m_dblNueronDelta
	double getParamWeight(int index);									//Returns m_VecWeights[i], the weight for given input at index
	double getParam(int index);											//Returns m_VecInputs[i], the param for given input at index
	double getBias();													//Returns m_dblBias
	double getBiasWeight();												//Returns m_dblBias
	unsigned int getInputCount();										//Returns the number of inputs (and by extension, weights) this neuron has
	double getOutput();													//Returns the pre-calculated output. Must call calculate result first to populate output
	double calculateResult();											//Sets (and returns) m_dblSigmoidResults, given input params

private:
	double m_dblBias;													//Neuron bias
	double m_dblBiasWeight;												//Neuron bias weight
	double m_dblSigmoidResult;											//Actual output of neuron. Set in calculateResults()
	double m_dblNueronDelta;											//Neuron delta value. Used by SigmoidNetwork during training.
	bool m_bParamsValid;												//True = m_vecDeblInputs is populated. False = inputs not set (i.e. setInput(params) has not been done)
	bool m_bOutputValid;												//True = m_dblSigmoid is valid. False = invalid (i.e. calculateResults() has not been done)
	vector<double> m_vecInputParams;									//Neuron param wt prv delta value, from last epoch. Used by SigmoidNetwork during training.
	vector<double> m_vecWeights;										//Dendrite (i.e. input) weights, where w_i = the weight of parameter_i

	double getVectorDotProduct(const vector<double> &v1, const vector<double> &v2);			// Utility function returning dot product of two vectors
};

//Constructors (Note: param wts initialized to .5 via the m_vecWeights assignment)
Sigmoid::Sigmoid(int paramcount, double bias, double biaswt) : m_vecWeights(paramcount),
m_vecInputParams(paramcount, 0), m_dblNueronDelta(0), m_dblBias(bias), m_dblBiasWeight(biaswt),
m_bParamsValid(false), m_bOutputValid(false), m_dblSigmoidResult(0)
{
	//randomize parameter weight biases
	for (int i = 0; i < paramcount; i++)
	{
		int j = 1;
		m_vecWeights[i] = ((double)(rand() % 10) + 1) / 10;
		if (m_vecWeights[i] > 0.5)
			m_vecWeights[i] *= -1;

	}
}

//Sets (and returns) m_dblSignmoidResults, which is the numeric output of the neuron, given input params. 
// Note: m_vecInputParams must be initialized first with setParams()
double Sigmoid::calculateResult()
{
	if (!m_bParamsValid)
	{
		cout << "ERROR: A sigmoid calculation was attempted on a neuron with no input data.";
		return m_dblSigmoidResult;
	}
		
	double dblResult = getVectorDotProduct(m_vecInputParams, m_vecWeights) + (m_dblBias * m_dblBiasWeight); //Summation of all params (including bias)
	m_dblSigmoidResult =  1.f / (1.f + exp(-dblResult));		//Sigmoid function
	m_bOutputValid = true;

	return m_dblSigmoidResult;
}

//Returns dot product of vectors v1 and v2.
double Sigmoid::getVectorDotProduct(const vector<double> &v1, const vector<double> &v2)
{
	double dblResult = 0;

	for (unsigned int i = 0; i < v1.size(); i++)
		dblResult += v1[i] * v2[i];

	return dblResult;
}

///Accessors
//Gets the pre-calculuated output. Calling this function before calling calculateResult() will result in the error msg.
double Sigmoid::getOutput()
{
	if (!m_bOutputValid)
		cout << "ERROR: A sigmoid output was requested from a neuron who's output has not been calculated.\n;";
	return m_dblSigmoidResult;
}
//Returns the input param for input "index. Calling this function before 
double Sigmoid::getParam(int index)
{
	if (!m_bParamsValid)
		cout << "ERROR: A sigmoid input parameter was requrested from a neuron with no input data.\n";

	return m_vecInputParams[index];
}
double Sigmoid::getBias()
{
	return m_dblBias;
}
double Sigmoid::getBiasWeight()
{
	return m_dblBiasWeight;
}
unsigned int Sigmoid::getInputCount()
{
	return m_vecInputParams.size();
}
double Sigmoid::getNueronDelta()
{
	return m_dblNueronDelta;
}
double Sigmoid::getParamWeight(int index)
{
	return m_vecWeights[index];
}

///Mutators
void Sigmoid::setParams(const vector<double> &params) //Sets the input params
{
	m_vecInputParams = params;
	m_bParamsValid = true;
}
void Sigmoid::setNueronDelta(double newdelta)
{
	m_dblNueronDelta = newdelta;
}
void Sigmoid::updateParamWeight(int index, double neweight)
{
	m_vecWeights[index] -= neweight;
}
void Sigmoid::updateBiasWeight(double newweight)
{
	m_dblBiasWeight -= newweight;
}






