///////////////////////////////////////////
// An abstraction of a nueral network of sigmoid "nuerons".
//	Network topology is described with m_pNetworkLayers and m_nLayerCount and implemented as a matrix
// See inline documentation for more info.
//
//TODO: Allow randomized or pre-specified param weights. 
//		templatize  
//
/// Author: Dustin Fast, June 2017

#pragma once
#include <vector>
#include <numeric>
#include <iostream>
#include "Sigmoid.h"
#include "SigmoidDataRow.h"

using namespace std;

class SigmoidNetwork
{
public:
	SigmoidNetwork(const int *networklayers, int layercount,								//Constructor
			double learningrate, double bias, double biaswt, bool verbose);								   

	void doTraining(const vector<SigmoidDataRow> &trainingset, int iterationcount);			//It does this iterationcount times.
	void propagateForward(const vector<double> &params);									//Start the process of the neuron firings, given input params, through hidden layers, to output layer.
	int getClassification(const vector<double> &params);									//Returns the neural network's output, given input params. The result is the index 
																							//   of the output neuron having the highest numeric result, from top to bottom.
	void printNeuronWeights();																//Outputs Neuron Weights

private:
	int m_nInputCount;																		//Number of inputs to each neuron.
	int m_nOutputCount;																		//Number of output layers in the network. 
	int m_nLayerCount;																		//Number of layers in network, including input layer. Should be m_pNetworkLayers.size()
	double m_dblLearningRate;
	bool m_bVerbose;																		//Verbose mode outputs the error per iteration to the console
	const int *m_pNetworkLayers;															//Array representation of the network layers.
																							//   Ex: {3, 4, 2} denotes 3 inputs, 1 hidden layer of 4 neuerons, and 2 output neurons
	double doLearn(double expectedresult, const vector<double> &params);					//Trains the sigmoid network, given input params and expected result
	void updateLayerInputs(int layerindex, const vector<double> &params);					//Function to set the neuron inputs to params in layer at index layerindex
	vector<double> getExpectedOutputVector(int expectedclassification);						//Returns a vector to be used to compare expected outputs per output neuron. 
	
	vector<vector<Sigmoid>> m_Network;														//A matrix representation of the neural network.
																							//   Ex: Given m_pNetwork = {3, 4, 2}, m_Network looks like:
																							//		I I I X
																							//      S S S S
																							//		S S X X
																							//   (Note that the I's are just placeholder objects used for convenience in indexing with m_pNetworkLayers. 
																							//		they are not actually referenced anywhere.

};

//Constructor
SigmoidNetwork::SigmoidNetwork(const int *networklayers, int layercount, double learningrate, double bias, double biaswt, bool verbose)
{
	//Ini vars
	m_nInputCount = networklayers[0];
	m_nOutputCount = networklayers[layercount - 1];
	m_nLayerCount = layercount;
	m_pNetworkLayers = networklayers;
	m_dblLearningRate = learningrate;
	m_bVerbose = verbose;
	//build network matrix representation
	vector<vector<Sigmoid>> t(m_nLayerCount);
	for (int i = 0; i < m_nLayerCount; i++)
	{
		for (int j = 0; j < m_pNetworkLayers[i]; j++)
		{
			int n = 0;
			if (i != 0)
				n = i - 1;
			Sigmoid s(networklayers[n], bias, biaswt);
			t[i].push_back(s);
		}
	}
	m_Network = t;
}

// Helper function for doLearn()
// Calls private doLearn(expectedresult, paramvector) for every row in data set. It does this iterationcount times
// outputs the RMS as calculated just before the last learning epoch
void SigmoidNetwork::doTraining(const vector<SigmoidDataRow> & trainingset, int iterationcount)
{
	if (iterationcount < 1 || trainingset.size() < 1)
		cout << "ERROR: Invalid iteration count or data set size.\n";
	else
	{
		double nError = 0;
		for (int i = 0; i < iterationcount; i++)
		{
			for (auto row : trainingset)
			{
				nError = doLearn(row.getExpectedResult(), row.getParams());
				if (nError == 0)
					cout << "Here";
			}
			if (m_bVerbose)
				cout << i + 1 << "," << nError << endl; //output epoch number and delta from the doLearn function.
		}
	}
}

//Adjust input weights via back propogation. The execution of this function constitutes one training epoch.
//Called from doTraining(). Returns output layer RMS error as it was calculated before weight adjustments.
double SigmoidNetwork::doLearn(double expectedresult, const vector<double> &params)
{
	double errorTotal = 0;	//RMS Error

	propagateForward(params);	//Start the process by doing a propagateForward through the network

	//Determine deltas for output layer neurons. 
	for (int i = 0; i < m_pNetworkLayers[m_nLayerCount - 1]; i++)  //iterate output layer neurons
	{
		double expOutput = getExpectedOutputVector((int)expectedresult)[i];
		double actOutput = m_Network[m_nLayerCount - 1][i].getOutput();
		m_Network[m_nLayerCount - 1][i].setNueronDelta(-(expOutput - actOutput) * actOutput * (1-actOutput));
		//errorTotal += sqrt((expOutput - actOutput) * (expOutput - actOutput));
		errorTotal += sqrt(m_Network[m_nLayerCount - 1][i].getNueronDelta() * m_Network[m_nLayerCount - 1][i].getNueronDelta());
	}
	
	//Determine deltas for hidden layer neurons, starting at rightmost hidden layer
	for (int i = m_nLayerCount - 2; i > 0; i--) //iterate hidden layers, r to l
	{
		for (int j = 0; j < m_pNetworkLayers[i]; j++) //iterate neurons
		{
			double deltaSum = 0.0;
			for (int k = 0; k < m_pNetworkLayers[i + 1]; k++) //iterate next layer's (to the right) neurons and sum the delta's multiplied by the param weight corresponding to the current neurons output
			{
				deltaSum += m_Network[i + 1][k].getNueronDelta() * m_Network[i + 1][k].getParamWeight(j);
			}
			double out = m_Network[i][j].getOutput();
			m_Network[i][j].setNueronDelta(deltaSum * out * (1-out));
		}
	}

	//Do weight corrections, excluding input layer
	for (int i = m_nLayerCount - 1; i > 0; i--) //iterate all layers r to l, excluding input layer
	{
		for (int j = 0; j < m_pNetworkLayers[i]; j++) // iterate neurons in layer
		{
			for (unsigned int k = 0; k < m_Network[i][j].getInputCount(); k++) //iterate inputs to the current layer and adjust their weight
				m_Network[i][j].updateParamWeight(k, m_dblLearningRate * m_Network[i][j].getNueronDelta() * m_Network[i][j].getParam(k));
		//bias weight correction
		m_Network[i][j].updateBiasWeight(m_dblLearningRate* m_Network[i][j].getNueronDelta());
		}
	}
	return errorTotal;
}

//Start the propogation of neuron outputs from Input layer to output layer
void SigmoidNetwork::propagateForward(const vector<double> &params)
{
	//send the input layer values to the first hidden layer
	updateLayerInputs(1, params); //Note layer 1, as in m_pNetworks[1], which is the first hidden layer.

	//for each layer (excluding the input layer) iterate through each neuron and calculate the result,
	// then set the inputs of each neuron in the next layer to the results
	for (int i = 1; i < m_nLayerCount; i++)
	{
		vector<double> vDblResults(m_pNetworkLayers[i], 0);

		for (int j = 0; j < m_pNetworkLayers[i]; j++)
			vDblResults[j] = m_Network[i][j].calculateResult();
		if (i + 1 < m_nLayerCount)
			updateLayerInputs(i + 1, vDblResults);
	}
}

//For given layer at index layer index, iterate through each neuron in that layer and set the input params for it. 
// When called from propagateForward, params is either the input params, or the results of a previous layer.
void SigmoidNetwork::updateLayerInputs(int layerindex, const vector<double> &params)
{
	for (int j = 0; j < m_pNetworkLayers[layerindex]; j++)
		m_Network[layerindex][j].setParams(params);
}

//Starts the process of getting a classification then returns the classifier's estimate
int SigmoidNetwork::getClassification(const vector<double> &params)
{
	propagateForward(params);
	double dblHigh = -1;
	int nResult = -1;
	//cycle through output-layer neurons and find the one with the highest output.
	// The index of the highest-value neuron is the index of our classification.
	// Ex: an nResult of 3 denotes a classification of D, because D's index in the alphabet is 3.
	for (int i = 0; i < m_pNetworkLayers[m_nLayerCount - 1]; i++)
	{
		double d = m_Network[m_nLayerCount - 1][i].getOutput();
		if (d > dblHigh)
		{
			nResult = i;
			dblHigh = d;
		}
	}
	return nResult;
}

//Returns a vector containing the expected oututs of the output layer, given an
// expected classification. We do it this way because we know, based on the 
// expected classification, that only one of the output neurons should be high.
// Thus, expectedclassification is the index of the output neuron we expect to be high.
vector<double> SigmoidNetwork::getExpectedOutputVector(int expectedclassification)
{
	vector<double> vDblResults(m_pNetworkLayers[m_nLayerCount - 1], .1); //.1 is the "pulled down" sigmoid output
	vDblResults[expectedclassification] = .9;	//.9 is the "pulled up" output

	return vDblResults;
}

//Output layer weights.
void SigmoidNetwork::printNeuronWeights()
{
	cout << "\nLayer Weights:\n";
	//for (int i = 0; i < m_nLayerCount; i++) //iterate all layers l to r, excluding input layer
	for (int i = m_nLayerCount -1; i < m_nLayerCount; i++) //iterate all layers l to r, excluding input layer
	{
		for (int j = 0; j < m_pNetworkLayers[i]; j++) // iterate neurons in layer
		{
			cout << "  Neuron " << j << ": ";
			for (unsigned int k = 0; k < m_Network[i][j].getInputCount(); k++) //iterate inputs to the current layer and output weight
				cout << m_Network[i][j].getParamWeight(k) << ", ";
			//output bias weight
			m_Network[i][j].updateBiasWeight(m_dblLearningRate* m_Network[i][j].getNueronDelta());
			cout << endl;
		}
	}

}


