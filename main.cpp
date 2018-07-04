///////////////////////////////////////////
// Demonstration of a multi-layer sigmoid (Sigmoid.h) neural network (SigmoidNetwork.h)
// The sigmoid is trained and validated for each combination of Learning Rate
// and Learning Iterations specified (in constants). Results are output as a
// confusion matrix.
// 
// See welcome msg, output, and inline documentation for more info.
//
//
/// Author: Dustin Fast, June 2017

#include <vector>
#include <iostream>
#include <cassert>
#include <fstream>
#include <string>
#include <ctime>
#include "SigmoidNetwork.h"
#include "ConfusionMatrix.h"

	using namespace std;

vector<vector<string>> populateVectorFromDataFile(string filename, char delimeter); //Utility function to populate a vector from a char delimited data file.
double rescaleFeature(double x, double minx, double maxx); //Utility function to rescale a feature variable x as x' = (x-min(x))/max(x)-min(x)

const string WELCOME_MSG = "\nSigmoid\n-------------------------------------------------------------------\n"
"This tool trains a multi-layer sigmpoid network from pre-specified training data, learning rate (LR), and\n "
"learning iterations. The sigmoid is then validated with pre-specified validation data.\n\n"
"data set obtained from https://archive.ics.uci.edu/ml/machine-learning-databases/letter-recognition/letter-recognition.data\n"
"Output is in CSV form - pipe output to csv for best results.";

const int LEARNING_ITERATIONS[] = { 1 };								//Iterations to iterate through
const int ITERATIONS_COUNT = 1;											//Size of LEARNING_ITERATIONS
const double LEARNING_RATE[] = { 0.01 };								//LR's to iterate through
const int RATE_COUNT = 1;												//Size of LEARNING_RATE
const double BIAS = -1;													//Bias of each neuron
const double BIAS_WEIGHT = 0.5;											//Initial bias Weight of each neuron
const int NETWORK_LAYERS[] = { 16, 14, 26 };							//Network Structure. Ex: {3, 4, 2} denotes 3 input layers, 1 hidden layer of 4 neuerons, and 2 output neurons
const int NETWORK_LAYER_COUNT = 3;										//Total number of network layers. Ex {3, 4, 2] = 3 layers. Will be size of NETWORK LAYERS
const bool VERBOSE = true;												//Verbose mode outputs the error per training iteration to the console
const string TRAINING_DATAFILE = "dataset/letter-recognition.train.data";
const string VALIDATION_DATAFILE = "dataset/letter-recognition.val.data";

int main()
{
	string strRestart = "y";
	while (strRestart == "y")
	{

		//Welcome Msg and menu
		////////////////////////////
		strRestart = "";  //Reset menu loop flag and display info txt.
		cout << WELCOME_MSG;
		cout << "\n\nTraining data is " << TRAINING_DATAFILE << ".\n";
		cout << "Validation data is " << VALIDATION_DATAFILE << ".\n\n";
		cout << "A new sigmoid will be trained " << ITERATIONS_COUNT << " time(s) for each learning rate of:\n";
		for (int i_rate = 0; i_rate < RATE_COUNT; i_rate++)
			cout << "   " << LEARNING_RATE[i_rate];
		cout << "\nFor each learning rate, the training routine will be iterated the following number of times:\n";
		for (int i_iters = 0; i_iters < ITERATIONS_COUNT; i_iters++)
			cout << "   " << LEARNING_ITERATIONS[i_iters];
		cout << "\nThese processes will take some time to complete.\n\n";

		//Read data from file into vectors. Vectors won't be in correct format, we do that in the 'converting' step
		//////////////////////////////////////
		vector<vector<string>> vPreDataTrain;		//Training data set, 
		vector<vector<string>> vPreDataValidate;	//Validation data set
		cout << "Reading Training Data...\n";
		vPreDataTrain = populateVectorFromDataFile(TRAINING_DATAFILE, ',');
		if (vPreDataTrain.size() <= 0)
			continue;
		cout << "Done.\nReading Validation Data...\n";
		vPreDataValidate = populateVectorFromDataFile(VALIDATION_DATAFILE, ',');
		if (vPreDataValidate.size() <= 0)
			continue;
		cout << "Done.\n";

		//Convert data to sigmoid data for both data sets, rescaling the features as we go.
		////////////////////////
		string strAlphaIndex = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"; 		//used for geting the index of the expected alphabetic character
																	//  also used for labeling confusion matrix rows/cols		
		unsigned int nRowsTrain = vPreDataTrain.size(), nRowsValidate = vPreDataValidate.size();			//Number of rows of data
		unsigned int nParamsTrain = vPreDataTrain[0].size(), nParamsValidate = vPreDataValidate[0].size();	//Size of vector containg data row, including expected result at index 0
		vector<double> vMinx(nParamsTrain - 1, 99999);				//min x values of all params, by column, excluding label
		vector<double> vMaxx(nParamsTrain - 1, -1);					//max x values of all params, by column, excluding label
		vector<SigmoidDataRow> vDataTrain;  						//Training Data in form sigmoid wants: (y, vector<double>)
		vector<SigmoidDataRow> vDataValidate; 						//Validation data in form sigmoid wants (y, vector<double>)

		cout << "Determining paramerter ranges...\n";
		//Get min and max feature values from training set to be used for feature rescaling. 
		for (unsigned int i = 1; i < nParamsTrain; i++) //iterate parameter, excluding label at index 0
		{
			for (unsigned int j = 0; j < nRowsTrain; j++) //iterate each row
			{
				double d = stod(vPreDataTrain[j][i]);
				if (d > vMaxx[i - 1])			//store maxx to maxx vector. Index is one less than the index of the corresponding parameter in vDataTrain/vDataVal
					vMaxx[i - 1] = d;
				if (d < vMinx[i - 1])			//store minx to minx vector
					vMinx[i - 1] = d;
			}
		}
		//Also check validation set for min/max
		for (unsigned int i = 1; i < nParamsValidate; i++) //iterate parameter, excluding expected value param at index 0
		{
			for (unsigned int j = 0; j < nRowsValidate; j++) //iterate each row
			{
				double d = stod(vPreDataValidate[j][i]);
				if (d > vMaxx[i - 1])			//store maxx to maxx vector. Index is one less than the index of the corresponding parameter in vDataTrain/vDataVal
					vMaxx[i - 1] = d;
				if (d < vMinx[i - 1])			//store minx to minx vector
					vMinx[i - 1] = d;
			}
		}
		cout << "Done.\n";
		cout << "Converting Training Vector to Sigmoid Data...\n";

		for (unsigned int i = 0; i < nRowsTrain; i++)
		{
			//Push data to vector as a row in needed format of (y, vector<double>)
			//First item is expected result. We do not rescale it, but convert it to an index in strAlphaIndex.
			//  i.e. our label is now a number value from 0-25, each corresponding to the OutputLayerNode index
			//  ex: Letter C is converted to 2, and 2 is the index of the output layer node that will be high when
			//  a given feature set is evaluated to be C.
			double y = strAlphaIndex.find_first_of(vPreDataTrain[i][0]);
			vector<double> t;
			for (unsigned int j = 1; j < nParamsTrain; j++) //next items are parameters. Rescale and add to feature vector
				t.push_back(rescaleFeature(stod(vPreDataTrain[i][j]), vMinx[j - 1], vMaxx[j - 1]));
			SigmoidDataRow dTemp(y, t); //Create sigmoid data row from label and feature vector
			vDataTrain.push_back(dTemp); //add the row to our finalized training data vector
		}
		cout << "Done.\n";
		cout << "Converting Validation Vector to Sigmoid Data...\n";
		for (unsigned int i = 0; i < nRowsValidate; i++)
		{
			//push data to vector as a row in needed format described above.
			double y = strAlphaIndex.find_first_of(vPreDataValidate[i][0]);  //first item is expected result
			vector<double> t;
			for (unsigned int j = 1; j < nParamsValidate; j++) //next items are parameters
				t.push_back(rescaleFeature(stod(vPreDataValidate[i][j]), vMinx[j - 1], vMaxx[j - 1]));
			SigmoidDataRow dTemp(y, t);
			vDataValidate.push_back(dTemp);
		}
		cout << "Done.\n";
		if (nParamsTrain != nParamsValidate)
		{
			cout << "ERROR: Differing parameter counts between training and validation sets.\n";
			continue;
		}

		//Train and validate with the given constants LEARNING_RATE and LEARNING_ITERATIONS
		for (int i_iters = 0; i_iters < ITERATIONS_COUNT; i_iters++)
		{
			for (int i_rate = 0; i_rate < RATE_COUNT; i_rate++)
			{
				//Build Sigmoid Network
				cout << "Operating on Sigmoid Network with " << NETWORK_LAYERS[0] << " inputs " << NETWORK_LAYER_COUNT - 2 << " hidden layer(s), and " << NETWORK_LAYERS[NETWORK_LAYER_COUNT - 1] << " outputs.\n";
				srand(time(NULL));
				SigmoidNetwork sNetwork(NETWORK_LAYERS, NETWORK_LAYER_COUNT, LEARNING_RATE[i_rate], BIAS, BIAS_WEIGHT, VERBOSE);

				//Pre-Validate Sigmoid, to see success rate before training
				//cout << "Pre-Validating Sigmoid...\n";
				//ConfusionMatrix m(26, strAlphaIndex);
				//for (unsigned int i = 0; i < vDataValidate.size(); i++)
				//	m.cellPlusOne((int)vDataValidate[i].getExpectedResult(), sNetwork.getClassification(vDataValidate[i].getParams()));
				//cout << "Results: (LR = " << LEARNING_RATE[i_rate] << " Iterations = " << LEARNING_ITERATIONS[i_iters] << "),\n";
				//m.outputMatrix();
				//cout << "\nAccuracy: (LR = " << LEARNING_RATE[i_rate] << " Iterations = " << LEARNING_ITERATIONS[i_iters] << ")\n";
				//m.outputAccuracy();
				//cout << endl;
				//sNetwork.printNeuronWeights();
				//cout << endl << endl;


				//Train Sigmoid Network
				cout << "Training Sigmoid Network (LR = " << LEARNING_RATE[i_rate] << " Iterations = " << LEARNING_ITERATIONS[i_iters] << ")...\n";
				sNetwork.doTraining(vDataTrain, LEARNING_ITERATIONS[i_iters]);
				cout << "Done.\n";
				//cout << endl;
				//sNetwork.printNeuronWeights();
				cout << endl << endl;

				//Validate Sigmoid 
				cout << "Validating Sigmoid...\n";
				ConfusionMatrix m(26, strAlphaIndex);	//create confusion matrix using strAlphaIndex as row/col labels
														//Here we update the confusion matrix with our classifications. We do this by incrementing m_Matrix[ExpectedResult][ActualResult].
														// ex: If A is expected but C is output of classifier, m_Matrix[0][2] is incremented by 1, counting an incorrect guess. 
														//     if C is expected and C is output of classified, m_Matrix[2][2] is incremented, counting a correct guess.
				for (unsigned int i = 0; i < vDataValidate.size(); i++)
					m.cellPlusOne((int)vDataValidate[i].getExpectedResult(), sNetwork.getClassification(vDataValidate[i].getParams()));
				cout << "Results: (LR = " << LEARNING_RATE[i_rate] << " Iterations = " << LEARNING_ITERATIONS[i_iters] << ")\n";
				m.outputMatrix();
				cout << "\nAccuracy: (LR = " << LEARNING_RATE[i_rate] << " Iterations = " << LEARNING_ITERATIONS[i_iters] << ")\n";
				m.outputAccuracy();
				//cout << endl;						//debug
				//sNetwork.printNeuronWeights();	//debug
				cout << endl << endl;
			}
		}
		cout << "Enter any key to exit, or 'y' to restart:\n";
		cin >> strRestart;
	} //while strRestart == 'y'

	return 0;
}


//Utility function to populate a vector<vector<string>> from a char delimeted data file
// Accepts: strFilename = data file to be opened.
//          chDelimeter = char to use as delimter. Ex CSV would have chDelimeter = ','
// Returns: A populated vector<vector<string>>
vector<vector<string>> populateVectorFromDataFile(string filename, char delimeter)
{
	vector<vector<string>> vData;

	//Open data file
	ifstream fsIn;
	fsIn.open(filename);
	if (fsIn.fail())
	{
		cout << "Error: File could not be opened.\n";
		return vData;
	}

	//Populate vector array with data
	string strTemp;
	while (getline(fsIn, strTemp, '\n')) //iterate file by line
	{
		//create new vector "row"
		vector<string> v;
		vData.push_back(v);

		//explode strTemp on chDelimeter and push values to new vector "row"
		string str = "";
		for (unsigned int i = 0; i < strTemp.length(); i++)
		{
			if (strTemp[i] != delimeter)
				str = str + strTemp[i];
			else
			{
				try
				{
					vData[vData.size() - 1].push_back(str);
				}
				catch (const invalid_argument& e)
				{
					cout << "ERROR: Data file contains an invalid data type.\n";
					return vData;
				}
				str = "";
			}
		}
		vData[vData.size() - 1].push_back(str);
	}

	//validate file size
	if (vData.size() <= 0)
		cout << "ERROR: File contained no data.\n\n";
	return vData;
}

//Utility function to rescale a feature variable x as x' = (x-min(x))/max(x)-min(x)
double rescaleFeature(double x, double minx, double maxx)
{
	if (maxx == minx) //if there is only one data row
		return x;
	else
		return (x - minx) / (maxx - minx);
}
