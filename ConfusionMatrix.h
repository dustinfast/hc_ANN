///////////////////////////////////////////
// An abstraction of a confusion matrix.
//	m_Matrix[row][col] is the number of times item at row index "row" was guessed and item at column "col" was received. 
//  Each character in string "labels" is one index label. row labels = col labels. i.e. Matrix width = matrix height.  
//See inline documentation for more
//TODO: Impl. labels as  cellPlusOne row/col parameters 
//		outputMatrix alignment gets bad with more than a single digit in each cell.
//		void outputLabels(char delim);
//
/// Author: Dustin Fast, June 2017

#pragma once
#include <vector>
#include <iostream>

using namespace std;

class ConfusionMatrix
{
public:
	ConfusionMatrix(int width);						//Constructor
	ConfusionMatrix(int width, string labels);		//Constructor
	void cellPlusOne(int row, int col);				//Does m_Matrix[row][col]++. Functions as a way to count correct/incorrect classifications,
	void outputAccuracy();							//Outputs matrix accuracy by column
	void outputMatrix();							//Output entire matrix, including labels if set

private:
	int getIndex(int row, int col);					//Returns m_Matrix[row][col]
	void setIndex(int row, int col, int set);		//Updates m_Matrix[row][col] with given value 'set'
	string m_strLabels;								//Row and Col labels. Ex: [A,B,C,D,E,...,Z]
	vector<vector<int>> m_Matrix;
};

ConfusionMatrix::ConfusionMatrix(int width) : m_Matrix(width, vector<int>(width, 0))
{}
ConfusionMatrix::ConfusionMatrix(int width, string labels) : m_Matrix(width, vector<int>(width, 0)), m_strLabels(labels)
{}

//Returns m_Matrix[row][col]
int ConfusionMatrix::getIndex(int row, int col)
{
	return m_Matrix[row][col];
}

//updates m_Matrix[row][col] with given value 'set'
void ConfusionMatrix::setIndex(int row, int col, int set)
{
	m_Matrix[row][col] = set;
}

//Increment m_Matrix[row][col] by 1. This functions as a way to count correct/incorrect classifications,
//  because a correct classification implies that row == col
// ex: If A is expected but C is output of classifier, m_Matrix[0][2] is incremented by 1, counting an incorrect guess. 
//     if C is expected and C is output of classified, m_Matrix[2][2] is incremented, counting a correct guess.
void ConfusionMatrix::cellPlusOne(int row, int col)
{
	setIndex(row, col, getIndex(row, col) + 1);
}

//outputs the accuracy as a percentage for each col in CSV form
void ConfusionMatrix::outputAccuracy()
{
	if (m_strLabels.size() != 0) //print col headers if set, comma delim
	{
		for (unsigned int i = 0; i < m_strLabels.size(); i++)
			cout << m_strLabels[i] << ",";
		cout << endl;
	}

	for (unsigned int j = 0; j < m_Matrix.size(); j++) //iterate cols
	{
		double nTotal = 0;
		for (unsigned int i = 0; i < m_Matrix.size(); i++) //iterate rows
			nTotal = nTotal + m_Matrix[i][j]; //add up entire col
		//Compute accuracy as, ex: number of letter A instances classified correctly / (number of letter A instances classified correctly AND incorrectly) * 100
		if (nTotal == 0)
			cout << "N,";
		else
			cout << (m_Matrix[j][j] / nTotal * 100) << ",";
	}
}

//outputs the confusion matrix to the console in CSV form
void ConfusionMatrix::outputMatrix()
{
	if (m_strLabels.size() != 0) //print col headers if set, comma delim
	{
		cout << " ,";
		for (unsigned int i = 0; i < m_strLabels.size(); i++)
			cout << m_strLabels[i] << ",";
		cout << endl;
	}

	for (unsigned int i = 0; i < m_Matrix.size(); i++) 
	{
		if (m_strLabels.size() != 0) //print row headers, if set
			cout << m_strLabels[i] << ",";

		for (unsigned int j = 0; j < m_Matrix[0].size(); j++)
			cout << m_Matrix[i][j] << ",";
		cout << endl;
	}
}