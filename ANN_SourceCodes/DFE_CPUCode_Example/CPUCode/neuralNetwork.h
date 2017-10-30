/*******************************************************************
* Basic Feed Forward Neural Network Class
* ------------------------------------------------------------------
* Bobby Anguelov - takinginitiative.wordpress.com (2008)
* MSN & email: banguelov@cs.up.ac.za
********************************************************************/

#ifndef NNetwork
#define NNetwork

#include "dataReader.h"

class neuralNetworkTrainer;

class neuralNetwork
{
	//class members
	//--------------------------------------------------------------------------------------------
private:
	//number of neurons
	int nInput, nHidden, nOutput;
	
	//neurons
	NetVariableType* inputNeurons;
	NetVariableType* hiddenNeurons;
	NetVariableType* outputNeurons;

	//weights
	NetVariableType** wInputHidden;
	NetVariableType** wHiddenOutput;
		
	//Friends
	//--------------------------------------------------------------------------------------------
	friend class neuralNetworkTrainer;
	
	//public methods
	//--------------------------------------------------------------------------------------------

public:

	//constructor & destructor
	neuralNetwork(int numInput, int numHidden, int numOutput);
	~neuralNetwork();

	//weight operations
	bool loadWeights(char* inputFilename);
	bool saveWeights(char* outputFilename);
	int* feedForwardPattern( NetVariableType* pattern );
	NetVariableType getSetAccuracy( std::vector<dataEntry*>& set );
	NetVariableType getSetMSE( std::vector<dataEntry*>& set );

	//private methods
	//--------------------------------------------------------------------------------------------

//dogev private:

	void initializeWeights();
	NetVariableType activationFunction( NetVariableType x );
	int clampOutput( NetVariableType x );
	void feedForward( NetVariableType* pattern );
	
};

#endif
