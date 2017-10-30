/*******************************************************************
* Neural Network Training Class
* ------------------------------------------------------------------
* Bobby Anguelov - takinginitiative.wordpress.com (2008)
* MSN & email: banguelov@cs.up.ac.za
*********************************************************************/

#ifndef NNetworkTrainer
#define NNetworkTrainer

//standard includes
#include <fstream>
#include <vector>

//neural network header
#include "neuralNetwork.h"

//Constant Defaults!
#define LEARNING_RATE 0.001
#define MOMENTUM 0.9
#define MAX_EPOCHS 1500
#define DESIRED_ACCURACY 90  
#define DESIRED_MSE 0.001 

#define TIME_DURATION_CHECK false
#include <cstdio>
#include <ctime>

/*******************************************************************
* Basic Gradient Descent Trainer with Momentum and Batch Learning wrapper for our NN
********************************************************************/
class neuralNetworkTrainer
{
	//class members
	//--------------------------------------------------------------------------------------------

private:

	//network to be trained
	neuralNetwork* NN;

	//learning parameters
	NetVariableType learningRate;					// adjusts the step size of the weight update	
	NetVariableType momentum;						// improves performance of stochastic learning (don't use for batch)

	//epoch counter
	long epoch;
	long maxEpochs;
	
	//accuracy/MSE required
	NetVariableType desiredAccuracy;
	
	//change to weights
	NetVariableType** deltaInputHidden;
	NetVariableType** deltaHiddenOutput;

	//error gradients
	NetVariableType* hiddenErrorGradients;
	NetVariableType* outputErrorGradients;

	//accuracy stats per epoch
	NetVariableType trainingSetAccuracy;
	NetVariableType validationSetAccuracy;
	NetVariableType generalizationSetAccuracy;
	NetVariableType trainingSetMSE;
	NetVariableType validationSetMSE;
	NetVariableType generalizationSetMSE;

	//batch learning flag
	bool useBatch;

	//log file handle
	bool loggingEnabled;
	std::fstream logFile;
	int logResolution;
	int lastEpochLogged;

	//public methods
	//--------------------------------------------------------------------------------------------
public:	
	
	neuralNetworkTrainer( neuralNetwork* untrainedNetwork );
	void setTrainingParameters( NetVariableType lR, NetVariableType m, bool batch );
	void setStoppingConditions( int mEpochs, NetVariableType dAccuracy);
	void useBatchLearning( bool flag ){ useBatch = flag; }
	void enableLogging( const char* filename, int resolution );

	void trainNetwork( trainingDataSet* tSet );

	//private methods
	//--------------------------------------------------------------------------------------------
private:
	inline NetVariableType getOutputErrorGradient( NetVariableType desiredValue, NetVariableType outputValue );
	NetVariableType getHiddenErrorGradient( int j );
	void runTrainingEpoch( std::vector<dataEntry*> trainingSet );
	void backpropagate(NetVariableType* desiredOutputs);
	void updateWeights();
};


#endif