/*******************************************************************
* a nice top running params on randering DS and NN wightes
* ------------------------------------------------------------------

********************************************************************/

#ifndef _HADER_TOP
#define _HADER_TOP


#include <MaxSLiCInterface.h>
#include "Maxfiles.h"




#define RAND_INIT_WIGHTS					(true)
#define RAND_SET_PREMUTATION_ONLY_AT_END_OF_SET (false)
#define PLOT_EPOCH_LOG_ACTIVETY				(false)
#define CMP_TO_RESULT_FROM_MATLAB			(false)
#define PLOT_ANN_OUTPUT						(false)
#define PLOT_DEBUG_MATRIX					(false)
#define DO_STATISTICS						(true)
#define PLOT_STATISTICS						(false)
#define EXPORT_STATISTICS_TO_FILE			(true)
#define CLAP_ANN_OUT_TO_BINARY				(false)
	#define CLAP_MAX_VALUE_FOR_0				(0.25)
	#define CLAP_MIN_VALUE_FOR_1				(0.75)



#define NUM_OF_EPOCHS_IN_SET 					(6*(128/DualKernel_EPOCH_SIZE))
#define REDO_SET_TIME 							(15000)
#define CORRECTION_LEARNING_RATE				(8.0)

// double = 64 bit, float = 32
typedef float NetVariableType; // universal use data type element for ANN calc


static int HIDDEN_LAYERS_NEURONS	=	DualKernel_HIDDEN_LAYERS_NEURONS;	// number of neurons in a single Hidden Layer
static int N_INPUTS 				= 	DualKernel_N_INPUTS;	// number of neurons in the Input Layer
static int N_OUTPUTS 				=	DualKernel_N_OUTPUTS;  // number of neurons in the Output Layer
static int NUM_OF_HIDDEN_LAYERS 	=	DualKernel_NUM_OF_HIDDEN_LAYERS; 	// total number of fully connected Hidden Layer







#define DO_random_shuffle_WHEN_LOADING_DS	(false)


#endif
