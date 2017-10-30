/**	==================================	Main Description:
*	Description:		The Manager connectin a fully associative ANN
*	Designed By:		Yogev Dotan, Dor Rosenblum
*	Revision History:	Date		Ver Num		Change
*						2.10.2016	0.1			the CPU host code, calling the DFE ANN
*						20.10.2016	0.2			Non Blocking DFE ANN
*	====================================	*/

#include <stdlib.h>
#include <cmath>
#include <limits>
#include <stdint.h>
#include <cstring>

#include <time.h>

#include <iostream>
#include <string>
#include <iostream>
#include <stdlib.h>
#include <sstream>
#include <fstream>
#include <vector>
#include <iomanip>
//#include <chrono>

//using namespace std::chrono;

using std::string;
using std::ifstream;
using std::stringstream;
using std::vector;
using std::setw;


#include <MaxSLiCInterface.h>
#include "Maxfiles.h"

//standard libraries
#include <iostream>
#include <ctime>

//custom includes
#include "neuralNetwork.h"
#include "neuralNetworkTrainer.h"

//use standard namespace
using namespace std;

#define EPSILON 				1000*(std::numeric_limits<float>::epsilon())

#define PLOT_EVEN_SPACE_NAME 	std::left << std::setfill(' ') << setw(30)
#define PLOT_EVEN_SPACE_EQULS 	std::left << std::setfill(' ') << setw(10)
#define PLOT_EVEN_SPACE_INT 	std::left << std::setfill(' ') << setw(5)
#define PLOT_EVEN_SPACE_FLOAT 	std::left << std::setfill(' ') << setw(10)





/*template<typename T> std::string to_string(const T& n) {
	std::ostringstream stm;
	stm << n;
	return stm.str();
}*/
template < typename T > std::string to_string( const T& n )
{
    std::stringstream stm ;
    stm << n ;
    return stm.str() ;
}


void 			ann() {
	//seed random number generator
	srand((unsigned int) time(0));

	//create data set reader and load data file
	dataReader d;
	d.loadDataFile("vowel-recognition.csv", 16, 10);
	d.setCreationApproach(STATIC, 20);

	//create neural network
	neuralNetwork nn(16, 19, 1);

	// dogev load weights
	//nn.loadWeights("weights.csv");

	//create neural network trainer
	neuralNetworkTrainer nT(&nn);
	nT.setTrainingParameters(/*0.001*/0.001, /*0.9*/0.1, true);
	nT.setStoppingConditions(550, 98);
	nT.enableLogging("log.csv", 5);

	//train neural network on data sets
	for (int i = 0; i < d.getNumTrainingSets(); ++i) {
		nT.trainNetwork(d.getTrainingDataSet());
	}

	//save the weights
	nn.saveWeights((char *) "weights.csv");

	cout << endl << endl << "-- END OF PROGRAM --" << endl;
	char c;
	cin >> c;
}

int 			check(int dataSize, NetVariableType *dataOut, NetVariableType *expectedOut, NetVariableType epsilon) {
	int status = 0;
	for (int i = 0; i < dataSize; ++i) {
		if (fabs(dataOut[i]-expectedOut[i]) > epsilon) { //compare floats via epsilon
			cout << "error! output data from DFE @ ["<<i<<"] = " <<dataOut[i]<<" (but expected " <<expectedOut[i]<< ")" << endl;
			status = 1;
		}
	}
	return status;
}

bool 			loadCsvInToArray(char* filename, NetVariableType* array, int elms_to_load, int start_at_elem){
	//open file for reading
	fstream inputFile;
	inputFile.open(filename, ios::in);
	bool 	flag_did_any_read = false;

	if ( inputFile.is_open() ){
		string line = "";
		
		//read data
		int i = 0;
		while ( !inputFile.eof() )
		{
			getline(inputFile, line);				
			
			//process line
			if (line.length() > 2 ) 
			{				
				//store inputs		
				char* cstr = new char[line.size()+1];
				char* t;
				strcpy(cstr, line.c_str());

				//tokenise
				t=strtok (cstr,",");
				while ( t!=NULL )
				{	
					if ((i < elms_to_load+start_at_elem) && (i >= start_at_elem)){
						array[i-start_at_elem]	=	( atof(t) );
						flag_did_any_read 		=	true;
					}
					//move token onwards
					t = strtok(NULL,",");
					i++;			
				}
				//free memory
				delete[] cstr;
			}
		}

		//check if sufficient array were loaded
		if ( i != elms_to_load)
		{
			if (flag_did_any_read){
				cout << endl << "Error - Incorrect number of elements in input file: " << filename <<"(got " <<i<< " instead of "<< elms_to_load << " )" << endl ;
			}else{
				cout << endl << "Error - DIDN'T READ ANY DATA, also Incorrect number of elements in input file: " << filename <<"(got " <<i<< " instead of "<< elms_to_load << " )" << endl ;
			}
			//close file
			inputFile.close();

			return false;
		}else{
			if (flag_did_any_read){
				//print success
				cout << endl << "CSV loaded successfully from '" << filename << "'" << " starting from the ["<<start_at_elem<< "] element"<< endl;

				//close file
				inputFile.close();

				return true;

			}else{
				cout << endl << "CSV loaded successfully from '" << filename << "'" << " but didn't read any element"<< endl;

				//close file
				inputFile.close();

				return false;

			}
		}


	}else {
		cout << endl << "Error - CSV input file '" << filename << "' could not be opened: " << endl;
		return false;
	}

	return false;
}

const char*		create_routing_string(){

	// routing string for fanout by net params:
	string final_fan_str;
	string fan_name;

	// for weight:
	for (int i = 0; i < NUM_OF_HIDDEN_LAYERS + 1; ++i) {
		fan_name = "weight_fan";
		fan_name.append(to_string(i));			//fan_name = "weight_fan" + to_string(i);


		// FF
		final_fan_str.append(fan_name);
		final_fan_str.append("->");
		final_fan_str.append("fan_weight_FF");
		final_fan_str.append(to_string(i));//final_fan_str.append("fan_weight_FF" + to_string(i));
		// BP
		final_fan_str.append(",");
		final_fan_str.append(fan_name);
		final_fan_str.append("->");
		final_fan_str.append("fan_weight_BP");
		final_fan_str.append(to_string(i));//final_fan_str.append("fan_weight_BP" + to_string(i));
		final_fan_str.append(",");//if (i< N_INPUTS*HIDDEN_LAYERS_NEURONS   -1)	{final_fan_str.append(",");}
	}
	final_fan_str.append("\n");

	// for neuron_output:
	for (int i = 0; i < NUM_OF_HIDDEN_LAYERS + 1; ++i) {
		fan_name = "neuron_output_fan";
		fan_name.append(to_string(i));	//fan_name = "neuron_output_fan" + to_string(i);

		// FF
		final_fan_str.append(fan_name);
		final_fan_str.append("->");
		final_fan_str.append("neuron_output_fan_FF");
		final_fan_str.append(to_string(i));	//final_fan_str.append("neuron_output_fan_FF" + to_string(i));
		// BP_Div
		final_fan_str.append(",");
		final_fan_str.append(fan_name);
		final_fan_str.append("->");
		final_fan_str.append("neuron_output_fan_BP_Div");
		final_fan_str.append(to_string(i));//final_fan_str.append("neuron_output_fan_BP_Div" + to_string(i));
		// BP_CB
		final_fan_str.append(",");
		final_fan_str.append(fan_name);
		final_fan_str.append("->");
		final_fan_str.append("neuron_output_fan_BP_CB");final_fan_str.append(to_string(i));//final_fan_str.append("neuron_output_fan_BP_CB" + to_string(i));

		final_fan_str.append(",");//if (i< N_INPUTS*HIDDEN_LAYERS_NEURONS   -1)	{final_fan_str.append(",");}
	}
	final_fan_str.append("\n");

	// input to ANN:
	fan_name = "data_into_ann_fan";
	// FF
	final_fan_str.append(fan_name);
	final_fan_str.append("->");
	final_fan_str.append("data_into_ann_fan_FF");
	final_fan_str.append(",");
	// BP
	final_fan_str.append(fan_name);
	final_fan_str.append("->");
	final_fan_str.append("data_into_ann_fan_BP");

	//return final_fan_str.c_str();
	char* chars = new char [final_fan_str.length()+1];
	for (int i=0 ; i<final_fan_str.length() ; ++i){
		chars[i]	=	final_fan_str.at(i);
	}
	chars[final_fan_str.length()]='\0';
	return chars;
	//return final_fan_str;
	//return "weight_fan0->fan_weight_FF0,weight_fan0->fan_weight_BP0,weight_fan1->fan_weight_FF1,weight_fan1->fan_weight_BP1,neuron_output_fan0->neuron_output_fan_FF0,neuron_output_fan0->neuron_output_fan_BP_Div0,neuron_output_fan0->neuron_output_fan_BP_CB0,neuron_output_fan1->neuron_output_fan_FF1,neuron_output_fan1->neuron_output_fan_BP_Div1,neuron_output_fan1->neuron_output_fan_BP_CB1,data_into_ann_fan->data_into_ann_fan_FF,data_into_ann_fan->data_into_ann_fan_BP";
}

void 			printArrInEvenRows(NetVariableType* array, string array_name, int array_size, int row_size){
	cout << array_name <<":"<< endl;
	for (int i = 0; i < array_size; ) {
		cout  <<"\t" <<PLOT_EVEN_SPACE_NAME<< array_name << "["<<PLOT_EVEN_SPACE_INT<< i <<" : "<<PLOT_EVEN_SPACE_INT<<(i+row_size-1) << PLOT_EVEN_SPACE_EQULS << "]= ";
		for (int j = 0; j < row_size; ++i,++j){
			if 			(j == 0 && j==row_size-1){
				cout <<"( " <<PLOT_EVEN_SPACE_FLOAT<< array[i] << ")" << endl;
			}else if 	(j == 0 && j!=row_size-1){
				cout <<"( " <<PLOT_EVEN_SPACE_FLOAT<< array[i] << " , ";
			}else if 	(j==row_size-1) {
				cout <<PLOT_EVEN_SPACE_FLOAT<< array[i] <<  " )" << endl;
			}else{
				cout <<PLOT_EVEN_SPACE_FLOAT<< array[i] << " , ";
			}
		}
	}
	cout << endl;
}

void 			printMATRIXInEvenRows(int** matrix, string array_name, int col_size, int row_size){
	cout << array_name <<":"<< endl;
	for(int i = 0; i < col_size; ++i){
		cout  <<"\t" <<PLOT_EVEN_SPACE_NAME<< array_name << "["<<PLOT_EVEN_SPACE_INT<< i <<"]["<<PLOT_EVEN_SPACE_INT<< "0" " : "<<PLOT_EVEN_SPACE_INT<<(row_size-1) << PLOT_EVEN_SPACE_EQULS << "]= ";
		for(int j = 0; j < row_size; ++j){
			if 			(j == 0 && j==row_size-1){
				cout <<"( " <<PLOT_EVEN_SPACE_INT<< matrix[i][j] << ")" << endl;
			}else if 	(j == 0 && j!=row_size-1){
				cout <<"( " <<PLOT_EVEN_SPACE_INT<< matrix[i][j] << " , ";
			}else if 	(j==row_size-1) {
				cout <<PLOT_EVEN_SPACE_INT<< matrix[i][j] <<  " )" << endl;
			}else{
				cout <<PLOT_EVEN_SPACE_INT<< matrix[i][j] << " , ";
			}
		}
	}
	cout << endl;
}


void 			getNetsFinalActivationCalcFromActivtionArr(NetVariableType* full_array, NetVariableType* only_selected_array , int full_array_size, int row_size, int jump_size){
	int k = 0;
	int i = jump_size-(row_size);
	for (; i < full_array_size; ) {
		for (int j = 0; j < row_size; ++i,++j, ++k){
			only_selected_array[k]	=	full_array[i];
		}
		i += jump_size-(row_size);
	}
}

NetVariableType getMSE(NetVariableType* target_array, NetVariableType* actual_array, int size){
	NetVariableType mse=0;
	for ( int i = 0; i < size; ++i ){//pattern incorrect if desired and output differ
		//calculate MSE via POWER 2
		mse += ( target_array[i] - actual_array[i] )*( target_array[i] - actual_array[i] );
	}
	// Normalize by arrays size:
	mse	=	mse/size;
	return mse;
}

NetVariableType getAccuracy(NetVariableType* target_array, NetVariableType* actual_array, int total_size, int row_size){
	int 			incorrect_patterns_cnt = 0;
	NetVariableType accuracy;


	int i=0;
	for ( ; i < total_size ; ){
		// get the class of target_array
		int target_max_index	=i;
		int actual_max_index	=i;
		for ( int k = 0; k < row_size ; ++k, ++i){
			if (target_array[target_max_index] < target_array[i]){
				target_max_index = i;
			}

			if (actual_array[actual_max_index] < actual_array[i]){
				actual_max_index = i;
			}
		}
		int target_class		= target_max_index % row_size;
		int actual_class 		= actual_max_index % row_size;

		if (target_class	!= 	actual_class){
			incorrect_patterns_cnt++;
		}
	}


	/*int i=0;
	for ( ; i < total_size ; ){	//pattern incorrect if desired and output differ
		bool pattern_is_correct = true;
		for ( int k = 0; k < row_size ; ++k, ++i){			// per output row (a.k.a output layer)
			if (target_array[i] != actual_array[i]){
				pattern_is_correct = false;
			}
		}

		if (pattern_is_correct == false){
			incorrect_patterns_cnt++;
		}
	}*/


	// Normalize by arrays size to [%]:
	int				total_entries	=	total_size/row_size;
	NetVariableType	error_rate 		= 	 (static_cast<NetVariableType>(incorrect_patterns_cnt) / total_entries);
	//cout << "incorrect_patterns_cnt = " << incorrect_patterns_cnt << ", total_entries = " << total_entries <<", error_rate=" << error_rate << endl;

	accuracy	=	100 - 100*error_rate;
	return accuracy;
}



void 			duplicateArray(NetVariableType* src_array, NetVariableType* dup_array, int src_array_size, int dup_times){
	for (int i=0 ; i < src_array_size*dup_times ; ++i){
		dup_array[i]	=	src_array[i%src_array_size];
	}
}


NetVariableType	 genRandPolarNormalDist(NetVariableType	mu, NetVariableType	sigma){
	NetVariableType 		U1, U2, W, mult;
  static NetVariableType 	X1, X2;
  static int 				call = 0;

  if (call == 1){
      call = !call;
      return (mu + sigma * (double) X2);
  }

  do{
      U1 = -1 + ((NetVariableType) rand () / RAND_MAX) * 2;
      U2 = -1 + ((NetVariableType) rand () / RAND_MAX) * 2;
      W = pow (U1, 2) + pow (U2, 2);
    } while (W >= 1 || W == 0);

  mult = sqrt ((-2 * log (W)) / W);
  X1 = U1 * mult;
  X2 = U2 * mult;

  call = !call;

  return (mu + sigma * (NetVariableType) X1);
}


void 			standardNormalDistributionArray(NetVariableType* array, int array_size){
	srand(time(NULL));
	rand();

	//std::default_random_engine						generator;
	//std::normal_distribution<NetVariableType>		distribution(0.0,1.0);	// Standard Normal Distribution: mean=0.0, stddev=1.0
	for (int i=0; i<array_size; ++i) {
		array[i]	= genRandPolarNormalDist(0.0, 1.0);//distribution(generator);
	}

	/*Myeng eng;
	eng.seed(rand());
	Mydist dist(0.0,1.0);

	dist.reset(); // discard any cached values
	for (int i=0; i<array_size; ++i) {
		array[i]	= dist(eng);
	}*/
}








void 			permutationOnPairOfArray(	int* elements_index_array, 		int elements_index_array_size,
																			int vector1_size,int vector2_size,
											NetVariableType* srs_vector1_array, NetVariableType* src_vector2_array,
											NetVariableType* prm_vector1_array, NetVariableType* prm_vector2_array,
											int BeginAt, 					int StopAt){
	// elements_index_array is the order of elements from 0 to elements_index_array_size-1
	//std::random_shuffle(elements_index_array, elements_index_array+elements_index_array_size);
	srand(time(NULL));
	rand();

	int k =  rand() % elements_index_array_size;
	// rand elemets index order
	for(int j = BeginAt+1; j < StopAt; ++j) {
        std::swap(elements_index_array[k % (j + 1)], elements_index_array[j]);
		//printf("(%d <-> %d) ", k % (j + 1) , j);
        //k = k / (j + 1);
		k =  rand() % elements_index_array_size;
	}


	if (RAND_SET_PREMUTATION_ONLY_AT_END_OF_SET){	// swap only once done entire set
		if (StopAt==elements_index_array_size){		// within final epoch in set
			// run new order from both source arrays into new
			for(int j = 0; j < elements_index_array_size; ++j) {
				for(int i = 0; i < vector1_size; ++i) {
					prm_vector1_array[j*vector1_size+i]		=	srs_vector1_array[elements_index_array[j]*vector1_size+i];
				}
				for(int i = 0; i < vector2_size; ++i) {
					prm_vector2_array[j*vector2_size+i]		=	src_vector2_array[elements_index_array[j]*vector2_size+i];
				}
			}
		}else{
			// dont move data entries yet
		}
	}else{// so swap only on current epoch

	}
	// run new order from both source arrays into new
	for(int j = BeginAt; j < StopAt; ++j) {
		for(int i = 0; i < vector1_size; ++i) {
			prm_vector1_array[j*vector1_size+i]		=	srs_vector1_array[elements_index_array[j]*vector1_size+i];
		}
		for(int i = 0; i < vector2_size; ++i) {
			prm_vector2_array[j*vector2_size+i]		=	src_vector2_array[elements_index_array[j]*vector2_size+i];
		}
	}

}




NetVariableType clampToBinaryOutput( NetVariableType x )
{
	if (CLAP_ANN_OUT_TO_BINARY){
		if ( x < CLAP_MAX_VALUE_FOR_0 ) 		return 0;
		else if ( x > CLAP_MIN_VALUE_FOR_1 ) 	return 1;
		else 									return -1;
	}
	// if clamp isnt needed, dont do a thing
	return x;
}


NetVariableType		calcConfusionMatrix(int** confusion_matrix, NetVariableType* target_array, NetVariableType* actual_array, int total_size, int row_size){
	if (CLAP_ANN_OUT_TO_BINARY){

		int i=0;
		for ( ; i < total_size ; ){
			// get the class of target_array
			int target_max_index	=i;
			for ( int k = 0; k < row_size ; ++k, ++i){
				if (target_array[target_max_index] < target_array[i]){
					target_max_index = i;
				}
			}
			int target_class		= target_max_index % row_size;
			// re run on same last row,
			// this time compare each element to its target (instead of just max class)
			i -= row_size;
			for ( int k = 0; k < row_size ; ++k, ++i){
				if ( clampToBinaryOutput(actual_array[i]) == target_array[i] ) confusion_matrix[target_class][k]++;
			}
		}


		int 			correct_patterns_cnt = 0;


		for (int j=0 ; j < row_size ; ++j){
			if (CLAP_ANN_OUT_TO_BINARY){// for each time we were right, not only on max elm
				for (int l=0 ; l < row_size ; ++l){
					correct_patterns_cnt	+=	confusion_matrix[j][l];
				}
			}else{	// calc Accuracy via diagonal:
				correct_patterns_cnt	+=	confusion_matrix[j][j];
			}
		}

		NetVariableType	match_rate 		= 	(static_cast<NetVariableType>(correct_patterns_cnt) / total_size);

		return	(100*match_rate);

	}else{// get class by max element in row to indicate it

		int i=0;
		for ( ; i < total_size ; ){
			// get the class of target_array
			int target_max_index	=i;
			int actual_max_index	=i;
			for ( int k = 0; k < row_size ; ++k, ++i){
				if (target_array[target_max_index] < target_array[i]){
					target_max_index = i;
				}

				if (actual_array[actual_max_index] < actual_array[i]){
					actual_max_index = i;
				}
			}
			int target_class		= target_max_index % row_size;
			int actual_class 		= actual_max_index % row_size;
			//cout << "calcConfusionMatrix: target_class=" << target_class<< " actual_class=" << actual_class << endl;

			confusion_matrix[target_class][actual_class]++;
		}


		// calc Accuracy via diagonal:
		int 			correct_patterns_cnt = 0;
		for (int j=0 ; j < row_size ; ++j){
			correct_patterns_cnt	+=	confusion_matrix[j][j];
		}
		int				total_entries	=	total_size/row_size;
		NetVariableType	match_rate 		= 	 (static_cast<NetVariableType>(correct_patterns_cnt) / total_entries);
		return	(100*match_rate);
	}



	return	-1.00000;
}



void doStatistics(NetVariableType* expected_result_of_ann, NetVariableType* actual_result_of_ann, int  expected_result_of_ann_size, FILE* AccuracyLogFile, bool do_plot){
	NetVariableType				accuracy/*Array[num_of_epochs_in_set*(redo_set_time+1)]*/;

	if (PLOT_STATISTICS || do_plot) cout << "epoch's MSE is " 		<< getMSE(		expected_result_of_ann, actual_result_of_ann, expected_result_of_ann_size) << endl;

	int** confusion_matrix = new int*[N_OUTPUTS];
	for(int i = 0; i < N_OUTPUTS; ++i){
		confusion_matrix[i] = new int[N_OUTPUTS];
		for(int j = 0; j < N_OUTPUTS; ++j){
			confusion_matrix[i][j] = 0;
		}
	}
	accuracy/*Array[set_itr*num_of_epochs_in_set+epoch_itr]*/		=	calcConfusionMatrix(confusion_matrix, expected_result_of_ann, actual_result_of_ann, expected_result_of_ann_size, N_OUTPUTS);

	if (PLOT_STATISTICS || do_plot) printMATRIXInEvenRows(confusion_matrix, "confusion_matrix", N_OUTPUTS, N_OUTPUTS);
	if (PLOT_STATISTICS || do_plot) cout << "epoch's Accuracy is " 	<< accuracy/*Array[set_itr*num_of_epochs_in_set+epoch_itr]*/ << endl;

	fprintf(AccuracyLogFile, "%f ", accuracy);

}

void doUpdateModelAfterDFE(	int epoch_size,								NetVariableType learning_rate,
							int weights_total_size, 					int biases_total_size,
							NetVariableType* weights_total, 			NetVariableType* biases_total,
							NetVariableType* weights_correction_total, 	NetVariableType* biases_correction_total){
	for (int i = 0; i < weights_total_size; ++i) {
		//printf("weights_correction_total[%d]=%f	+=(%f*%f/%d=%f) \n" , i, weights_total[i], learning_rate , weights_correction_total[i] , epoch_size,(learning_rate*weights_correction_total[i]/epoch_size));
		weights_total[i]				+=	learning_rate*weights_correction_total[i]/epoch_size;
	}
	for (int i = 0; i < biases_total_size; ++i) {
		//printf("biases_total[%d]=%f	+=(%f*%f/%d=%f) \n" , i, biases_total[i], learning_rate , biases_correction_total[i] , epoch_size,(learning_rate*weights_correction_total[i]/epoch_size));
		biases_total[i]					+=	learning_rate*biases_correction_total[i]/epoch_size;
	}
}

// MAIN:
int main() {
	int 			epoch_size 							= DualKernel_EPOCH_SIZE;// the size of a epoch in [entries]
	int 			num_of_epochs_in_set 				= NUM_OF_EPOCHS_IN_SET;	// the number of epochs in a training set
	int 			redo_set_time 						= REDO_SET_TIME;
	NetVariableType learning_rate						= CORRECTION_LEARNING_RATE;

	cout 	<< "===================================" <<  endl

			<< "ANN Model:" << endl
			<< "\t inputs=" << N_INPUTS << endl
			<< "\t number of hidden layers=" << NUM_OF_HIDDEN_LAYERS << endl
			<< "\t number of neurons per each hidden layer=" << HIDDEN_LAYERS_NEURONS << endl
			<< "\t outputs=" << N_OUTPUTS << endl

			<< "Learning Parameters:" << endl
			<< "\t epoch_size=" << epoch_size << endl
			<< "\t num_of_epochs_in_set=" << num_of_epochs_in_set << endl
			<< "\t redo_set_time=" << redo_set_time << endl
			<< "\t learning_rate=" << learning_rate << endl

			<< "Plot Setting:" << endl
			<< "\t RAND_INIT_WIGHTS=" 			<< RAND_INIT_WIGHTS << endl
			<< "\t PLOT_EPOCH_LOG_ACTIVETY=" 	<< PLOT_EPOCH_LOG_ACTIVETY << endl
			<< "\t CMP_TO_RESULT_FROM_MATLAB=" 	<< CMP_TO_RESULT_FROM_MATLAB << endl
			<< "\t PLOT_ANN_OUTPUT=" 			<< PLOT_ANN_OUTPUT << endl
			<< "\t PLOT_DEBUG_MATRIX=" 			<< PLOT_DEBUG_MATRIX << endl
			<< "\t DO_STATISTICS=" 				<< DO_STATISTICS << endl
			<< "\t PLOT_STATISTICS=" 			<< PLOT_STATISTICS << endl
			<< "\t CLAP_ANN_OUT_TO_BINARY=" 	<< CLAP_ANN_OUT_TO_BINARY << endl
			<< "\t\t CLAP_MAX_VALUE_FOR_0=" 			<< CLAP_MAX_VALUE_FOR_0 << endl
			<< "\t\t CLAP_MIN_VALUE_FOR_1=" 			<< CLAP_MIN_VALUE_FOR_1 << endl

			<< "===================================" <<  endl;


	printf("Allocating Streams.\n");

	FILE * AccuracyLogFile;
	AccuracyLogFile = fopen ("AccuracyLogFile.txt","w");

	const char * 	routing_string 						= create_routing_string();
	cout << "routing_string is :" << routing_string << endl;


	int status;

	const int data_into_ann_size 			= epoch_size*N_INPUTS;
	const int expected_result_of_ann_size 	= epoch_size*N_OUTPUTS;


	NetVariableType* src_set_data_into_ann 			= static_cast<NetVariableType*> (malloc(sizeof(NetVariableType)*data_into_ann_size*num_of_epochs_in_set));
	NetVariableType* src_set_expected_result_of_ann = static_cast<NetVariableType*> (malloc(sizeof(NetVariableType)*expected_result_of_ann_size*num_of_epochs_in_set));
	NetVariableType* permutation_set_data_into_ann 			= static_cast<NetVariableType*> (malloc(sizeof(NetVariableType)*data_into_ann_size*num_of_epochs_in_set));
	NetVariableType* permutation_set_expected_result_of_ann = static_cast<NetVariableType*> (malloc(sizeof(NetVariableType)*expected_result_of_ann_size*num_of_epochs_in_set));
	NetVariableType* NEXTpermutation_set_data_into_ann 			= static_cast<NetVariableType*> (malloc(sizeof(NetVariableType)*data_into_ann_size*num_of_epochs_in_set));
	NetVariableType* NEXTpermutation_set_expected_result_of_ann = static_cast<NetVariableType*> (malloc(sizeof(NetVariableType)*expected_result_of_ann_size*num_of_epochs_in_set));
	loadCsvInToArray("training_set.csv", 			src_set_data_into_ann, 			(num_of_epochs_in_set*data_into_ann_size), 			0);
	loadCsvInToArray("training_class.csv", 			src_set_expected_result_of_ann, (num_of_epochs_in_set*expected_result_of_ann_size),	0);
	int	elements_index_array[epoch_size*num_of_epochs_in_set];	// the order on entries for shuffling
	for (int i=0 ; i<epoch_size*num_of_epochs_in_set ; ++i){
		elements_index_array[i]	=	i;
	}
	duplicateArray(		src_set_data_into_ann, permutation_set_data_into_ann,
						data_into_ann_size*num_of_epochs_in_set, 1);
	duplicateArray(		src_set_expected_result_of_ann, permutation_set_expected_result_of_ann,
						expected_result_of_ann_size*num_of_epochs_in_set, 1);
	duplicateArray(		src_set_data_into_ann, NEXTpermutation_set_data_into_ann,
						data_into_ann_size*num_of_epochs_in_set, 1);
	duplicateArray(		src_set_expected_result_of_ann, NEXTpermutation_set_expected_result_of_ann,
						expected_result_of_ann_size*num_of_epochs_in_set, 1);

	// FF:
	NetVariableType* data_into_ann;
	NetVariableType* expected_result_of_ann;
	NetVariableType* PREVexpected_result_of_ann;
	if (CMP_TO_RESULT_FROM_MATLAB){// the ALLOCATE
		data_into_ann 			= static_cast<NetVariableType*> (malloc(sizeof(NetVariableType)*data_into_ann_size));
		expected_result_of_ann 	= static_cast<NetVariableType*> (malloc(sizeof(NetVariableType)*expected_result_of_ann_size));
		PREVexpected_result_of_ann 	= static_cast<NetVariableType*> (malloc(sizeof(NetVariableType)*expected_result_of_ann_size));
	}else{
		// keep it just a pointer
		data_into_ann			=	src_set_data_into_ann;
		expected_result_of_ann	=	src_set_expected_result_of_ann;
	}

	const int weights_total_size 			= (	N_INPUTS*HIDDEN_LAYERS_NEURONS +
												HIDDEN_LAYERS_NEURONS*HIDDEN_LAYERS_NEURONS*(NUM_OF_HIDDEN_LAYERS-1)+
												HIDDEN_LAYERS_NEURONS*N_OUTPUTS);
		const int weight0_size = N_INPUTS*HIDDEN_LAYERS_NEURONS;
		const int weight1_size = HIDDEN_LAYERS_NEURONS*HIDDEN_LAYERS_NEURONS;
		const int weight2_size = HIDDEN_LAYERS_NEURONS*N_OUTPUTS ;
	NetVariableType* weights_total 			= static_cast<NetVariableType*> (malloc(sizeof(NetVariableType)*weights_total_size));
	// make pointers per input stream from the main stream
		NetVariableType* tmp_ptr_weights_summer = weights_total;
		NetVariableType* weight0 			= tmp_ptr_weights_summer;	tmp_ptr_weights_summer+=N_INPUTS*HIDDEN_LAYERS_NEURONS;
		NetVariableType* weight1 			= tmp_ptr_weights_summer;	if (NUM_OF_HIDDEN_LAYERS>1) tmp_ptr_weights_summer+=HIDDEN_LAYERS_NEURONS*HIDDEN_LAYERS_NEURONS;
		NetVariableType* weight2 			= tmp_ptr_weights_summer;	tmp_ptr_weights_summer+=HIDDEN_LAYERS_NEURONS*N_OUTPUTS;


	const int biases_total_size 			= (		HIDDEN_LAYERS_NEURONS +
													HIDDEN_LAYERS_NEURONS*(NUM_OF_HIDDEN_LAYERS-1)+
													N_OUTPUTS);
			const int bias0_size 	=		HIDDEN_LAYERS_NEURONS;
			const int bias1_size 	= 		HIDDEN_LAYERS_NEURONS;
			const int bias2_size 	= 		N_OUTPUTS;
	NetVariableType* biases_total 	= static_cast<NetVariableType*> (malloc(sizeof(NetVariableType)*biases_total_size));
	// make pointers per input stream from the main stream
		NetVariableType* tmp_ptr_bias_summer= biases_total;
		NetVariableType* bias0				= tmp_ptr_bias_summer;	tmp_ptr_bias_summer+=HIDDEN_LAYERS_NEURONS;
		NetVariableType* bias1				= tmp_ptr_bias_summer;	if (NUM_OF_HIDDEN_LAYERS>1) tmp_ptr_bias_summer+=HIDDEN_LAYERS_NEURONS;
		NetVariableType* bias2				= tmp_ptr_bias_summer;	tmp_ptr_bias_summer+=N_OUTPUTS;



	// ANN Result:
	const int         	actual_result_of_ann_size      	= expected_result_of_ann_size;
	NetVariableType* 	actual_result_of_ann           	= static_cast<NetVariableType*> (malloc(sizeof(NetVariableType)*actual_result_of_ann_size));
	NetVariableType* 	PREVactual_result_of_ann           	= static_cast<NetVariableType*> (malloc(sizeof(NetVariableType)*actual_result_of_ann_size));




	// BP:
	NetVariableType* weights_correction_total 			= static_cast<NetVariableType*> (malloc(sizeof(NetVariableType)*weights_total_size));
	// make pointers per input stream from the main stream
		NetVariableType* tmp_ptr_corrected_weights_summer = weights_correction_total;
		NetVariableType* weight_correction0 			= tmp_ptr_corrected_weights_summer;	tmp_ptr_corrected_weights_summer+=N_INPUTS*HIDDEN_LAYERS_NEURONS;
		NetVariableType* weight_correction1 			= tmp_ptr_corrected_weights_summer;	if (NUM_OF_HIDDEN_LAYERS>1) tmp_ptr_corrected_weights_summer+=HIDDEN_LAYERS_NEURONS*HIDDEN_LAYERS_NEURONS;
		NetVariableType* weight_correction2 			= tmp_ptr_corrected_weights_summer;	tmp_ptr_corrected_weights_summer+=HIDDEN_LAYERS_NEURONS*N_OUTPUTS;



	NetVariableType* biases_correction_total 	= static_cast<NetVariableType*> (malloc(sizeof(NetVariableType)*biases_total_size));
	// make pointers per input stream from the main stream
		NetVariableType* tmp_ptr_bias_correction_summer= biases_correction_total;
		NetVariableType* bias_correction0				= tmp_ptr_bias_correction_summer;	tmp_ptr_bias_correction_summer+=HIDDEN_LAYERS_NEURONS;
		NetVariableType* bias_correction1				= tmp_ptr_bias_correction_summer;	if (NUM_OF_HIDDEN_LAYERS>1) tmp_ptr_bias_correction_summer+=HIDDEN_LAYERS_NEURONS;
		NetVariableType* bias_correction2				= tmp_ptr_bias_correction_summer;	tmp_ptr_bias_correction_summer+=N_OUTPUTS;





	const int         	activations_size      	= epoch_size*(NUM_OF_HIDDEN_LAYERS*HIDDEN_LAYERS_NEURONS+N_OUTPUTS);
	NetVariableType* 	activations				=	static_cast<NetVariableType*> (malloc(sizeof(NetVariableType)*activations_size));
	const int         	last_activations_size   = epoch_size*(N_OUTPUTS);
	NetVariableType* 	last_activations		=	static_cast<NetVariableType*> (malloc(sizeof(NetVariableType)*last_activations_size));




	// Load & Init ANN weights and Bias:
	printf("Streams Initial Weights and Bias.\n");
	if (RAND_INIT_WIGHTS){	// Rand From Standard Normal Dist
		standardNormalDistributionArray(weights_total, 			(weights_total_size));
		standardNormalDistributionArray(biases_total, 			(biases_total_size));

	}else{					// load from CSV:
		loadCsvInToArray("initial_model_weights.csv", 	weights_total, 			(weights_total_size), 0);
		loadCsvInToArray("initial_model_biases.csv", 	biases_total, 			(biases_total_size), 0);
	}



	// plot initial model:
	printArrInEvenRows(weight0, 				"initial weight0", 					weight0_size, HIDDEN_LAYERS_NEURONS);
	printArrInEvenRows(weight1, 				"initial weight1", 					weight1_size, HIDDEN_LAYERS_NEURONS);
	printArrInEvenRows(weight2, 				"initial weight2", 					weight2_size, N_OUTPUTS);

	printArrInEvenRows(bias0, 					"initial bias0", 					bias0_size, HIDDEN_LAYERS_NEURONS);
	printArrInEvenRows(bias1, 					"initial bias1", 					bias1_size, HIDDEN_LAYERS_NEURONS);
	printArrInEvenRows(bias2, 					"initial bias2", 					bias2_size, N_OUTPUTS);



	// rand set's order Before first run
	if (! CMP_TO_RESULT_FROM_MATLAB){
		permutationOnPairOfArray(	elements_index_array			,	(epoch_size*num_of_epochs_in_set),
									N_INPUTS						,	N_OUTPUTS,
									src_set_data_into_ann			,	src_set_expected_result_of_ann,
									permutation_set_data_into_ann	,	permutation_set_expected_result_of_ann,
									0								,	(epoch_size*num_of_epochs_in_set));
		permutationOnPairOfArray(	elements_index_array			,	(epoch_size*num_of_epochs_in_set),
									N_INPUTS						,	N_OUTPUTS,
									src_set_data_into_ann			,	src_set_expected_result_of_ann,
									NEXTpermutation_set_data_into_ann,	NEXTpermutation_set_expected_result_of_ann,
									0								,	(epoch_size*num_of_epochs_in_set));

	}





	// repeat set:
    //high_resolution_clock::time_point total_training_begin = high_resolution_clock::now();
    clock_t total_training_begin = clock();
	for (int set_itr = 0; set_itr < redo_set_time   + 1  ; ++set_itr) {
		if (set_itr < redo_set_time){
			if (PLOT_EPOCH_LOG_ACTIVETY) printf("Begin set iteration number %d out of %d\n",(set_itr+1),redo_set_time);
		}
		else{
			printf("=====================================\n Begin TESTING iteration AFTER finishing training set.\n=====================================\n");
			loadCsvInToArray("testing_set.csv", 			permutation_set_data_into_ann, 			(num_of_epochs_in_set*data_into_ann_size), 			0);
			loadCsvInToArray("testing_class.csv", 			permutation_set_expected_result_of_ann, (num_of_epochs_in_set*expected_result_of_ann_size),	0);

			printArrInEvenRows(weight0, 				"final weight0", 					weight0_size, HIDDEN_LAYERS_NEURONS);
			printArrInEvenRows(weight1, 				"final weight1", 					weight1_size, HIDDEN_LAYERS_NEURONS);
			printArrInEvenRows(weight2, 				"final weight2", 					weight2_size, N_OUTPUTS);

			printArrInEvenRows(bias0, 					"final bias0", 						bias0_size, HIDDEN_LAYERS_NEURONS);
			printArrInEvenRows(bias1, 					"final bias1", 						bias1_size, HIDDEN_LAYERS_NEURONS);
			printArrInEvenRows(bias2, 					"final bias2", 						bias2_size, N_OUTPUTS);
		}





		/// tmp permut
		// rand set's order Before first run
		if (! CMP_TO_RESULT_FROM_MATLAB){
			permutationOnPairOfArray(	elements_index_array			,	(epoch_size*num_of_epochs_in_set),
										N_INPUTS						,	N_OUTPUTS,
										src_set_data_into_ann			,	src_set_expected_result_of_ann,
										permutation_set_data_into_ann	,	permutation_set_expected_result_of_ann,
										0								,	(epoch_size*num_of_epochs_in_set));
		}









		// loop per epoch:

		for (int epoch_itr = 0; epoch_itr < num_of_epochs_in_set; ++epoch_itr) {
		    clock_t epoch_training_begin = clock();

			if (PLOT_EPOCH_LOG_ACTIVETY) printf("Begin epoch iteration number %d out of %d\n",(epoch_itr+1),num_of_epochs_in_set);

			// for fill PSI steam, dup for epoch size :<
			//duplicateArray(weight0, weight0dup, weight0_size, epoch_size);
			//duplicateArray(weight2, weight2dup, weight2_size, epoch_size);
			//duplicateArray(bias0, bias0dup, bias0_size, epoch_size);
			//duplicateArray(bias2, bias2dup, bias2_size, epoch_size);


			if (CMP_TO_RESULT_FROM_MATLAB){
				// load input data per epoch ONLY!!!!
				loadCsvInToArray("training_set.csv", 			data_into_ann, 			(data_into_ann_size), 			epoch_itr*data_into_ann_size);
				loadCsvInToArray("training_class.csv", 			expected_result_of_ann, (expected_result_of_ann_size),	epoch_itr*expected_result_of_ann_size);

				// the expected ANN output from matlab:
				// load expected matlab simulation of same ANN:
				loadCsvInToArray("activations.csv", 		activations, 			(activations_size),			epoch_itr*activations_size);
				getNetsFinalActivationCalcFromActivtionArr(	activations, last_activations,
														activations_size, last_activations_size/epoch_size,  activations_size/epoch_size);
			}else{	// dont compare, just read data
				data_into_ann			=	epoch_itr*data_into_ann_size			+	permutation_set_data_into_ann;
				expected_result_of_ann	=	epoch_itr*expected_result_of_ann_size	+	permutation_set_expected_result_of_ann;
			}



			// plot model before run
			if (PLOT_DEBUG_MATRIX){
				printArrInEvenRows(weight0, 				"weight0", 					weight0_size, HIDDEN_LAYERS_NEURONS);
				printArrInEvenRows(weight1, 				"weight1", 					weight1_size, HIDDEN_LAYERS_NEURONS);
				printArrInEvenRows(weight2, 				"weight2", 					weight2_size, N_OUTPUTS);

				printArrInEvenRows(bias0, 					"bias0", 					bias0_size, HIDDEN_LAYERS_NEURONS);
				printArrInEvenRows(bias1, 					"bias1", 					bias1_size, HIDDEN_LAYERS_NEURONS);
				printArrInEvenRows(bias2, 					"bias2", 					bias2_size, N_OUTPUTS);

				// print input data
				printArrInEvenRows(data_into_ann, 			"data_into_ann", 			data_into_ann_size, N_INPUTS);
				printArrInEvenRows(expected_result_of_ann, 	"expected_result_of_ann", 	expected_result_of_ann_size, N_OUTPUTS);

				// compare with matlab
				if (CMP_TO_RESULT_FROM_MATLAB){
					printArrInEvenRows(last_activations, 		"last_activations", 		last_activations_size, N_OUTPUTS);
				}
			}




			// run the DFE
			if (PLOT_EPOCH_LOG_ACTIVETY) printf("Running DFE.\n");
			max_run_t* dfe = DualKernel_nonblock(
				1,				// we run 1 epoch per call for this sim:


				bias0,
				//bias1,
				bias2,

				data_into_ann,
				expected_result_of_ann,

				weight0,
				//weight1,
				weight2,

				actual_result_of_ann,

				bias_correction0,
				//bias_correction1,
				bias_correction2,


				weight_correction0,
				//weight_correction1,
				weight_correction2,

				routing_string);



			// NON BLOCK DFE SO CPU HOST WILL DO:
			// Rand next set's order
			if (! CMP_TO_RESULT_FROM_MATLAB){
				permutationOnPairOfArray(	elements_index_array			,	(epoch_size*num_of_epochs_in_set),
											N_INPUTS						,	N_OUTPUTS,
											src_set_data_into_ann			,	src_set_expected_result_of_ann,
											NEXTpermutation_set_data_into_ann,	NEXTpermutation_set_expected_result_of_ann,
											epoch_itr*epoch_size			,	(epoch_itr+1)*epoch_size);
			}
			// Statistics on Prev DFE run:
			if (DO_STATISTICS && (!(set_itr==0 && epoch_itr==0))){// dont run on first epocj ever, for there are no valid prev to do statistics on!
				doStatistics(PREVexpected_result_of_ann, PREVactual_result_of_ann, expected_result_of_ann_size, AccuracyLogFile, ((set_itr>redo_set_time)) );
			}

			// Wait for end of DFE:
			max_wait(dfe);
			if (PLOT_EPOCH_LOG_ACTIVETY) printf("Done running DFE.\n");

			// update weights and bias:
			if (set_itr == redo_set_time){	// i.e its the testing set, so don't update
			}else{ 							// i.e its the training set, so do update
				doUpdateModelAfterDFE(		epoch_size,					learning_rate,
											weights_total_size, 		biases_total_size,
											weights_total, 				biases_total,
											weights_correction_total, 	biases_correction_total);
				if (PLOT_EPOCH_LOG_ACTIVETY) printf("Done Update ANN Model.\n");
			}




			// run check on data against CPU&Matlab code to functionally emulate the DFE
			if (CMP_TO_RESULT_FROM_MATLAB){
				status = check(expected_result_of_ann_size,actual_result_of_ann ,last_activations, EPSILON);
				if (status){
					printf("Test failed at ANN actual result.\n");
				}else{
					loadCsvInToArray("new_model_weights.csv", 	weights_correction_total, 			(weights_total_size), 	(epoch_itr)*weights_total_size);
					loadCsvInToArray("new_model_biases.csv", 	biases_correction_total, 			(biases_total_size), 	(epoch_itr)*biases_total_size);

					if (	check(weights_total_size,	weights_correction_total,	weights_total , 	100*EPSILON)  ||
							check(biases_total_size,	biases_correction_total,	biases_total , 		100*EPSILON)	){
						printf("Test failed at new corrected weights & bias.\n");
					}else{
						printf("Test passed OK!\n");
					}
				}
			}else{// can't compare, but cand print
				if (PLOT_ANN_OUTPUT){
					printArrInEvenRows(expected_result_of_ann, 	"expected_result_of_ann", 	expected_result_of_ann_size, N_OUTPUTS);
					printArrInEvenRows(actual_result_of_ann, 	"actual_result_of_ann", 	expected_result_of_ann_size, N_OUTPUTS);
				}

			}



			// swap pointers for next and prev stream:
			NetVariableType*					tmp_hold_ptr;
			tmp_hold_ptr								=	permutation_set_data_into_ann;
			permutation_set_data_into_ann				=	NEXTpermutation_set_data_into_ann;
			NEXTpermutation_set_data_into_ann			=	tmp_hold_ptr;

			tmp_hold_ptr								=	permutation_set_expected_result_of_ann;
			permutation_set_expected_result_of_ann		=	NEXTpermutation_set_expected_result_of_ann;
			NEXTpermutation_set_expected_result_of_ann	=	tmp_hold_ptr;

			tmp_hold_ptr								=	PREVexpected_result_of_ann;
			PREVexpected_result_of_ann					=	expected_result_of_ann;
			expected_result_of_ann						=	tmp_hold_ptr;

			tmp_hold_ptr								=	PREVactual_result_of_ann;
			PREVactual_result_of_ann					=	actual_result_of_ann;
			actual_result_of_ann						=	tmp_hold_ptr;


		    //clock_t epoch_training_end = clock();
		    //double elapsed_secs = double(epoch_training_end - epoch_training_begin) / CLOCKS_PER_SEC;
		    //cout << "epoch learning time is " << elapsed_secs <<  "[sec]" <<endl;

		}	// END OF loop per epoch:
	}	// END OF loop per set:


	// do last test epoch statistics:
	if (DO_STATISTICS){// dont run on first epocj ever, for there are no valid prev to do statistics on!
		doStatistics(PREVexpected_result_of_ann, PREVactual_result_of_ann, expected_result_of_ann_size, AccuracyLogFile, (DO_STATISTICS) );
	}


    //high_resolution_clock::time_point total_training_end = high_resolution_clock::now();
    //auto duration = duration_cast<microseconds>( total_training_end - total_training_begin ).count();
	//cout << duration;
	clock_t total_training_end = clock();
    double elapsed_secs = double(total_training_end - total_training_begin) / CLOCKS_PER_SEC;
    cout << "Total learning time is " << elapsed_secs <<  "[sec]" <<endl;

	// after doing all the epochs wanted. we are done!
	printArrInEvenRows(weight0, 				"final weight0", 					weight0_size, HIDDEN_LAYERS_NEURONS);
	printArrInEvenRows(weight1, 				"final weight1", 					weight1_size, HIDDEN_LAYERS_NEURONS);
	printArrInEvenRows(weight2, 				"final weight2", 					weight2_size, N_OUTPUTS);

	printArrInEvenRows(bias0, 					"final bias0", 						bias0_size, HIDDEN_LAYERS_NEURONS);
	printArrInEvenRows(bias1, 					"final bias1", 						bias1_size, HIDDEN_LAYERS_NEURONS);
	printArrInEvenRows(bias2, 					"final bias2", 						bias2_size, N_OUTPUTS);

	/*printf("final accuracyArray\n");
	for (int i=0 ; i < num_of_epochs_in_set*redo_set_time ; ++i){
		cout << " " << accuracyArray[i];
	}cout << endl;*/


	cout << "DONE" << endl;

	// free me from this alloc:
	if(CMP_TO_RESULT_FROM_MATLAB){ // free the allocated
		free(data_into_ann				);
		free(expected_result_of_ann     );
		free(PREVexpected_result_of_ann );
	}
	free(weights_total              );
	free(biases_total               );
	free(PREVactual_result_of_ann   );
	free(actual_result_of_ann       );
	free(weights_correction_total   );
	free(biases_correction_total    );
	free(activations                );
	free(last_activations           );


	/*free(weight0dup);
	free(weight2dup);
	free(bias0dup);
	free(bias2dup);*/

	free(src_set_data_into_ann);
	free(src_set_expected_result_of_ann);
	free(permutation_set_data_into_ann);
	free(permutation_set_expected_result_of_ann);
	free(NEXTpermutation_set_data_into_ann);
	free(NEXTpermutation_set_expected_result_of_ann);

    fclose (AccuracyLogFile);


	// end
	return 0;
}
