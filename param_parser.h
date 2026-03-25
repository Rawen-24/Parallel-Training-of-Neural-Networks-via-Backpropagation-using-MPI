/*Steps i'm following for the Implementation of this File
1-Define the struct that holds dataset parameters

2-Declare the function that loads the config file

*/
typedef struct
{
    int *one_hot_vector; // vector of one-hot encoded label 
    float *normalized_pixels; // array of normalized pixel values 
} Sample;

typedef struct
{
    int input_neurons;
    int hidden_neurons_1;
    int hidden_neurons_2;
    int output_neurons;
    float learning_rate;
    int epochs;
    int training_samples;
    int testing_samples;
    char data_file[256];
} DatasetParams;

// Load dataset parameters from config file
int load_dataset_parameters(const char *config_file, DatasetParams *param);

// Load training data from .dat file
// load data from .bin file
Sample *load_text_dataset(const char *filename, int samples, DatasetParams *params);