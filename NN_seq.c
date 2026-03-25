#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <math.h>
#include "param_parser.h"

// activation function
float sigmoid(float x)
{
    return 1.0f / (1.0f + expf(-x));
}

float sigmoid_derivation(float x)
{
    return x * (1.0f - x);
}

float initialize_weight()
{
    return ((float)rand() / RAND_MAX) - 0.5f;
}

int main()
{

    srand((unsigned)time(NULL));
    clock_t start_time, end_time;
    double total_computation_time = 0.0;

    //==========================================
    printf("=== SIMPLE TEST: Loading Config ===\n\n");

    DatasetParams params;

    // Call the parser
    int result = load_dataset_parameters("mnist_bench_case1.cfg", &params);

    if (result != 0)
    {
        printf("ERROR: Parser failed!\n");
        return -1;
    }

    printf("\n=== SUCCESS! Data extracted ===\n\n");

    printf("params.input_neurons = %d\n", params.input_neurons);
    printf("params.output_neurons = %d\n", params.output_neurons);
    printf("params.hidden_neurons_1 = %d\n", params.hidden_neurons_1);
    printf("params.hidden_neurons_2 = %d\n", params.hidden_neurons_2);
    printf("params.learning_rate = %.4f\n", params.learning_rate);
    printf("params.epochs = %d\n", params.epochs);
    printf("params.training_samples = %d\n", params.training_samples);
    printf("params.testing_samples = %d\n", params.testing_samples);
    printf("params.data_file = %s\n", params.data_file);

    float **Input = (float *)malloc(params.training_samples * sizeof(float *)); // (float *)
    float **Output = (float *)malloc(params.training_samples * sizeof(float *));
    for (int i = 0; i < params.training_samples; i++)
    {
        Input[i] = malloc(params.input_neurons * sizeof(float));
        Output[i] = malloc(params.output_neurons * sizeof(float));
    }

    float **weight_input_hidden1 = malloc(params.input_neurons * sizeof(float *));
    for (int i = 0; i < params.input_neurons; i++)
        weight_input_hidden1[i] = malloc(params.hidden_neurons_1 * sizeof(float));

    float **weight_hidden1_hidden2 = malloc(params.hidden_neurons_1 * sizeof(float *));
    for (int i = 0; i < params.hidden_neurons_1; i++)
        weight_hidden1_hidden2[i] = malloc(params.hidden_neurons_2 * sizeof(float));

    float **weight_hidden2_Output = malloc(params.hidden_neurons_2 * sizeof(float *));
    for (int i = 0; i < params.hidden_neurons_2; i++)
        weight_hidden2_Output[i] = malloc(params.output_neurons * sizeof(float));

    float *bias_hidden_1 = malloc(params.hidden_neurons_1 * sizeof(float));
    float *bias_hidden_2 = malloc(params.hidden_neurons_2 * sizeof(float));
    float *bias_Output = malloc(params.output_neurons * sizeof(float));
    //========TRAINING_DATA_LOADING_TEST==========

    printf("=== SIMPLE TEST: Loading Training Samples  ===\n\n");
    printf("Reading MNIST dataset ....\n");
    start_time = clock();

    Sample *data = load_text_dataset("mnist_dataset.dat", params.training_samples, &params);

    if (!data)
    {
        printf("error loading data \n");
        return -1;
    }
    for (int i = 0; i < params.training_samples; i++)
    {
        int label = -1;

        // find label from one-hot vector
        for (int j = 0; j < params.output_neurons; j++)
        {
            if (data[i].one_hot_vector[j] == 1)
            {
                label = j;
                break;
            }
        }

        // fill Output[i]
        for (int j = 0; j < params.output_neurons; j++)
            Output[i][j] = (j == label) ? 1.0f : 0.0f;
            //redundant j == label and data[i].one_hot_vector[j]

        // fill Input[i]
        for (int j = 0; j < params.input_neurons; j++)
            Input[i][j] = data[i].normalized_pixels[j];
    }

    end_time = clock();
    double data_loading_time =
        ((double)(end_time - start_time)) / CLOCKS_PER_SEC;
    printf("Data loading completed in %.4f seconds\n", data_loading_time);

    printf("Initializing weights and biases...\n");

    for (int i = 0; i < params.input_neurons; i++)
        for (int j = 0; j < params.hidden_neurons_1; j++)
            weight_input_hidden1[i][j] = initialize_weight();

    for (int i = 0; i < params.hidden_neurons_1; i++)
        for (int j = 0; j < params.hidden_neurons_2; j++)
            weight_hidden1_hidden2[i][j] = initialize_weight();

    for (int i = 0; i < params.hidden_neurons_2; i++)
        for (int j = 0; j < params.output_neurons; j++)
            weight_hidden2_Output[i][j] = initialize_weight();

    for (int i = 0; i < params.hidden_neurons_1; i++)
        bias_hidden_1[i] = initialize_weight();

    for (int i = 0; i < params.hidden_neurons_2; i++)
        bias_hidden_2[i] = initialize_weight();

    for (int i = 0; i < params.output_neurons; i++)
        bias_Output[i] = initialize_weight();

    printf("Start training for %d epochs .....\n", params.epochs);
    start_time = clock();

    //========Training Loop =======
    // move malloc() outside of the loop
    // Temporary arrays for forward/backward pass
    float *out1 = (float *)malloc(params.hidden_neurons_1 * sizeof(float)); // cast (float *) aplly on all malloc() in the whole code
    float *out2 = malloc(params.hidden_neurons_1 * sizeof(float));
    float *out3 = malloc(params.hidden_neurons_2 * sizeof(float));
    float *out4 = malloc(params.hidden_neurons_2 * sizeof(float));
    float *out5 = malloc(params.output_neurons * sizeof(float));
    float *out6 = malloc(params.output_neurons * sizeof(float));

    float *out_error = malloc(params.output_neurons * sizeof(float));
    float *out_delta = malloc(params.output_neurons * sizeof(float));

    float *hidden_error_2 = malloc(params.hidden_neurons_2 * sizeof(float));
    float *hidden_delta_2 = malloc(params.hidden_neurons_2 * sizeof(float));

    float *hidden_error_1 = malloc(params.hidden_neurons_1 * sizeof(float));
    float *hidden_delta_1 = malloc(params.hidden_neurons_1 * sizeof(float));
    for (int epoch = 0; epoch < params.epochs; epoch++)
    {

        double total_error = 0.0;
        clock_t epoch_start = clock();
        

        for (int i = 0; i < params.training_samples; i++)
        {
            // tmp arrays were allocated here
            /*     // Temporary arrays for forward/backward pass
             float *out1 =(float*) malloc(params.hidden_neurons_1 * sizeof(float));// cast (float *) aplly on all malloc() in the whole code
             float *out2 = malloc(params.hidden_neurons_1 * sizeof(float));
             float *out3 = malloc(params.hidden_neurons_2 * sizeof(float));
             float *out4 = malloc(params.hidden_neurons_2 * sizeof(float));
             float *out5 = malloc(params.output_neurons * sizeof(float));
             float *out6 = malloc(params.output_neurons * sizeof(float));

             float *out_error = malloc(params.output_neurons * sizeof(float));
             float *out_delta = malloc(params.output_neurons * sizeof(float));

             float *hidden_error_2 = malloc(params.hidden_neurons_2 * sizeof(float));
             float *hidden_delta_2 = malloc(params.hidden_neurons_2 * sizeof(float));

             float *hidden_error_1 = malloc(params.hidden_neurons_1 * sizeof(float));
             float *hidden_delta_1 = malloc(params.hidden_neurons_1 * sizeof(float));
 */

            // forward pass hidden layer 1
            for (int j = 0; j < params.hidden_neurons_1; j++)
            {
                out1[j] = bias_hidden_1[j];
                for (int k = 0; k < params.input_neurons; k++)
                    out1[j] += Input[i][k] * weight_input_hidden1[k][j];
                out2[j] = sigmoid(out1[j]);
            }

            // forward pass hidden layer 2
            for (int j = 0; j < params.hidden_neurons_2; j++)
            {
                out3[j] = bias_hidden_2[j];
                for (int k = 0; k < params.hidden_neurons_1; k++)
                    out3[j] += out2[k] * weight_hidden1_hidden2[k][j];
                out4[j] = sigmoid(out3[j]);
            }
            // out1,out3,ou5 nneds to be replaced  with only out2,out4,out6
            // output layer
            for (int j = 0; j < params.output_neurons; j++)
            {
                out5[j] = bias_Output[j];
                for (int k = 0; k < params.hidden_neurons_2; k++)
                    out5[j] += out4[k] * weight_hidden2_Output[k][j];
                out6[j] = sigmoid(out5[j]);
            }

            // output layer error
            for (int j = 0; j < params.output_neurons; j++)
            {
                out_error[j] = Output[i][j] - out6[j];
                total_error += fabs(out_error[j]);
                out_delta[j] = out_error[j] * sigmoid_derivation(out6[j]);
            }

            // hidden layer 2 error
            for (int j = 0; j < params.hidden_neurons_2; j++)
            {
                hidden_error_2[j] = 0.0f;
                for (int k = 0; k < params.output_neurons; k++)
                    hidden_error_2[j] += out_delta[k] * weight_hidden2_Output[j][k];
                hidden_delta_2[j] = hidden_error_2[j] * sigmoid_derivation(out4[j]);
            }

            // hidden layer 1 error
            for (int j = 0; j < params.hidden_neurons_1; j++)
            {
                hidden_error_1[j] = 0.0f;
                for (int k = 0; k < params.hidden_neurons_2; k++)
                    hidden_error_1[j] += hidden_delta_2[k] * weight_hidden1_hidden2[j][k];
                hidden_delta_1[j] = hidden_error_1[j] * sigmoid_derivation(out2[j]);
            }

            // update weights + biases
            for (int j = 0; j < params.hidden_neurons_2; j++)
                for (int k = 0; k < params.output_neurons; k++)
                    weight_hidden2_Output[j][k] += params.learning_rate * out_delta[k] * out4[j];

            for (int j = 0; j < params.output_neurons; j++)
                bias_Output[j] += params.learning_rate * out_delta[j];

            for (int j = 0; j < params.hidden_neurons_1; j++)
                for (int k = 0; k < params.hidden_neurons_2; k++)
                    weight_hidden1_hidden2[j][k] += params.learning_rate * hidden_delta_2[k] * out2[j];

            for (int j = 0; j < params.hidden_neurons_2; j++)
                bias_hidden_2[j] += params.learning_rate * hidden_delta_2[j];

            for (int j = 0; j < params.input_neurons; j++)
                for (int k = 0; k < params.hidden_neurons_1; k++)
                    weight_input_hidden1[j][k] += params.learning_rate * hidden_delta_1[k] * Input[i][j];

            for (int j = 0; j < params.hidden_neurons_1; j++)
                bias_hidden_1[j] += params.learning_rate * hidden_delta_1[j];
        }
       
        clock_t epoch_end = clock();
        double epoch_time = ((double)(epoch_end - epoch_start)) / CLOCKS_PER_SEC;

        if (epoch % 10 == 0)
            printf("Epoch %d, Error: %.6f, Time: %.4f seconds\n",
                   epoch, total_error / params.training_samples, epoch_time);
    }
    // free temporary arrays
    free(out1);
    free(out2);
    free(out3);
    free(out4);
    free(out5);
    free(out6);
    free(out_error);
    free(out_delta);
    free(hidden_error_2);
    free(hidden_delta_2);
    free(hidden_error_1);
    free(hidden_delta_1);

    end_time = clock();
    total_computation_time = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;

    printf("\n=== TRAINING COMPLETED ===\n");
    printf("Total training time: %.4f seconds\n", total_computation_time);
    printf("Average time per epoch: %.4f seconds\n", total_computation_time / params.epochs);
    printf("Data loading time: %.4f seconds\n", data_loading_time);
    printf("Total execution time: %.4f seconds\n", total_computation_time + data_loading_time);

    // ---------------------- TESTING ----------------------
    start_time = clock();

    int total_test_samples = params.testing_samples;
    printf("\nTesting %d samples...\n\n", total_test_samples);

    for (int i = 0; i < total_test_samples; i++)
    {
        float *h1 = malloc(params.hidden_neurons_1 * sizeof(float));
        float *h2 = malloc(params.hidden_neurons_2 * sizeof(float));
        float *out_f = malloc(params.output_neurons * sizeof(float));

        for (int j = 0; j < params.hidden_neurons_1; j++)
        {
            h1[j] = bias_hidden_1[j];
            for (int k = 0; k < params.input_neurons; k++)
                h1[j] += weight_input_hidden1[k][j] * Input[i][k];
            h1[j] = sigmoid(h1[j]);
        }

        for (int j = 0; j < params.hidden_neurons_2; j++)
        {
            h2[j] = bias_hidden_2[j];
            for (int k = 0; k < params.hidden_neurons_1; k++)
                h2[j] += h1[k] * weight_hidden1_hidden2[k][j];
            h2[j] = sigmoid(h2[j]);
        }

        for (int j = 0; j < params.output_neurons; j++)
        {
            out_f[j] = bias_Output[j];
            for (int k = 0; k < params.hidden_neurons_2; k++)
                out_f[j] += h2[k] * weight_hidden2_Output[k][j];
            out_f[j] = sigmoid(out_f[j]);
        }

        int predicted = 0;
        float max_activation = out_f[0];
        for (int k = 1; k < params.output_neurons; k++)
            if (out_f[k] > max_activation)
            {
                max_activation = out_f[k];
                predicted = k;
            }

        int actual = 0;
        for (int k = 0; k < params.output_neurons; k++)
            if (Output[i][k] == 1.0)
            {
                actual = k;
                break;
            }

        int is_correct = (predicted == actual);

        printf("----------------------------------------\n");
        printf("Sample %d\n", i);
        printf("Actual: %d, Predicted: %d, Result: %s\n",
               actual, predicted,
               is_correct ? "CORRECT" : "WRONG");

        printf("Image (28x28):\n");
        printf("+----------------------------+\n");
        for (int r = 0; r < 28; r++)
        {
            printf("|");
            for (int c = 0; c < 28; c++)
            {
                float val = Input[i][r * 28 + c];
                if (val > 0.7f)
                    printf("#");
                else if (val > 0.4f)
                    printf("*");
                else if (val > 0.1f)
                    printf(".");
                else
                    printf(" ");
            }
            printf("|\n");
        }
        printf("+----------------------------+\n\n");

        free(h1);
        free(h2);
        free(out_f);
    }

    end_time = clock();
    double testing_time = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;

    printf("\n=== TESTING COMPLETED ===\n");
    printf("Total samples tested: %d\n", total_test_samples);
    printf("Total testing time: %.4f seconds\n", testing_time);
    printf("Average time per sample: %.6f seconds\n", testing_time / total_test_samples);
    printf("Total execution time (training + testing + data): %.4f seconds\n",
           total_computation_time + data_loading_time + testing_time);

    // ---------------------- FREE MEMORY ----------------------
    for (int i = 0; i < params.training_samples; i++)
    {
        free(Input[i]);
        free(Output[i]);
    }
    free(Input);
    free(Output);

    for (int i = 0; i < params.input_neurons; i++)
        free(weight_input_hidden1[i]);
    free(weight_input_hidden1);

    for (int i = 0; i < params.hidden_neurons_1; i++)
        free(weight_hidden1_hidden2[i]);
    free(weight_hidden1_hidden2);

    for (int i = 0; i < params.hidden_neurons_2; i++)
        free(weight_hidden2_Output[i]);
    free(weight_hidden2_Output);

    free(bias_hidden_1);
    free(bias_hidden_2);
    free(bias_Output);

    free(data);

    return 0;
}
