#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <math.h>
#include "param_parser.h"
#include <mpi.h>

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

int main(int argc, char **argv)
{

    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    srand((unsigned)time(NULL) + rank);
    clock_t start_time, end_time;
    double total_computation_time = 0.0;

    DatasetParams params;

    if (rank == 0)
    {
        printf("Running with %d MPI process(es)\n\n", size);

        int result = load_dataset_parameters("mnist_bench_1_v2.cfg", &params);
        if (result != 0)
        {
            printf("ERROR: cannot load cfg file\n");
            MPI_Abort(MPI_COMM_WORLD, -1);
        }

        // printf("\n=== SUCCESS! Data extracted ===\n\n");
        printf("=== Config loaded successfully ===\n");
        printf("  input_neurons    = %d\n", params.input_neurons);
        printf("  hidden_neurons_1 = %d\n", params.hidden_neurons_1);
        printf("  hidden_neurons_2 = %d\n", params.hidden_neurons_2);
        printf("  output_neurons   = %d\n", params.output_neurons);
        printf("  learning_rate    = %.4f\n", params.learning_rate);
        printf("  epochs           = %d\n", params.epochs);
        printf("  training_samples = %d\n", params.training_samples);
        printf("  testing_samples  = %d\n\n", params.testing_samples);
    }

    MPI_Bcast(&params, sizeof(DatasetParams), MPI_BYTE, 0, MPI_COMM_WORLD);

    /* ---------------------------------------------------------
     *   weight_input_hidden1   [input_neurons   ][hidden_neurons_1]
     *   weight_hidden1_hidden2 [hidden_neurons_1][hidden_neurons_2]
     *   weight_hidden2_Output  [hidden_neurons_2][output_neurons  ]
     *   bias_hidden_1          [hidden_neurons_1]
     *   bias_hidden_2          [hidden_neurons_2]
     *   bias_Output            [output_neurons  ]
     * ---------------------------------------------------------- */
    float **weight_input_hidden1 = (float **)malloc(params.input_neurons * sizeof(float *));
    for (int i = 0; i < params.input_neurons; i++)
        weight_input_hidden1[i] = (float *)malloc(params.hidden_neurons_1 * sizeof(float));

    float **weight_hidden1_hidden2 = (float **)malloc(params.hidden_neurons_1 * sizeof(float *));
    for (int i = 0; i < params.hidden_neurons_1; i++)
        weight_hidden1_hidden2[i] = (float *)malloc(params.hidden_neurons_2 * sizeof(float));

    float **weight_hidden2_Output = (float **)malloc(params.hidden_neurons_2 * sizeof(float *));
    for (int i = 0; i < params.hidden_neurons_2; i++)
        weight_hidden2_Output[i] = (float *)malloc(params.output_neurons * sizeof(float));

    float *bias_hidden_1 = (float *)malloc(params.hidden_neurons_1 * sizeof(float));
    float *bias_hidden_2 = (float *)malloc(params.hidden_neurons_2 * sizeof(float));
    float *bias_Output = (float *)malloc(params.output_neurons * sizeof(float));

    float *Input_flat = NULL;
    float *Output_flat = NULL;

    if (rank == 0)
    {
        start_time = clock();

        Sample *data = load_text_dataset("mnist_Input.dat", "mnist_Output.dat", params.training_samples, &params);
        if (data == NULL)
        {
            printf("ERROR: cannot load .dat file\n");
            MPI_Abort(MPI_COMM_WORLD, -1);
        }

        /*for (int i = 0 ; i<params.training_samples ; i++ ){

    for (int j = 0 ; j< params.output_neurons ;j++){
        if (data[i].one_hot_vector[j] == 1)
        {
            Output[i][j] = 1;
            break;
        }
    }

    // now extract the pixels
    for (int j = 0 ; j< params.input_neurons;j++){
        Input[i][j]= data[i].normalized_pixels[j];
    }
    }*/
        // Code optimization :
        // i got rid of the other 2 D arrays Input & Output and im using only 1 D arrays instead
        Input_flat = (float *)malloc(params.training_samples * params.input_neurons * sizeof(float)); // don't forget cast
        Output_flat = (float *)malloc(params.training_samples * params.output_neurons * sizeof(float));

        for (int i = 0; i < params.training_samples; i++)
        {
            for (int j = 0; j < params.input_neurons; j++)
                Input_flat[i * params.input_neurons + j] = data[i].normalized_pixels[j];

            for (int j = 0; j < params.output_neurons; j++)
                Output_flat[i * params.output_neurons + j] = data[i].one_hot_vector[j];
        }

        end_time = clock();
        printf("Data loading : %.4f seconds\n\n",
               (double)(end_time - start_time) / CLOCKS_PER_SEC);

        free(data);
    }

    int samples_per_proc = params.training_samples / size;
    int rem = params.training_samples % size;
    int local_count = samples_per_proc + (rank < rem ? 1 : 0);

    int *displs_input = NULL;
    int *displs_output = NULL;
    int *sendcounts_input = NULL;
    int *sendcounts_output = NULL;

    if (rank == 0)
    {
        displs_input = (int *)malloc(size * sizeof(int));
        displs_output = (int *)malloc(size * sizeof(int));
        sendcounts_input = (int *)malloc(size * sizeof(int));
        sendcounts_output = (int *)malloc(size * sizeof(int));

        int offset_in = 0, offset_out = 0;
        for (int i = 0; i < size; i++)
        {
            int c = samples_per_proc + (i < rem ? 1 : 0);
            sendcounts_input[i] = c * params.input_neurons;
            sendcounts_output[i] = c * params.output_neurons;
            displs_input[i] = offset_in;
            displs_output[i] = offset_out;
            offset_in += sendcounts_input[i];
            offset_out += sendcounts_output[i];
        }
    }

    /* ----------------------------------------------------------
     * Scatter training data. After this every process holds only
     * its own slice in local_Input / local_Output.
     * ---------------------------------------------------------- */
    float *local_Input = (float *)malloc(local_count * params.input_neurons * sizeof(float));
    float *local_Output = (float *)malloc(local_count * params.output_neurons * sizeof(float));

    MPI_Scatterv(Input_flat, sendcounts_input, displs_input, MPI_FLOAT,
                 local_Input, local_count * params.input_neurons, MPI_FLOAT,
                 0, MPI_COMM_WORLD);

    MPI_Scatterv(Output_flat, sendcounts_output, displs_output, MPI_FLOAT,
                 local_Output, local_count * params.output_neurons, MPI_FLOAT,
                 0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        free(Input_flat);
        free(Output_flat);
    }

    if (rank == 0)
    {
        printf("Initializing weights and biases on root...\n");

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
    }

    for (int i = 0; i < params.input_neurons; i++)
        MPI_Bcast(weight_input_hidden1[i], params.hidden_neurons_1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    for (int i = 0; i < params.hidden_neurons_1; i++)
        MPI_Bcast(weight_hidden1_hidden2[i], params.hidden_neurons_2, MPI_FLOAT, 0, MPI_COMM_WORLD);
    for (int i = 0; i < params.hidden_neurons_2; i++)
        MPI_Bcast(weight_hidden2_Output[i], params.output_neurons, MPI_FLOAT, 0, MPI_COMM_WORLD);

    MPI_Bcast(bias_hidden_1, params.hidden_neurons_1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(bias_hidden_2, params.hidden_neurons_2, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(bias_Output, params.output_neurons, MPI_FLOAT, 0, MPI_COMM_WORLD);

    if (rank == 0)
        printf("Weights broadcast to all processes.\n\n");

    float *out1 = (float *)malloc(params.hidden_neurons_1 * sizeof(float));
    float *out2 = (float *)malloc(params.hidden_neurons_1 * sizeof(float));
    float *out3 = (float *)malloc(params.hidden_neurons_2 * sizeof(float));
    float *out4 = (float *)malloc(params.hidden_neurons_2 * sizeof(float));
    float *out5 = (float *)malloc(params.output_neurons * sizeof(float));
    float *out6 = (float *)malloc(params.output_neurons * sizeof(float));

    float *out_error = (float *)malloc(params.output_neurons * sizeof(float));
    float *out_delta = (float *)malloc(params.output_neurons * sizeof(float));

    float *hidden_error_2 = (float *)malloc(params.hidden_neurons_2 * sizeof(float));
    float *hidden_delta_2 = (float *)malloc(params.hidden_neurons_2 * sizeof(float));

    float *hidden_error_1 = (float *)malloc(params.hidden_neurons_1 * sizeof(float));
    float *hidden_delta_1 = (float *)malloc(params.hidden_neurons_1 * sizeof(float));
    // temporary updated weights and biases
    float *tmp_w1 = (float *)malloc(params.hidden_neurons_1 * sizeof(float));
    float *tmp_w2 = (float *)malloc(params.hidden_neurons_2 * sizeof(float));
    float *tmp_w3 = (float *)malloc(params.output_neurons * sizeof(float));
    float *tmp_bias1 = (float *)malloc(params.hidden_neurons_1 * sizeof(float));
    float *tmp_bias2 = (float *)malloc(params.hidden_neurons_2 * sizeof(float));
    float *tmp_bias3 = (float *)malloc(params.output_neurons * sizeof(float));

    if (rank == 0)
        printf("Starting training for %d epoch(s)...\n\n", params.epochs);

    start_time = clock();

    /* ==========================================================
     *
     * Each epoch:
     *   1. Every process runs forward + backward pass on its own
     *      local subset of samples, updating weights locally.
     *   2. After all local samples are processed, an Allreduce
     *      sums weights across all procs; dividing by size
     *      gives the global average.
     *   3. All procs now have identical averaged weights - the
     *      correct starting point for the next epoch.
     * ========================================================== */
    for (int epoch = 0; epoch < params.epochs; epoch++)
    {
        clock_t epoch_start = clock();

        /* -- Forward + Backward pass over local samples  */
        for (int i = 0; i < local_count; i++)
        {
            /* Forward: Input -> Hidden1 */
            for (int j = 0; j < params.hidden_neurons_1; j++)
            {
                out1[j] = bias_hidden_1[j];
                for (int k = 0; k < params.input_neurons; k++)
                    out1[j] += local_Input[i * params.input_neurons + k] * weight_input_hidden1[k][j];
                out2[j] = sigmoid(out1[j]);
            }

            /* Forward: Hidden1 -> Hidden2 */
            for (int j = 0; j < params.hidden_neurons_2; j++)
            {
                out3[j] = bias_hidden_2[j];
                for (int k = 0; k < params.hidden_neurons_1; k++)
                    out3[j] += out2[k] * weight_hidden1_hidden2[k][j];
                out4[j] = sigmoid(out3[j]);
            }

            /* Forward: Hidden2 -> Output */
            for (int j = 0; j < params.output_neurons; j++)
            {
                out5[j] = bias_Output[j];
                for (int k = 0; k < params.hidden_neurons_2; k++)
                    out5[j] += out4[k] * weight_hidden2_Output[k][j];
                out6[j] = sigmoid(out5[j]);

                // i got rid of  the 2 D arrays , just to make the code more optimized
                out_error[j] = local_Output[i * params.output_neurons + j] - out6[j];
                out_delta[j] = out_error[j] * sigmoid_derivation(out6[j]);
            }

            /* Backward: Hidden2 */
            for (int j = 0; j < params.hidden_neurons_2; j++)
            {
                hidden_error_2[j] = 0.0f;
                for (int k = 0; k < params.output_neurons; k++)
                    hidden_error_2[j] += out_delta[k] * weight_hidden2_Output[j][k];
                hidden_delta_2[j] = hidden_error_2[j] * sigmoid_derivation(out4[j]);
            }

            /* Backward: Hidden1 */
            for (int j = 0; j < params.hidden_neurons_1; j++)
            {
                hidden_error_1[j] = 0.0f;
                for (int k = 0; k < params.hidden_neurons_2; k++)
                    hidden_error_1[j] += hidden_delta_2[k] * weight_hidden1_hidden2[j][k];
                hidden_delta_1[j] = hidden_error_1[j] * sigmoid_derivation(out2[j]);
            }

            /* Weight update: hidden2 -> output */
            for (int j = 0; j < params.hidden_neurons_2; j++)
                for (int k = 0; k < params.output_neurons; k++)
                    weight_hidden2_Output[j][k] += params.learning_rate * out_delta[k] * out4[j];

            for (int k = 0; k < params.output_neurons; k++)
                bias_Output[k] += params.learning_rate * out_delta[k];

            /* Weight update: hidden1 -> hidden2 */
            for (int j = 0; j < params.hidden_neurons_1; j++)
                for (int k = 0; k < params.hidden_neurons_2; k++)
                    weight_hidden1_hidden2[j][k] += params.learning_rate * hidden_delta_2[k] * out2[j];

            for (int k = 0; k < params.hidden_neurons_2; k++)
                bias_hidden_2[k] += params.learning_rate * hidden_delta_2[k];

            /* Weight update: input -> hidden1 */
            for (int j = 0; j < params.input_neurons; j++)
                for (int k = 0; k < params.hidden_neurons_1; k++)
                    weight_input_hidden1[j][k] +=
                        params.learning_rate * hidden_delta_1[k] * local_Input[i * params.input_neurons + j];

            for (int k = 0; k < params.hidden_neurons_1; k++)
                bias_hidden_1[k] += params.learning_rate * hidden_delta_1[k];
        }

        /*DETAILEED EXPLANATION :

        *After each epoch, every MPI process has updated its own copy of weight_input_hidden1(for examplem ) independently using only its local samples.
       * Every process sends its row i and every process receives the same summed result in tmp_w1. This is what makes it Allreduce — not just root gets the result, everyone gets it simultaneously.
       * very Important Note : Because each process only saw a part of the training data. If process 0 saw 5 samples and process 1 saw 5 samples, their individual weight updates are each based on 5 samples.
       * Summing them gives you the equivalent of 10 samples worth of updates  which is correct.
       *  But i have 4 processes so the sum overcounts by 4.
       * Dividing by size rescales back to what a single process would have computed if it had received  all the data sequentially.

        Single process sees 20 samples  :weight update w
        4 processes each see 5 samples : each computes w/4
        not my idea , i found this in Internet , but seems  very reasonable to me !!
        */

        for (int i = 0; i < params.input_neurons; i++)
        {
            MPI_Allreduce(weight_input_hidden1[i], tmp_w1, params.hidden_neurons_1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
            for (int j = 0; j < params.hidden_neurons_1; j++)
                weight_input_hidden1[i][j] = tmp_w1[j] / (float)size; // get the average of the weight , as if it was a sequential result
        }

        for (int i = 0; i < params.hidden_neurons_1; i++)
        {
            MPI_Allreduce(weight_hidden1_hidden2[i], tmp_w2,
                          params.hidden_neurons_2, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
            for (int j = 0; j < params.hidden_neurons_2; j++)
                weight_hidden1_hidden2[i][j] = tmp_w2[j] / (float)size;
        }

        for (int i = 0; i < params.hidden_neurons_2; i++)
        {
            MPI_Allreduce(weight_hidden2_Output[i], tmp_w3,
                          params.output_neurons, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
            for (int j = 0; j < params.output_neurons; j++)
                weight_hidden2_Output[i][j] = tmp_w3[j] / (float)size;
        }

        MPI_Allreduce(bias_hidden_1, tmp_bias1, params.hidden_neurons_1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(bias_hidden_2, tmp_bias2, params.hidden_neurons_2, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(bias_Output, tmp_bias3, params.output_neurons, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);

        for (int i = 0; i < params.hidden_neurons_1; i++)
            bias_hidden_1[i] = tmp_bias1[i] / (float)size;
        for (int i = 0; i < params.hidden_neurons_2; i++)
            bias_hidden_2[i] = tmp_bias2[i] / (float)size;
        for (int i = 0; i < params.output_neurons; i++)
            bias_Output[i] = tmp_bias3[i] / (float)size;

        if (rank == 0)
        {
            clock_t epoch_end = clock();
            double epoch_time = (double)(epoch_end - epoch_start) / CLOCKS_PER_SEC;
            total_computation_time += epoch_time;

            printf("Epoch %4d / %d  |  Time: %.4fs\n",
                   epoch + 1,
                   params.epochs,
                   epoch_time);
        }

    } /* end epoch loop */

    if (rank == 0)
    {
        printf("\n=== TRAINING COMPLETED ===\n");
        printf("Total training time    : %.4f seconds\n", total_computation_time);
        printf("Average time per epoch : %.4f seconds\n",
               total_computation_time / params.epochs);
    }

    /* ==========================================================
     * TESTING PHASE

     * ========================================================== */
    if (rank == 0)
    {
        start_time = clock();

        printf("\n=== TESTING PHASE ===\n");
        printf("Loading %d test samples...\n\n", params.testing_samples);

        Sample *test_data = load_text_dataset("mnist_test_labels.dat",
                                              "mnist_test_pixels",
                                              params.testing_samples,
                                              &params);
        if (test_data == NULL)
        {
            printf("ERROR: cannot load test .dat files\n");
            MPI_Abort(MPI_COMM_WORLD, -1);
        }

        /* Allocate forward-pass buffers  */
        float *h1 = (float *)malloc(params.hidden_neurons_1 * sizeof(float));
        float *h2 = (float *)malloc(params.hidden_neurons_2 * sizeof(float));
        float *out_f = (float *)malloc(params.output_neurons * sizeof(float));

        int correct = 0;

        for (int i = 0; i < params.testing_samples; i++)
        {
            /* Forward: Input -> Hidden1 */
            for (int j = 0; j < params.hidden_neurons_1; j++)
            {
                h1[j] = bias_hidden_1[j];
                for (int k = 0; k < params.input_neurons; k++)
                    h1[j] += test_data[i].normalized_pixels[k] *
                             weight_input_hidden1[k][j];
                h1[j] = sigmoid(h1[j]);
            }

            /* Forward: Hidden1 -> Hidden2 */
            for (int j = 0; j < params.hidden_neurons_2; j++)
            {
                h2[j] = bias_hidden_2[j];
                for (int k = 0; k < params.hidden_neurons_1; k++)
                    h2[j] += h1[k] * weight_hidden1_hidden2[k][j];
                h2[j] = sigmoid(h2[j]);
            }

            /* Forward: Hidden2 -> Output */
            for (int j = 0; j < params.output_neurons; j++)
            {
                out_f[j] = bias_Output[j];
                for (int k = 0; k < params.hidden_neurons_2; k++)
                    out_f[j] += h2[k] * weight_hidden2_Output[k][j];
                out_f[j] = sigmoid(out_f[j]);
            }

            /* Predicted class = index of highest output neuron */
            int predicted = 0;
            float max_activation = out_f[0];
            for (int k = 1; k < params.output_neurons; k++)
                if (out_f[k] > max_activation)
                {
                    max_activation = out_f[k];
                    predicted = k;
                }

            /* Actual class = index of the 1 in the one-hot vector */

            // no need to use the 2 D array
            int actual = 0;
            for (int k = 0; k < params.output_neurons; k++)
                if (test_data[i].one_hot_vector[k] == 1)
                {
                    actual = k;
                    break;
                }

            int is_correct = (predicted == actual);
            if (is_correct)
                correct++;

            printf("----------------------------------------\n");
            printf("Sample %d\n", i + 1);
            printf("Actual: %d  |  Predicted: %d  |  %s\n",
                   actual, predicted,
                   is_correct ? "CORRECT" : "WRONG");

            /* ASCII of the 28x28 image */
            printf("Image (28x28):\n");
            printf("+----------------------------+\n");
            for (int r = 0; r < 28; r++)
            {
                printf("|");
                for (int c = 0; c < 28; c++)
                {
                    float val = test_data[i].normalized_pixels[r * 28 + c];
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

        } /* end testing loop */

        end_time = clock();
        double testing_time = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;

        printf("\n=== TESTING COMPLETED ===\n");
        printf("Samples tested  : %d\n", params.testing_samples);
        printf("Correct         : %d\n", correct);
        printf("Wrong           : %d\n", params.testing_samples - correct);
        printf("Testing time    : %.4f seconds\n", testing_time);

        free(h1);
        free(h2);
        free(out_f);
        free(test_data);
    }
    free(tmp_w1);
    free(tmp_w2);
    free(tmp_w3);
    free(tmp_bias1);
    free(tmp_bias2);
    free(tmp_bias3);

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

    free(local_Input);
    free(local_Output);

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

    if (rank == 0)
    {
        free(displs_input);
        free(displs_output);
        free(sendcounts_input);
        free(sendcounts_output);
    }

    MPI_Finalize();
    return 0;
}