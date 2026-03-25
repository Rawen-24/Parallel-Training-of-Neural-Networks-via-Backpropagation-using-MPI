#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "param_parser.h"


void skip_spaces(char **p)
{
    while ((**p == ' ' || **p == '\t') && **p != '\n' && **p != 0)
        (*p)++;
}

void forward_to_next_line(char **p)
{
    while (**p != '\n' && **p != 0)
        (*p)++;
    if (**p == '\n')
        (*p)++;
}

void skip_comments(char **p)
{
    skip_spaces(p);
    while (**p == '#')
    {
        forward_to_next_line(p);
        skip_spaces(p);
    }
}

long get_long(char **p)
{
    long value;
    char *endptr;

    skip_comments(p);
    errno = 0; /* To distinguish success/failure after call */
    value = strtol(*p, &endptr, 10);

    /* Check for various possible errors. */
    if (errno != 0)
    {
        perror("strtol");
        exit(EXIT_FAILURE);
    }

    *p = endptr;
    if (**p == '\n')
        (*p)++;
    return value;
}

float get_float(char **p)
{
    float value;
    char *endptr;

    skip_comments(p);
    errno = 0;
    value = strtof(*p, &endptr);

    if (errno != 0)
    {
        perror("strtof");
        exit(EXIT_FAILURE);
    }

    *p = endptr;
    if (**p == '\n')
        (*p)++;
    return value;
}

char *find_eol(char *p)
{
    while (*p != '\n' && *p != 0)
        p++;
    return p;
}

char *get_string(char **p)
{
    char *s, *eol;
    int n;

    skip_comments(p);
    eol = find_eol(*p);
    n = eol - *p;

    s = (char *)malloc((n + 1) * sizeof(char));
    s[n] = 0;
    strncpy(s, *p, n);

    *p = eol;
    if (**p == '\n')
        (*p)++;
    return s;
}

// Load dataset parameters from .cfg
int load_dataset_parameters(const char *filename, DatasetParams *params)
{
    FILE *f = fopen(filename, "r");
    if (!f)
    {
        perror("Cannot open cfg file");
        return -1;
    }

    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);

    char *buf = malloc(size + 1);
    fread(buf, 1, size, f);// raeds up to the size Objects into the buffer 
    buf[size] = 0;
    fclose(f);

    char *p = buf;
    params->input_neurons = get_long(&p);
    params->output_neurons = get_long(&p);
    params->hidden_neurons_1 = get_long(&p);
    params->hidden_neurons_2 = get_long(&p);
    params->learning_rate = get_float(&p);
    params->epochs = get_long(&p);
    params->training_samples = get_long(&p);
    params->testing_samples = get_long(&p);

    char *tmp = get_string(&p);
    strncpy(params->data_file, tmp, 255);
    params->data_file[255] = 0;

    free(tmp);
    free(buf);

    return 0; 
}


Sample *load_text_dataset(const char *filename, int samples, DatasetParams *params)
{
    FILE *f = fopen(filename, "r");
    if (!f)
    {
        perror("Cannot open data file");
        return NULL;
    }

    Sample *arr = malloc(samples * sizeof(Sample));// allocate arr of samples 
    if (!arr)
    {
        perror("Memory allocation failed");
        fclose(f);
        return NULL;
    }

    for (int i = 0; i < samples; i++)// iterate over each Sample 
    // each Sample has one hot encoded label & a normalized array of Pixel Values 
    {
        arr[i].one_hot_vector = malloc(params->output_neurons * sizeof(int)); // allocate memory for the one_hot encoded Label / vector
        arr[i].normalized_pixels = malloc(params->input_neurons * sizeof(float)); // allocate memory for  normalized pixels array 

        // Read one-hot vector
        for (int j = 0; j < params->output_neurons; j++)
            fscanf(f, "%d", &arr[i].one_hot_vector[j]);

        // Read normalized pixel values
        for (int j = 0; j < params->input_neurons; j++)
            fscanf(f, "%f", &arr[i].normalized_pixels[j]);
    }

    fclose(f);
    return arr;
}
