%module index_quantization_ranker

%{
    #define SWIG_FILE_WITH_INIT
    #include "index_quantization_ranker.h"
%}

%include "numpy.i"

%init%{
    import_array();
%}

%apply (uint8_t* IN_ARRAY2, int DIM1, int DIM2) {(uint8_t *input_codes, int base_size, int num_chunks)}
%apply (float* IN_ARRAY3, int DIM1, int DIM2, int DIM3) {(float *distance_array, int batchsize, int num_chunks, int dict_size)}
%apply (long int* INPLACE_ARRAY2, int DIM1, int DIM2) {(long int *res_ranking, int batchsize2, int k)}

%include "index_quantization_ranker.h"
