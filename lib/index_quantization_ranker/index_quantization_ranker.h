#ifndef INDEXQUANTIZATIONRANKER_H
#define INDEXQUANTIZATIONRANKER_H

#include <vector>
#include <stdint.h>

class IndexQuantizationRanker
{
    std::vector<uint8_t> codes;
    int base_size_;
    int num_chunks_;
public:
    IndexQuantizationRanker(uint8_t *input_codes, int base_size, int num_chunks);
    void search(float *distance_array, int batchsize, int num_chunks, int dict_size, long int *res_ranking, int batchsize2, int k);
};

#endif
