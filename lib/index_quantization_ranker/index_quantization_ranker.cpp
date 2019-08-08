#include "index_quantization_ranker.h"
#include <vector>
#include <stdint.h>
#include <iostream>
#include <cassert>
#include <iterator>
#include <algorithm>
#include "Heap.h"

IndexQuantizationRanker::IndexQuantizationRanker(uint8_t *input_codes, int base_size, int num_chunks) : base_size_(base_size), num_chunks_(num_chunks)
{
    std::copy(input_codes, input_codes + base_size * num_chunks, std::back_inserter(codes));
}

void IndexQuantizationRanker::search(float *distance_array, int batchsize, int num_chunks, int dict_size, long int *res_ranking, int batchsize2, int k)
{
    assert(batchsize == batchsize2);
    assert(num_chunks_ == num_chunks);

    float *distances = new float[batchsize * k];

    faiss::float_maxheap_array_t res = { size_t(batchsize), size_t(k), res_ranking, distances };

#pragma omp parallel for
    for (int i = 0; i < batchsize; i++) {
        const float *curr_distance_array = distance_array + i * dict_size * num_chunks;
        const uint8_t *tmpcodes = codes.data();

        long * __restrict heap_ids = res.ids + i * k;
        float * __restrict heap_dis = res.val + i * k; 

        faiss::heap_heapify<faiss::CMax<float, long> > (k, heap_dis, heap_ids);

        for (size_t j = 0; j < base_size_; j++) {
            float dis = 0;
            const float *dt = curr_distance_array;

            for (size_t m = 0; m < num_chunks_; m+=4) {
                float dism = 0;
                dism  = dt[*tmpcodes++]; dt += dict_size;
                dism += dt[*tmpcodes++]; dt += dict_size;
                dism += dt[*tmpcodes++]; dt += dict_size;
                dism += dt[*tmpcodes++]; dt += dict_size;
                dis += dism;
            }

            if (heap_dis[0] > dis) {
                faiss::heap_pop<faiss::CMax<float, long> > (k, heap_dis, heap_ids);
                faiss::heap_push<faiss::CMax<float, long> > (k, heap_dis, heap_ids, dis, j);
            }
        }

        faiss::heap_reorder<faiss::CMax<float, long> > (k, heap_dis, heap_ids);
    }


    delete [] distances;
}
