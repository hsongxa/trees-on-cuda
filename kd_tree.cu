/*
  Copyright (C) 2025  hsongxa

  This program is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#include "kd_tree.cuh"

#include <thrust/device_vector.h>
#include <thrust/sort.h>

#include <cassert>
#include <cmath>
#include <bit>
#include <iostream>

__device__ int subtree_size(int node, int num_nodes, int l, int L)
{
    // it is legal to query on node >= num_nodes 
    assert(node >= 0 && num_nodes > 0);

    int first_lowest_level_child = ~((~node) << (L - l - 1));
    int full_lowest_level_size = (1 << (L - l - 1));
    return full_lowest_level_size - 1 +
           thrust::min(thrust::max(0, num_nodes - first_lowest_level_child),
                       full_lowest_level_size);
}

__device__ int subtree_begin(int node, int num_nodes, int l, int L)
{
    // it is legal to query on node >= num_nodes 
    assert(node >= 0 && num_nodes > 0);

    int num_left_siblings = node - ((1 << l) - 1);
    int full_lowest_level_size = (1 << (L - l - 1));

    return (1 << l) - 1 +
           num_left_siblings * (full_lowest_level_size - 1) +
           thrust::min(num_left_siblings * full_lowest_level_size,
                       num_nodes - ((1 << (L - 1)) - 1));
}

__global__ void initialize_tuples(thrust::tuple<const point_2d*, int>* tuples, const point_2d* points, unsigned int size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        thrust::get<0>(tuples[i]) = &points[i];
        thrust::get<1>(tuples[i]) = 0;
    }
}

__global__ void update_tags(thrust::tuple<const point_2d*, int>* tuples, unsigned int size, int l, int L)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size || i < ((1 << l) - 1)) return;

    int current_tag = thrust::get<1>(tuples[i]);
    int pivot_pos = subtree_begin(current_tag, size, l, L) + subtree_size(left_child(current_tag), size, l + 1, L);
    if (i < pivot_pos)
        thrust::get<1>(tuples[i]) = left_child(current_tag);
    else if (i > pivot_pos)
        thrust::get<1>(tuples[i]) = right_child(current_tag);
}

__global__ void update_points(point_2d* points, unsigned int size, const thrust::tuple<const point_2d*, int>* tuples)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) points[i] = *thrust::get<0>(tuples[i]);
}

int build_kd_tree(point_2d* points, unsigned int size)
{
    // the algorithm operates on tuples of point_2d and its tag
    thrust::device_vector<thrust::tuple<const point_2d*, int>> d_tuples(size); 

    const int BLOCK_SIZE = 32;
    const int NUM_BLOCKS = (size - 1) / BLOCK_SIZE + 1;
    initialize_tuples<<<NUM_BLOCKS, BLOCK_SIZE>>>(thrust::raw_pointer_cast(d_tuples.data()), points, size);

    int L = std::bit_width(size); // TODO: no need to pass L down to the kernel -- device code can use __clz(or __clzll) to get this from size (i.e., num_nodes)
    for (int l = 0; l < L; ++l) {
        int skip_sorting = l == 0 ? 0 : (1 << (l - 1)) - 1;
        thrust::stable_sort(d_tuples.begin() + skip_sorting, d_tuples.end(), less_at_level<2>(l));
        if (l < L - 1) // skip the lowest level update
            update_tags<<<NUM_BLOCKS, BLOCK_SIZE>>>(thrust::raw_pointer_cast(d_tuples.data()), size, l, L);
    }

    update_points<<<NUM_BLOCKS, BLOCK_SIZE>>>(points, size, thrust::raw_pointer_cast(d_tuples.data()));

    // Check for any errors launching the kernel
    cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        std::cout << "addKernel launch failed: " << cudaGetErrorString(cudaStatus) << std::endl;
        return 1;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        std::cout << "cudaDeviceSynchronize returned error code " << cudaStatus << " after launching addKernel!" << std::endl;
        return 2;
    }

    return 0;
}
