/*
  Copyright (C) 2023  hsongxa

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

#ifndef KD_TREE_H
#define KD_TREE_H

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "point.h"

#include <thrust/tuple.h>

#include <cassert>

// clz implementations
#if defined(__GNUC__)
#define clz(x) __builtin_clz(x)
#define ctz(x) __builtin_ctz(x)
#elif defined(_MSC_VER)
#include <intrin.h>
#define clz(x) _lzcnt_u32(x)
#define ctz(x) _tzcnt_u32(x)
#endif

// The implementation follows this paper:
//
// GPU-friendly, Parallel, and (Almost-)In-Place Construction of
// Left-Balanced k-d Trees, Ingo Wald, ArXiv v4, April 4, 2023

inline int level(int node) { assert(sizeof(int) == 4); return 31 - clz(node + 1); }

int build_kd_tree(point_2d* points, unsigned int size);

__global__ void initialize_tuples(thrust::tuple<const point_2d*, int>* tuples, const point_2d* points, unsigned int size);

__global__ void update_tags(thrust::tuple<const point_2d*, int>* tuples, unsigned int size, int l, int L);

__global__ void update_points(point_2d* points, unsigned int size, const thrust::tuple<const point_2d*, int>* tuples);

__device__ inline int parent(int node) { assert(node > 0); return (node - 1) / 2; }

__device__ inline int left_child(int node) { assert(node >= 0); return 2 * node + 1; }

__device__ inline int right_child(int node) { assert(node >= 0); return 2 * node + 2; }

__device__ int subtree_size(int node, int num_nodes, int l, int L);

__device__ int subtree_begin(int node, int num_nodes, int l, int L);

template<int K>
struct less_at_level {
	less_at_level(int level) : m_level(level) {}

	__device__
	bool operator() (const thrust::tuple<const point<K>*, int>& a, const thrust::tuple<const point<K>*, int>& b) const {
		int tag_a = thrust::get<1>(a);
		int tag_b = thrust::get<1>(b);

		if (tag_a < tag_b) return true;

		if (tag_a == tag_b) {
		    int dim = m_level % K;
			auto x_a = thrust::get<0>(a)->x[dim];
			auto x_b = thrust::get<0>(b)->x[dim];
			return (x_a < x_b) ? true : false;
		}

		return false;
	}
private:
	int m_level;
};

#endif
