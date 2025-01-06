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

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <iostream>

int main()
{
    // input data on host
    thrust::host_vector<point_2d> h_points(10);
    h_points[0] = point_2d{10, 15};
    h_points[1] = point_2d{46, 63};
    h_points[2] = point_2d{68, 21};
    h_points[3] = point_2d{40, 33};
    h_points[4] = point_2d{25, 54};
    h_points[5] = point_2d{15, 43};
    h_points[6] = point_2d{44, 58};
    h_points[7] = point_2d{45, 40};
    h_points[8] = point_2d{62, 69};
    h_points[9] = point_2d{53, 67};

    // build kd_tree on device with all tags initialized to zero
    thrust::device_vector<point_2d> d_points = h_points;
    int status = build_kd_tree(thrust::raw_pointer_cast(d_points.data()), d_points.size());

    if (status != 0) {
        std::cout << "build_kd_tree failed!" << std::endl;
        return 1;
    }

    // copy results back to host
    thrust::copy(d_points.begin(), d_points.end(), h_points.begin());

    // output the tree
    for (auto i = 0; i < h_points.size(); ++i)
        std::cout << "{" << h_points[i].x[0] << ", " << h_points[i].x[1] << "}" << std::endl;
    std::cout << "build_kd_tree succeeded!" << std::endl;

    return 0;
}

