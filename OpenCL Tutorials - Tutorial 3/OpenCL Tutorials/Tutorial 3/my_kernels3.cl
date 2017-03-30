// minimum
__kernel void min_val(__global const float* A, __global float* B, __local float* scratch) {
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);
	int g = get_group_id(0);

	//cache all N values from global memory to local memory
	scratch[lid] = A[id];

	barrier(CLK_LOCAL_MEM_FENCE);//wait for all local threads to finish copying from global to local memory

	for (int i = 1; i < N; i *= 2) {
		if (!(lid % (i * 2)) && ((lid + i) < N)) 
		{ 
			if(scratch[lid] > scratch[lid + i]) 
			{
				scratch[lid] = scratch[lid + i]; 
			}
		}
		barrier(CLK_LOCAL_MEM_FENCE);//wait for all local threads to finish copying from global to local memory
	}

	if (!lid) {
		B[g] = scratch[lid];
	}

	//we add results from all local groups to the first element of the array
	//serial operation! but works for any group size
	//copy the cache to output array

		//atomic_min(&B[0],scratch[lid]);
	
}

// maximum (basically a copy of minimum with just the comparign symbol reversed)
__kernel void max_val(__global const float* A, __global float* B, __local float* scratch) {
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);
	int g = get_group_id(0);

	//cache all N values from global memory to local memory
	scratch[lid] = A[id];

	barrier(CLK_LOCAL_MEM_FENCE);//wait for all local threads to finish copying from global to local memory

	for (int i = 1; i < N; i *= 2) {
		if (!(lid % (i * 2)) && ((lid + i) < N)) 
		{ 
			if(scratch[lid] < scratch[lid + i]) 
			{
				scratch[lid] = scratch[lid + i]; 
			}
		}
		barrier(CLK_LOCAL_MEM_FENCE);//wait for all local threads to finish copying from global to local memory
	}


	if (!lid) {
		B[g] = scratch[lid];
	}

	//we add results from all local groups to the first element of the array
	//serial operation! but works for any group size
	//copy the cache to output array
	//if (!lid) {
	//	atomic_max(&B[0],scratch[lid]);
	//}
}

// averaging
__kernel void avg(__global const float* A, __global float* B, __local float* scratch) {
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);
	int g = get_group_id(0);

	int temp = 0;
	//cache all N values from global memory to local memory
	scratch[lid] = A[id];

	barrier(CLK_LOCAL_MEM_FENCE);//wait for all local threads to finish copying from global to local memory

	for (int i = 1; i < N; i *= 2) {
		if (!(lid % (i * 2)) && ((lid + i) < N)) 
			scratch[lid] += scratch[lid + i];
		barrier(CLK_LOCAL_MEM_FENCE);
	}
	
	if (!lid) {
		B[g] = scratch[lid];
	}
}

// standard deviation kernal
__kernel void std_dev(__global const float* A, __global float* B, float mean, __local float* scratch) {
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);
	int g = get_group_id(0);

	// cache all N values from global memory to local memory
	scratch[lid] = A[id];
	barrier(CLK_GLOBAL_MEM_FENCE); //wait for all threads to finish copying

	// working out variance
	// subtract mean from all values
	scratch[lid] = scratch[lid] - mean;
	barrier(CLK_LOCAL_MEM_FENCE); //wait for all threads to finish copying

	scratch[lid] = scratch[lid] * scratch[lid];
	barrier(CLK_LOCAL_MEM_FENCE); //wait for all threads to finish copying

	for (int i = 1; i < N; i *= 2) {
		if (!(lid % (i * 2)) && ((lid + i) < N)) 
			scratch[lid] += scratch[lid + i];
		barrier(CLK_LOCAL_MEM_FENCE);
	}
	//we add results from all local groups to the first element of the array
	//serial operation! but works for any group size
	//copy the cache to output array

	//atomic_add(&B[id],scratch[lid]);
	
	// testing buffer
	if (!lid) {
		B[g] = scratch[lid];
	}
	
}