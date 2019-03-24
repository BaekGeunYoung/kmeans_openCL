__kernel void assign(__global float2 *centroids,
		__global float2 *data,
		__global int *partitioned,
		__global float *g_distance,
		__global int *g_idx,
		__local float *l_distance,
		__local int *l_idx,
		int data_n, int class_n){

	int data_i = get_global_id(1);
	int j = get_global_id(0);

	int l_i = get_local_id(0);

	float min_dist = DBL_MAX;
	
	float2 t1, t2;
	int p;

	if(j < class_n){
		t1.x = data[data_i].x - centroids[j].x;
		t1.y = data[data_i].y - centroids[j].y;
		l_distance[l_i] = t1.x * t1.x + t1.y * t1.y;
		l_idx[l_i] = l_i;
	}else{
		l_distance[l_i] = 0.0f;
		l_idx[l_i] = 0;
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	
	if(data_i < data_n){
		for(p = get_local_size(0) / 2; p >= 1 ; p = p >>1){
			if(l_i < p){
				if(l_distance[l_i] > l_distance[l_i + p]){
					l_distance[l_i] = l_distance[l_i + p];
					l_idx[l_i] = l_idx[l_i + p];
				}
			}
			barrier(CLK_LOCAL_MEM_FENCE);
		}

		if(l_i == 0){
			g_distance[get_group_id(0) + (get_global_size(0)/get_local_size(0))* data_i] = l_distance[0];
			g_idx[get_group_id(0) + (get_global_size(0)/get_local_size(0)) * data_i] = l_idx[0];
		}
		barrier(CLK_GLOBAL_MEM_FENCE);
		
		int iter;
		for(iter = 0 ; iter < (get_global_size(0) / get_local_size(0)) ; iter++){
			if(g_distance[iter] < min_dist){
				min_dist = g_distance[iter + (get_global_size(0)/get_local_size(0)) * data_i];
				partitioned[data_i] = g_idx[iter + (get_global_size(0)/get_local_size(0)) * data_i];
			}
		}
	}
}

__kernel void clear(int class_n, __global float2 *centroids, __global int *count){
  int class_i = get_global_id(0);

  if(class_i < class_n){
  	centroids[class_i].x = 0.0;
  	centroids[class_i].y = 0.0;
  	count[class_i] = 0;
  }
}


__kernel void divide(int class_n, __global float2 *centroids, __global int *count){
	int class_i = get_global_id(0);

	if(class_i < class_n){
		centroids[class_i].x /= count[class_i];
		centroids[class_i].y /= count[class_i];
	}
}
