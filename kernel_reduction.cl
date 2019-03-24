typedef struct{
	float x,y;
} Point;

__kernel void func1(__global Point *centroids,
		__global Point *data,
		__global int *partitioned,
		__global float *g_distance,
		__global int *g_idx,
		__local float *l_distance,
		__local int *l_idx,
		int data_n, int class_n){

	int data_i = get_global_id(0);
	int j = get_global_id(1);

	int l_i = get_local_id(1);

	float min_dist = DBL_MAX;
	
	Point t1, t2;
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
	
		for(p = get_local_size(1) / 2; p >= 1 ; p = p >>1){
			if(l_i < p){
				if(l_distance[l_i] > l_distance[l_i + p]){
					l_distance[l_i] = l_distance[l_i + p];
					l_idx[l_i] = l_idx[l_i + p];
				}
			}
			barrier(CLK_LOCAL_MEM_FENCE);
		}
		if(l_i == 0){
			g_distance[get_group_id(1) + (get_global_size(1)/get_local_size(1))* data_i] = l_distance[0];
			g_idx[get_group_id(1) + (get_global_size(1)/get_local_size(1)) * data_i] = l_idx[0];
		}

		
		int iter;
		for(iter = 0 ; iter < (get_global_size(1) / get_local_size(1)) ; iter++){
			if(g_distance[iter] < min_dist){
				min_dist = g_distance[iter + (get_global_size(1)/get_local_size(1)) * data_i];
				partitioned[data_i] = g_idx[iter + (get_global_size(1)/get_local_size(1)) * data_i];
			}
		}
	}
}

__kernel void func2(__global Point *centroids,
		__global int *count,
		int class_n){

	int class_i = get_global_id(0);
	if(class_i < class_n){
		centroids[class_i].x = 0.0;
		centroids[class_i].y = 0.0;
		count[class_i] = 0;
	}
}

__kernel void func3(__global Point *centroids,
		__global Point *data,
		__global int* partitioned,
		__global int *count,
		__global Point *big_arr,
		int data_n){
	int data_i = get_global_id(0);
	int class_i = partitioned[data_i];

	if(data_i < data_n){
		atomic_inc(&count[class_i]);
		big_arr[data_n * class_i + count[class_i] -1] = data[data_i];
	}
}

__kernel void func4(__global Point *big_arr,
		__global int *count,
		__global Point *g_sum,
		__local Point *l_sum,
		int class_i,
		int data_n){
	int i = get_global_id(0);
	int l_i = get_local_id(0);

	l_sum[l_i].x = (i < count[class_i]) ? big_arr[data_n * class_i + i].x : 0;
	l_sum[l_i].y = (i < count[class_i]) ? big_arr[data_n * class_i + i].y : 0;
	barrier(CLK_LOCAL_MEM_FENCE);

	for(int p = get_local_size(0) / 2 ; p >= 1 ; p = p >> 1){
		if(l_i < p){
			l_sum[l_i].x += l_sum[l_i + p].x;
			l_sum[l_i].y += l_sum[l_i + p].y;
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	if(l_i == 0){
		g_sum[get_group_id(0)].x = l_sum[0].x;
		g_sum[get_group_id(0)].y = l_sum[0].y;
	}
}

__kernel void func5(__global Point *centroids,
		__global int *count,
		int class_n){
	int class_i = get_global_id(0);

	if(class_i < class_n){
		centroids[class_i].x = centroids[class_i].x / count[class_i];
		centroids[class_i].y = centroids[class_i].y / count[class_i];
	}
}
