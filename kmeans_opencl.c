#include <CL/cl.h>
#include "kmeans.h"
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <unistd.h>

#define CHECK_ERROR(err) \
	if(err != CL_SUCCESS){\
		printf("[%s:%d] OpenCL error %d\n", __FILE__, __LINE__, err);\
		exit(EXIT_FAILURE);\
	}

double get_time(){
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return (double)tv.tv_sec + (double)1e-6 * tv.tv_usec;
}

char *get_source_code(const char *file_name, size_t *len){
	char *source_code;
	size_t length;
	FILE *file = fopen(file_name, "r");
	if(file == NULL){
		printf("[%s:%d] Failed to open %s\n", __FILE__, __LINE__, file_name);
		exit(EXIT_FAILURE);
	}

	fseek(file, 0, SEEK_END);
	length = (size_t)ftell(file);
	rewind(file);

	source_code = (char*)malloc(length + 1);
	fread(source_code, length, 1, file);
	source_code[length] = '\0';

	fclose(file);

	*len = length;
	return source_code;
}
/*
 * TODO
 * Define global variables here. For example,
 * cl_platform_id platform;
 */
cl_platform_id platform;
cl_device_id device;
cl_context context;
cl_command_queue queue;
cl_program program;
char* kernel_source;
size_t kernel_source_size;
cl_kernel kernel1, kernel2, kernel3, kernel5;
cl_int err;

void kmeans_init() {
    err = clGetPlatformIDs(1, &platform, NULL);
    CHECK_ERROR(err);

    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    CHECK_ERROR(err);

    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    CHECK_ERROR(err);

    queue = clCreateCommandQueue(context, device, 0, &err);
    CHECK_ERROR(err);

    kernel_source = get_source_code("kernel.cl", &kernel_source_size);
    program = clCreateProgramWithSource(context, 1, (const char**)&kernel_source, &kernel_source_size, &err);
    CHECK_ERROR(err);

    err = clBuildProgram(program, 1, &device, "", NULL, NULL);
    if(err == CL_BUILD_PROGRAM_FAILURE){
  	size_t log_size;
	char *log;

	err = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
	CHECK_ERROR(err);

	log = (char*)malloc(log_size + 1);

	err = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
	CHECK_ERROR(err);

	log[log_size] = '\0';
	printf("Compiler error:\n%s\n", log);
	free(log);
    }
    CHECK_ERROR(err);
    
    kernel1 = clCreateKernel(program, "func1", &err);
    kernel2 = clCreateKernel(program, "func2", &err);
//    kernel3 = clCreateKernel(program, "func3", &err);
    kernel5 = clCreateKernel(program, "func5", &err);
    CHECK_ERROR(err);
}

void kmeans(int iteration_n, int class_n, int data_n, Point* centroids, Point* data, int* partitioned)
{
    cl_mem buf_centroid;
    cl_mem buf_data;
    cl_mem buf_partitioned;
    cl_mem buf_count;
    cl_mem buf_partial_distance;
    cl_mem buf_partial_idx;


    buf_centroid = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(Point) * class_n, NULL, &err);
    CHECK_ERROR(err);
    buf_data = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(Point) * data_n, NULL, &err);
    CHECK_ERROR(err);
    buf_partitioned = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int) * data_n, NULL, &err);
    CHECK_ERROR(err);
    buf_count = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int) * class_n, NULL, &err);
    CHECK_ERROR(err);


    err = clEnqueueWriteBuffer(queue, buf_centroid, CL_FALSE, 0, sizeof(Point) * class_n, centroids, 0, NULL, NULL);
    CHECK_ERROR(err);
    err = clEnqueueWriteBuffer(queue, buf_data, CL_FALSE, 0, sizeof(Point) * data_n, data, 0, NULL, NULL);
    CHECK_ERROR(err);
    err = clEnqueueWriteBuffer(queue, buf_partitioned, CL_FALSE, 0, sizeof(int) * data_n, partitioned, 0, NULL, NULL);
    CHECK_ERROR(err);


    int iter, data_i;
    int* count = (int*)malloc(sizeof(int)*class_n);
    double for_time = 0.0, reduction_time = 0.0;

    size_t global_size, local_size, num_work_groups;
    size_t global_size_arr[2], local_size_arr[2];
    
    global_size_arr[1] = class_n; global_size_arr[0] = data_n;
    local_size_arr[1] = (class_n < 256) ? class_n : 256; local_size_arr[0] = 1;
	num_work_groups = global_size_arr[1] / local_size_arr[1];

    	buf_partial_distance = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * num_work_groups*data_n, NULL, &err);
	CHECK_ERROR(err);
	buf_partial_idx = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * num_work_groups*data_n, NULL, &err);
	CHECK_ERROR(err);

   	err = clSetKernelArg(kernel1, 0, sizeof(cl_mem), &buf_centroid);
   	CHECK_ERROR(err);
   	err = clSetKernelArg(kernel1, 1, sizeof(cl_mem), &buf_data);
   	CHECK_ERROR(err);
	err = clSetKernelArg(kernel1, 2, sizeof(cl_mem), &buf_partitioned);
	CHECK_ERROR(err);
   	err = clSetKernelArg(kernel1, 3, sizeof(cl_mem), &buf_partial_distance);
   	CHECK_ERROR(err);
	err = clSetKernelArg(kernel1, 4, sizeof(cl_mem), &buf_partial_idx);
	CHECK_ERROR(err);
   	err = clSetKernelArg(kernel1, 5, sizeof(float)*local_size_arr[1], NULL);
   	CHECK_ERROR(err);
	err = clSetKernelArg(kernel1, 6, sizeof(float)*local_size_arr[1], NULL);
	CHECK_ERROR(err);
   	err = clSetKernelArg(kernel1, 7, sizeof(cl_int), &data_n);
   	CHECK_ERROR(err);
   	err = clSetKernelArg(kernel1, 8, sizeof(cl_int), &class_n);
    	CHECK_ERROR(err);

	err = clSetKernelArg(kernel2, 0, sizeof(cl_mem), &buf_centroid);
	CHECK_ERROR(err);
	err = clSetKernelArg(kernel2, 1, sizeof(cl_mem), &buf_count);
	CHECK_ERROR(err);
	err = clSetKernelArg(kernel2, 2, sizeof(cl_int), &class_n);
	CHECK_ERROR(err);
/*
	err = clSetKernelArg(kernel3, 0, sizeof(cl_mem), &buf_centroid);
	CHECK_ERROR(err);
	err = clSetKernelArg(kernel3, 1, sizeof(cl_mem), &buf_data);
	CHECK_ERROR(err);
	err = clSetKernelArg(kernel3, 2, sizeof(cl_mem), &buf_partitioned);
	CHECK_ERROR(err);
	err = clSetKernelArg(kernel3, 3, sizeof(cl_mem), &buf_count);
	CHECK_ERROR(err);
	err = clSetKernelArg(kernel3, 4, sizeof(cl_int), &data_n);
	CHECK_ERROR(err);
*/
	err = clSetKernelArg(kernel5, 0, sizeof(cl_mem), &buf_centroid);
	CHECK_ERROR(err);
	err = clSetKernelArg(kernel5, 1, sizeof(cl_mem), &buf_count);
	CHECK_ERROR(err);
	err = clSetKernelArg(kernel5, 2, sizeof(cl_int), &class_n);
	CHECK_ERROR(err);



    for(iter = 0 ; iter < iteration_n ; iter++){
	double start = get_time();
    	err = clEnqueueNDRangeKernel(queue, kernel1, 2, NULL, global_size_arr, local_size_arr, 0, NULL, NULL);
  	CHECK_ERROR(err);
	clFinish(queue);
	double end = get_time();
	reduction_time += end - start;



	global_size = class_n;
	local_size = 2;
	global_size = (global_size + local_size -1) / local_size * local_size;
    	
	err = clEnqueueNDRangeKernel(queue, kernel2, 1, NULL, &global_size, &local_size, 0, NULL, NULL);
	CHECK_ERROR(err);



	err = clEnqueueReadBuffer(queue, buf_count, CL_FALSE, 0, sizeof(int) * class_n, count, 0, NULL, NULL);
	CHECK_ERROR(err);
	err = clEnqueueReadBuffer(queue, buf_centroid, CL_FALSE, 0, sizeof(Point) * class_n, centroids, 0, NULL, NULL);
	CHECK_ERROR(err);
//	err = clEnqueueReadBuffer(queue, buf_partitioned, CL_TRUE, 0, sizeof(int) * data_n, partitioned, 0, NULL, NULL);
//	CHECK_ERROR(err);

	double start_time = get_time();
	for(data_i = 0; data_i < data_n-1 ; data_i = data_i + 2){
		centroids[partitioned[data_i]].x += data[data_i].x;
		centroids[partitioned[data_i + 1]].x += data[data_i + 1].x;
		centroids[partitioned[data_i]].y += data[data_i].y;
		centroids[partitioned[data_i + 1]].y += data[data_i + 1].y;
		count[partitioned[data_i]]++;
		count[partitioned[data_i + 1]]++;
	}

	for(;data_i < data_n ; data_i++){
		centroids[partitioned[data_i]].x += data[data_i].x;
		centroids[partitioned[data_i]].y += data[data_i].y;
		count[partitioned[data_i]]++;
	}
	double end_time = get_time();
	for_time += end_time - start_time;


	err = clEnqueueWriteBuffer(queue, buf_centroid, CL_FALSE, 0, sizeof(Point) * class_n, centroids, 0, NULL, NULL);
	CHECK_ERROR(err);
	err = clEnqueueWriteBuffer(queue, buf_count, CL_FALSE, 0, sizeof(int) * class_n, count, 0, NULL, NULL);
	CHECK_ERROR(err);

	global_size = class_n;
	local_size = 2;
	global_size = (global_size + local_size -1) / local_size * local_size;

	err = clEnqueueNDRangeKernel(queue, kernel5, 1, NULL, &global_size, &local_size, 0, NULL, NULL);
	CHECK_ERROR(err);
	err = clFinish(queue);
	CHECK_ERROR(err);
    }

    err = clEnqueueReadBuffer(queue, buf_centroid, CL_FALSE, 0, sizeof(Point) * class_n, centroids, 0, NULL, NULL);
    CHECK_ERROR(err);
    err = clEnqueueReadBuffer(queue, buf_partitioned, CL_TRUE, 0, sizeof(int) * data_n, partitioned, 0, NULL, NULL);
    CHECK_ERROR(err);

    printf("Elapsed time at reduction step : %f\n", reduction_time);
    printf("Elapsed time at for loop : %f\n",for_time);

    clReleaseMemObject(buf_centroid);
    clReleaseMemObject(buf_partitioned);
    clReleaseMemObject(buf_count);
    clReleaseMemObject(buf_data);
    clReleaseKernel(kernel1);
    clReleaseKernel(kernel2);
//    clReleaseKernel(kernel3);
    clReleaseKernel(kernel5);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

}
