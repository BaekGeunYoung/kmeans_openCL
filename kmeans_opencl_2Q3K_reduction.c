#include <CL/cl.h>
#include "kmeans.h"

/*
 * TODO
 * Define global variables here. For example,
 * cl_platform_id platform;
 */
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <unistd.h>

#define CHECK_ERROR(err) \
	if(err != CL_SUCCESS){ \
		printf("[%s:%d] OpenCL error %d\n", __FILE__, __LINE__, err); \
		exit(EXIT_FAILURE); \
	}

cl_int err;
cl_platform_id platform;
cl_device_id device;
cl_context context;
cl_command_queue queue[2];
cl_program program;
cl_kernel kernel_assign, kernel_clear, kernel_divide;
cl_event events1[2], events2[2], event3, events4[2];

double get_time(){
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return (double)tv.tv_sec + (double)1e-6 * tv.tv_usec;
}
double start, end;

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

  source_code = (char *)malloc(length + 1);
  fread(source_code, length, 1, file);
  source_code[length] = '\0';

  fclose(file);

  *len = length;
  return source_code;
}

void kmeans_init() {
    /*
     * TODO
     * Initialize OpenCL objects as global variables. For example,
     * clGetPlatformIDs(1, &platform, NULL);
     */
  err = clGetPlatformIDs(1, &platform, NULL);
  CHECK_ERROR(err);

  clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
  CHECK_ERROR(err);

  context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
  CHECK_ERROR(err);

  queue[0] = clCreateCommandQueue(context, device, 0, &err);
  CHECK_ERROR(err);
  queue[1] = clCreateCommandQueue(context, device, 0, &err);
  CHECK_ERROR(err);
  //queue[2] = clCreateCommandQueue(context, device, 0, &err);
  //CHECK_ERROR(err);

  size_t source_size;
  char *source_code = get_source_code("kernel.cl", &source_size);
  program = clCreateProgramWithSource(context, 1, (const char**)&source_code, &source_size, &err);
  CHECK_ERROR(err);

  err = clBuildProgram(program, 1, &device, "", NULL, NULL);
  if(err == CL_BUILD_PROGRAM_FAILURE){
      size_t log_size;
      char *log;
      clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
      log = (char *)malloc(log_size + 1);
      clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
      log[log_size] = '\0';
      printf("Compile error:\n%s\n", log);
      free(log);
  }
  CHECK_ERROR(err);

  kernel_assign = clCreateKernel(program, "assign", &err);
  CHECK_ERROR(err);
  kernel_clear = clCreateKernel(program, "clear", &err);
  CHECK_ERROR(err);
  kernel_divide = clCreateKernel(program, "divide", &err);
  CHECK_ERROR(err);
}

void kmeans(int iteration_n, int class_n, int data_n, Point* centroids, Point* data, int* partitioned)
{
    /*
     * TODO
     * Implement here.
     * See "kmeans_seq.c" if you don't know what to do.
     */
  size_t global_size_cl = class_n;
  size_t local_size_cl = 32;
  global_size_cl = (global_size_cl + local_size_cl - 1) / local_size_cl * local_size_cl;
  //size_t num_wg_cl = global_size_cl / local_size_cl;

  size_t global_size_data = data_n / 2;
  size_t local_size_data = 256;
  global_size_data = (global_size_data + local_size_data - 1) / local_size_data * local_size_data;
  //size_t num_wg_data = global_size_data / local_size_data;

  cl_mem bufCent, bufData[2], bufPart[2], bufCnt, buf_partial_distance, buf_partial_idx;
  int i; /*base_cl, offset_cl, base_data, offset_data;*/

  	bufCent = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(Point) * class_n, NULL, &err);
  	CHECK_ERROR(err);
  	bufData[0] = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(Point) * data_n / 2, NULL, &err);
  	CHECK_ERROR(err);
	bufData[1] = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(Point) * data_n / 2, NULL, &err);
	CHECK_ERROR(err);
  	bufPart[0] = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int) * data_n / 2, NULL, &err);
  	CHECK_ERROR(err);
	bufPart[1] = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int) * data_n / 2, NULL, &err);
	CHECK_ERROR(err);
  	bufCnt = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int) * class_n, NULL, &err);
  	CHECK_ERROR(err);

    //err = clEnqueueWriteBuffer(queue[0], bufData[0], CL_TRUE, 0, sizeof(Point) * data_n, data, 0, NULL, NULL);
    //CHECK_ERROR(err);
	//err = clFlush(queue[0]);
	//CHECK_ERROR(err);
  
    err = clEnqueueWriteBuffer(queue[0], bufCent, CL_TRUE, 0, sizeof(Point) * class_n, centroids, 0, NULL, NULL);
    CHECK_ERROR(err);
	err = clFlush(queue[0]);
	CHECK_ERROR(err);

	err = clEnqueueWriteBuffer(queue[0], bufData[0], CL_FALSE, 0, sizeof(Point) * data_n / 2, data, 0, NULL, &events1[0]);
	CHECK_ERROR(err);
	err = clFlush(queue[0]);
    CHECK_ERROR(err);
    //err = clEnqueueWriteBuffer(queue[0], bufCent[1], CL_FALSE, 0, sizeof(Point) * class_n, centroids, 0, NULL, &events1[2]);
    //CHECK_ERROR(err);
	err = clEnqueueWriteBuffer(queue[0], bufData[1], CL_FALSE, 0, sizeof(Point) * data_n / 2, data + (data_n / 2), 0, NULL, &events1[1]);
    CHECK_ERROR(err);
	err = clFlush(queue[0]);
	CHECK_ERROR(err);

	size_t global_size_arr[2] = {class_n, data_n / 2};
	size_t local_size_arr[2] = {(class_n < 256) ? class_n : 256, 1};
	global_size_arr[0] = (global_size_arr[0] + local_size_arr[0] -1) / local_size_arr[0] * local_size_arr[0];
	global_size_arr[1] = (global_size_arr[1] + local_size_arr[1] -1) / local_size_arr[1] * local_size_arr[1];
	size_t num_work_groups = global_size_arr[0] / local_size_arr[0];
	
	buf_partial_distance = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * num_work_groups*data_n, NULL, &err);
	buf_partial_idx = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int) * num_work_groups*data_n, NULL, &err);

    err = clSetKernelArg(kernel_assign, 0, sizeof(cl_mem), &bufCent);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel_assign, 3, sizeof(cl_mem), &buf_partial_distance);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel_assign, 4, sizeof(cl_mem), &buf_partial_idx);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel_assign, 5, sizeof(float)*local_size_arr[0], NULL);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel_assign, 6, sizeof(float)*local_size_arr[0], NULL);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel_assign, 7, sizeof(cl_int), &data_n);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel_assign, 8, sizeof(cl_int), &class_n);
    CHECK_ERROR(err);
  
    err = clSetKernelArg(kernel_clear, 0, sizeof(cl_int), &class_n);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel_clear, 1, sizeof(cl_mem), &bufCent);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel_clear, 2, sizeof(cl_mem), &bufCnt);
    CHECK_ERROR(err);

    err = clSetKernelArg(kernel_divide, 0, sizeof(cl_int), &class_n);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel_divide, 1, sizeof(cl_mem), &bufCent);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel_divide, 2, sizeof(cl_mem), &bufCnt);
    CHECK_ERROR(err);


  int j,/*class_i,*/ data_i;
  int* count = (int*)malloc(sizeof(int) * class_n);
  
  for(i = 0; i < iteration_n; i++){
    //err = clEnqueueWriteBuffer(queue, bufCent, CL_FALSE, 0, sizeof(Point) * class_n, centroids, 0, NULL, NULL);
	//CHECK_ERROR(err);
	for(j = 0; j < 2; j++){
		 /*if(j == 0){
			 base_data = 0; offset_data = data_n / 2;
		 }
		 else{
			 base_data = data_n / 2; offset_data = data_n / 2;
		 }*/
	     /*err = clSetKernelArg(kernel_assign, 2, sizeof(cl_int), &base_data);
	     CHECK_ERROR(err);
	     err = clSetKernelArg(kernel_assign, 3, sizeof(cl_int), &offset_data);
	     CHECK_ERROR(err);*/
	     //err = clSetKernelArg(kernel_assign, 2, sizeof(cl_mem), &bufCent);
	     //CHECK_ERROR(err);
	     err = clSetKernelArg(kernel_assign, 1, sizeof(cl_mem), &bufData[j]);
	     CHECK_ERROR(err);
	     err = clSetKernelArg(kernel_assign, 2, sizeof(cl_mem), &bufPart[j]);
	     CHECK_ERROR(err);
    //start = get_time();
  	     err = clEnqueueNDRangeKernel(queue[1], kernel_assign, 2, NULL, global_size_arr, local_size_arr, 1, &events1[j], &events2[j]);
  		 CHECK_ERROR(err);
		 err = clFlush(queue[1]);
		 CHECK_ERROR(err);
	}

	
      err = clEnqueueReadBuffer(queue[0], bufPart[0], CL_FALSE, 0, sizeof(int) * data_n / 2, partitioned, 1, &events2[0], NULL);
	  CHECK_ERROR(err);
	  err = clFlush(queue[0]);
	  CHECK_ERROR(err);
	  
	  err = clEnqueueReadBuffer(queue[0], bufPart[1], CL_FALSE, 0, sizeof(int) * data_n / 2, partitioned + (data_n / 2), 1, &events2[1], NULL);
	  CHECK_ERROR(err);
	  err = clFlush(queue[0]);
	  CHECK_ERROR(err);
	//end = get_time();
	//printf("kernel_assign: %f sec\n", end - start);

/*
	for(class_i = 0; class_i < class_n; class_i++){
		centroids[class_i].x = 0.0;
		centroids[class_i].y = 0.0;
		count[class_i] = 0;
	}
*/

    /*err = clEnqueueWriteBuffer(queue, bufCent, CL_FALSE, 0, sizeof(Point) * class_n, centroids, 0, NULL, NULL);
	CHECK_ERROR(err);
	err = clEnqueueWriteBuffer(queue, bufCnt, CL_FALSE, 0, sizeof(int) * class_n, count, 0, NULL, NULL);
	CHECK_ERROR(err);*/

	//start = get_time();
	err = clEnqueueNDRangeKernel(queue[1], kernel_clear, 1, NULL, &global_size_cl, &local_size_cl, 0, NULL, &event3);
	CHECK_ERROR(err);
			
  	err = clEnqueueReadBuffer(queue[0], bufCent, CL_TRUE, 0, sizeof(Point) * class_n, centroids, 1, &event3, NULL);
  	CHECK_ERROR(err);
  	err = clEnqueueReadBuffer(queue[0], bufCnt, CL_TRUE, 0, sizeof(int) * class_n, count, 1, &event3, NULL);
 	CHECK_ERROR(err);
	//end = get_time();
	//printf("kernel_clear: %f sec\n", end - start);

	start = get_time();
	for(data_i = 0; data_i < data_n; data_i++){
		centroids[partitioned[data_i]].x += data[data_i].x;
		centroids[partitioned[data_i]].y += data[data_i].y;
		count[partitioned[data_i]]++;
	}
	end = get_time();
	//printf("sum and count: %f sec\n\n", end - start);

    err = clEnqueueWriteBuffer(queue[0], bufCnt, CL_FALSE, 0, sizeof(int) * class_n, count, 0, NULL, &events4[0]);
	CHECK_ERROR(err);
	err = clEnqueueWriteBuffer(queue[0], bufCent, CL_FALSE, 0, sizeof(Point) * class_n, centroids, 0, NULL, &events4[1]);
	CHECK_ERROR(err);

	start = get_time();
	err = clEnqueueNDRangeKernel(queue[1], kernel_divide, 1, NULL, &global_size_cl, &local_size_cl, 2, events4, NULL);
	CHECK_ERROR(err);

	//err = clEnqueueReadBuffer(queue, bufCent, CL_TRUE, 0, sizeof(Point) * class_n, centroids, 0, NULL, NULL);
	//CHECK_ERROR(err);

    //err = clFinish(queue);
	//CHECK_ERROR(err);
	/*for(class_i = 0; class_i < class_n; class_i++){
		centroids[class_i].x /= count[class_i];
		centroids[class_i].y /= count[class_i];
	}*/
  }

  err = clEnqueueReadBuffer(queue[0], bufCent, CL_TRUE, 0, sizeof(Point) * class_n, centroids, 2, events4, NULL);
  CHECK_ERROR(err);
  err = clEnqueueReadBuffer(queue[0], bufPart[0], CL_TRUE, 0, sizeof(int) * data_n / 2, partitioned, 2, events4, NULL);
  CHECK_ERROR(err);
  err = clEnqueueReadBuffer(queue[0], bufPart[1], CL_TRUE, 0, sizeof(int) * data_n / 2, partitioned + (data_n / 2), 2, events4, NULL);
  CHECK_ERROR(err);

  clReleaseMemObject(bufCent);
  clReleaseMemObject(bufData[0]);
  clReleaseMemObject(bufData[1]);
  clReleaseMemObject(bufPart[0]);
  clReleaseMemObject(bufPart[1]);
  clReleaseMemObject(bufCnt);
  clReleaseKernel(kernel_assign);
  clReleaseKernel(kernel_clear);
  clReleaseKernel(kernel_divide);
  clReleaseProgram(program);
  clReleaseCommandQueue(queue[0]);
  clReleaseCommandQueue(queue[1]);
  clReleaseContext(context);
  free(count);
}
