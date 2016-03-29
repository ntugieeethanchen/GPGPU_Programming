#include "counting.h"
#include <cstdio>
#include <cassert>
#include <thrust/scan.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>

#include "SyncedMemory.h"
#include <deque>
#include <cmath>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <thrust/device_malloc.h>
#include <thrust/device_free.h>
#include <thrust/device_vector.h>

using namespace std;

__device__ __host__ int CeilDiv(int a, int b) { return (a - 1) / b + 1; }
__device__ __host__ int CeilAlign(int a, int b) { return CeilDiv(a, b) * b; }



//	thrust::device_ptr<int> WorS
__global__ void KernelSetWorS(const char *text, int *WorS, const int startup, int levelsize)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < levelsize)
	{
		if (text[idx] == '\n')
			WorS[idx + startup] = 0;
		else
			WorS[idx + startup] = 1;
	}
}

__global__ void KernelMakeTree(int *BIT, int h, int startup, int levelsize)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	
	if(idx < levelsize)
	{
		idx += startup;
		int tempvalue = BIT[2 * idx];
		tempvalue += BIT[2 * idx + 1];
		int levelalign = 1;
		
		for (int i = 0; i < h; i++)
		{
			levelalign *= 2;
		}
		if (tempvalue != levelalign)
			BIT[idx] = 0;
		else
			BIT[idx] = levelalign;
	}

}

__global__ void KernelDecideLength(int *pos, int *BIT, int startup, int BITsize)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	idx += startup;
	pos[idx-startup] = 0;

	int nowidx = idx;
	int lastidx = NULL;
	int updown = 0;

	int temp = 0;
	
/*	int test = 367;
	if(idx - startup == test)
	{
		printf("idx value : %d\n",idx);
		printf("level idx value : %d\n",idx - startup);
	}*/
	
	while (true)
	{
		if (updown == 0)
		{
			
			
			if (BIT[nowidx] == 0)
			{
/*				if(idx - startup == test)
				{
					printf("Stop climb \n");
				}*/
				pos[idx-startup] += temp;
				temp = 0;
				updown = 1;
			}
			else
			{
				temp = BIT[nowidx];
				if (nowidx % 2 == 0)
				{
					pos[idx-startup] += temp;
					temp = 0;
					
					
					if ((idx & -idx) != idx)
					{
/*						if(idx - startup == test)
						{
							printf("Go left \n");
						}*/
						lastidx = nowidx;
						nowidx = nowidx - 1;
					}
					else
					{
/*						if(idx - startup == test)
						{
							printf("At left end \n");
						}*/
						break;
					}
				}
				else
				{
/*					if(idx - startup == test)
					{
						printf("Go up \n");
					}*/
					lastidx = nowidx;
					nowidx = nowidx / 2;
				}
			}
		}
		else
		{
			pos[idx-startup] += BIT[nowidx];
			if (nowidx * 2 + 1 >= BITsize)
			{
/*				if(idx - startup == test)
				{
					printf("No right child \n");
				}*/
				break;
			}
			else
			{
				if (BIT[nowidx] == 0)
				{
					if (lastidx == (nowidx * 2 + 1))
					{
/*						if(idx - startup == test)
						{
							printf("Go leftchild \n");
						}*/
						lastidx = nowidx;
						nowidx = nowidx * 2;
					}
					else
					{
/*						if(idx - startup == test)
						{
							printf("Go rightchild \n");
						}*/
						lastidx = nowidx;
						nowidx = (nowidx * 2 + 1);
					}
				}
				else
				{	
/*					if(idx - startup == test)
					{
						printf("Go left \n");
					}*/
					lastidx = nowidx;
					nowidx = nowidx - 1;
				}
			}
		}
		
/*		if(idx - startup == test)
		{
			printf("Now value : %d\n",BIT[nowidx]);
			printf("Pos value : %d\n",pos[idx - startup]);
		}*/
		
	}
/*	if(idx - startup == test)
	{
		printf("BIT value : %d\n",BIT[idx]);
		printf("Pos value : %d\n",pos[idx - startup]);
	}*/
}

void CountPosition(const char *text, int *pos, int text_size)
{
	int *h_BIT;
	int height = 10;
	int BITheight = (int) ceil( log2(text_size) ) + 1;
	int BITsize = (int)pow(2, BITheight) - 1;
	h_BIT = new int[BITsize + 1];
	size_t BITtotalsize = BITsize * sizeof(int);
	memset(h_BIT,0,BITtotalsize);


	cout << "BIT size = " << BITsize << endl;
	int levelsize = text_size;
	for (int h = 0; h < height; h++, levelsize = (levelsize + 1) / 2)
	{
		int *d_BIT;

		int level = BITheight - h;
		int startup = (int) pow(2,(level - 1));
		int blockNum = CeilDiv(levelsize, 512);

		cout << "Round " << h + 1 << " ~~~" << endl;
		cout << "Level size = " << levelsize << endl;
		cout << "Block size = " << blockNum << endl;
		cout << "Level = " << level << endl;
		cout << "Startup = " << startup << endl;
		cudaMalloc(&d_BIT, BITtotalsize); 
//		puts(cudaGetErrorString(cudaGetLastError()));
		cudaMemset(d_BIT, 0, BITtotalsize); 
//		puts(cudaGetErrorString(cudaGetLastError()));
		cudaMemcpy(d_BIT, h_BIT, BITtotalsize, cudaMemcpyHostToDevice); 
//		puts(cudaGetErrorString(cudaGetLastError()));

		if (h == 0)
		{

			KernelSetWorS <<< blockNum, 512 >>> (text, d_BIT, startup, levelsize);
//			puts(cudaGetErrorString(cudaGetLastError()));
		}
		else
		{
			KernelMakeTree <<< blockNum, 512 >>> (d_BIT, h, startup, levelsize); 
//			puts(cudaGetErrorString(cudaGetLastError()));
		}
		cudaMemcpy(h_BIT, d_BIT, BITtotalsize, cudaMemcpyDeviceToHost);
		cudaFree(d_BIT);

	}

	levelsize = text_size;
	int *d_BIT;
	int level = BITheight;
	int startup = (int)pow(2, (level - 1));
	int blockNum = CeilDiv(levelsize, 512);

	cudaMalloc(&d_BIT, BITtotalsize); puts(cudaGetErrorString(cudaGetLastError()));
	cudaMemcpy(d_BIT, h_BIT, BITtotalsize, cudaMemcpyHostToDevice); 
//	puts(cudaGetErrorString(cudaGetLastError()));
	KernelDecideLength <<< blockNum, 512 >>> (pos, d_BIT, startup, BITsize); 
//	puts(cudaGetErrorString(cudaGetLastError()));
	cudaMemcpy(h_BIT, d_BIT, BITtotalsize, cudaMemcpyDeviceToHost); 
//	puts(cudaGetErrorString(cudaGetLastError()));
	cudaFree(d_BIT);
	
/*	char *h_text = new char[text_size];
	cudaMemcpy(h_text, text, text_size*sizeof(char), cudaMemcpyDeviceToHost);
	int *h_pos = new int[text_size];
	cudaMemcpy(h_pos, pos, text_size*sizeof(int), cudaMemcpyDeviceToHost);

	int newstartup = startup + 369;
	for (int i = 0; i < 369; i++)
	{
		cout << h_text[i] << "  " << h_pos[i] << "\t";
	}
	cout << endl;
	for (int i = startup; i < newstartup; i++)
	{
		cout << h_BIT[i] << "\t";
		
	}
	cout << endl;
	for (int i = startup/2; i < newstartup/2; i++)
	{
		cout << h_BIT[i] << "\t";
		
	}
	cout << endl;
	for (int i = startup/4; i < newstartup/4; i++)
	{
		cout << h_BIT[i] << "\t";
		
	}
	cout << endl;
	for (int i = startup/8; i < newstartup/8; i++)
	{
		cout << h_BIT[i] << "\t";
		
	}
	cout << endl;*/
	
}



struct head_functor
{
	head_functor(){}
	__host__ __device__ int operator()(const int& x, const int& y) const
	{
		if (x == 1)
			return y;
		else
			return -1;
	}
};

int ExtractHead(const int *pos, int *head, int text_size)
{
	int *buffer;
	int nhead;
	cudaMalloc(&buffer, sizeof(int)*text_size * 2); // this is enough
	thrust::device_ptr<const int> pos_d(pos);
	thrust::device_ptr<int> head_d(head), flag_d(buffer), cumsum_d(buffer + text_size);

	// TODO
	thrust::sequence(flag_d, flag_d + text_size, 0, 1);
	thrust::transform(pos_d, pos_d + text_size, flag_d, flag_d, head_functor());
	nhead = thrust::count(pos_d, pos_d + text_size, 1);
	thrust::remove_copy(flag_d, flag_d + text_size, head_d, -1);

	cudaFree(buffer);
	return nhead;
}

__global__ void changeChar(char* text, int* pos, char* d_newtext)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (pos[idx] % 2 == 0)
	{
		if (pos[idx] != 0 && pos[idx - 1] != 0)
		{
			d_newtext[idx - 1] = text[idx];
			d_newtext[idx] = text[idx - 1];
		}
		else
		{
			d_newtext[idx - 1] = text[idx - 1];
			d_newtext[idx] = text[idx];
		}
	}
	else
	{
		if (pos[idx] != 0 && pos[idx + 1] != 0)
		{
			d_newtext[idx + 1] = text[idx];
			d_newtext[idx] = text[idx + 1];
		}
		else
		{
			d_newtext[idx + 1] = text[idx + 1];
			d_newtext[idx] = text[idx];
		}
	}
}

void Part3(char *text, int *pos, int *head, int text_size, int n_head)
{
	char *h_newtext;
	char *d_newtext;
	size_t texttotalsize = text_size*sizeof(char);
	h_newtext = new char[text_size];
	memset(h_newtext, 0, text_size);
	cudaMalloc(&d_newtext, texttotalsize);
	cudaMemcpy(d_newtext, h_newtext, texttotalsize, cudaMemcpyHostToDevice);
	int blocknum = CeilDiv(text_size, 512);

	changeChar <<< blocknum, 512 >>> (text, pos, d_newtext);

	cudaMemcpy(h_newtext, d_newtext, texttotalsize, cudaMemcpyDeviceToHost);
	
	char *h_text = new char[text_size];
	cudaMemcpy(h_text, text, text_size*sizeof(char), cudaMemcpyDeviceToHost);
	int *h_pos = new int[text_size];
	cudaMemcpy(h_pos, pos, text_size*sizeof(int), cudaMemcpyDeviceToHost);
	for (int i = 0; i < 369; i++)
	{
		cout << h_text[i] << "  " << h_pos[i] << "\t";
	}
	cout << endl;
	cout << endl;
	for (int i = 0; i < 369; i++)
	{
		cout << h_newtext[i] << "  " << h_pos[i] << "\t";
	}
	cout << endl;
	
	free(h_newtext);
	free(h_text);
	cudaFree(d_newtext);
}

