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


__device__ bool is2exp(int idx)
{
	if (idx & -idx == idx)
		return true;
	else
		return false;

}

//	thrust::device_ptr<int> WorS
__global__ void KernelSetWorS(const char *text, int *WorS, const int startup)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (text[idx] == '\n')
		WorS[idx + startup] = 0;
	else
		WorS[idx + startup] = 1;

}

__global__ void KernelMakeTree(int *BIT, int h, int startup)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
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

__global__ void KernelDecideLength(int *pos, int *BIT, int startup, int BITsize)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	idx += startup;
	pos[idx] = 0;

	int nowidx = idx;
	int lastidx = NULL;
	int updown = 0;

	int temp = 0;

	while (true)
	{
		if (updown == 0)
		{
			temp = BIT[nowidx];
			if (BIT[nowidx] == 0)
			{
				updown = 1;
			}
			else
			{
				if (nowidx % 2 == 0)
				{
					pos[idx] += temp;
					temp = 0;
					if (is2exp(idx) == false)
					{
						lastidx = nowidx;
						nowidx = nowidx - 1;
					}
					else
					{
						break;
					}
				}
				else
				{
					lastidx = nowidx;
					nowidx = nowidx / 2;
				}
			}
		}
		else
		{
			pos[idx] += BIT[nowidx];
			if (nowidx * 2 + 1 >= BITsize)
			{
				break;
			}
			else
			{
				if (BIT[nowidx] == 0)
				{
					if (lastidx == (nowidx * 2 + 1))
					{
						lastidx = nowidx;
						nowidx = nowidx * 2;
					}
					else
					{
						lastidx = nowidx;
						nowidx = (nowidx * 2 + 1);
					}
				}
				else
				{
					lastidx = nowidx;
					nowidx = (nowidx * 2 + 1);
				}
			}
		}
	}
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


	//	thrust::device_ptr<int> d_BIT = thrust::device_malloc<int>(BITsize);

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
		cudaMalloc(&d_BIT, BITtotalsize); puts(cudaGetErrorString(cudaGetLastError()));
		cudaMemset(d_BIT, 0, BITtotalsize); puts(cudaGetErrorString(cudaGetLastError()));
		cudaMemcpy(d_BIT, h_BIT, BITtotalsize, cudaMemcpyHostToDevice); puts(cudaGetErrorString(cudaGetLastError()));

		if (h == 0)
		{

			KernelSetWorS <<< blockNum, 512 >>> (text, d_BIT, startup);puts(cudaGetErrorString(cudaGetLastError()));
		}
		else
		{
			KernelMakeTree <<< blockNum, 512 >>> (d_BIT, h, startup); puts(cudaGetErrorString(cudaGetLastError()));
		}
		cudaMemcpy(h_BIT, d_BIT, BITtotalsize, cudaMemcpyDeviceToHost);
		cudaFree(d_BIT);
//		thrust::device_vector<int> temp(d_BIT + startup, d_BIT + startup + levelsize);
//		for (int i = 0; i < 1000; i++)
//		{
//			cout << temp[i];
//		}
	}

	levelsize = text_size;
	int *d_BIT;
	int level = BITheight;
	int startup = (int)pow(2, (level - 1));
	int blockNum = CeilDiv(levelsize, 512);

	cudaMalloc(&d_BIT, BITtotalsize); puts(cudaGetErrorString(cudaGetLastError()));
	cudaMemcpy(d_BIT, h_BIT, BITtotalsize, cudaMemcpyHostToDevice); puts(cudaGetErrorString(cudaGetLastError()));
	KernelDecideLength <<< blockNum, 512 >>> (pos, d_BIT, startup, BITsize); puts(cudaGetErrorString(cudaGetLastError()));
	cudaMemcpy(h_BIT, d_BIT, BITtotalsize, cudaMemcpyDeviceToHost); puts(cudaGetErrorString(cudaGetLastError()));
	cudaFree(d_BIT);

/*	for (int i = 1; i < BITsize; i++)
	{
		if (h_BIT[i] != 0)
		{
			cout << h_BIT[i] << "\t";
		}
		if ((i+1) % (int)(pow(2,(int)log2(i))) == 0)
		cout << endl;
	}*/

}


__global__ void init_tree(const char *text, thrust::device_ptr<bool> seg_tree, int text_size) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < text_size && text[idx] != '\n')
		seg_tree[idx] = 1;
	else
		seg_tree[idx] = 0;
}

__global__ void build_tree(thrust::device_ptr<bool> seg_tree, int num, int nodes, int start, int last_start) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < nodes && seg_tree[last_start + 2 * idx] != 0 && seg_tree[last_start + 2 * idx + 1] != 0)
		seg_tree[start + idx] = 1;
	else
		seg_tree[start + idx] = 0;
}

__global__ void count_p(int *pos, thrust::device_ptr<bool> seg_tree, int expand_text_size, int tree_size, int text_size) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int previous_node = idx;
	int up_search = 0;
	int pow_2_i = 1;
	int final_node = 0;
	int layer_start = 0;
	int node_pos = 0;
	int layer_nodes = expand_text_size;

	if (idx == 0){
		pos[0] = seg_tree[0];
	}
	else{
		for (int i = 0; i<10; i++){
			if (idx / pow_2_i == 0){
				previous_node = 1;
				break;
			}
			if (seg_tree[up_search + idx / pow_2_i - 1] == 0){
				if (pow_2_i == 1)
					layer_start = 0;
				else{
					layer_start = up_search - expand_text_size / (pow_2_i / 2);
					layer_nodes *= 2;
				}
				break;
			}
			else{
				previous_node = up_search + idx / pow_2_i - 1;
				up_search += expand_text_size / pow_2_i;
				pow_2_i *= 2;
				layer_nodes /= 2;
			}
		}

		node_pos = previous_node - 1 - layer_start;
		for (int i = 0; i<10; i++){
			if (layer_start == 0){
				if (seg_tree[node_pos] == 0)
					final_node = node_pos + 1;
				else
					final_node = node_pos;
				break;
			}
			else{
				if (seg_tree[layer_start + node_pos] == 0){
					layer_nodes *= 2;
					layer_start -= layer_nodes;
					node_pos = node_pos * 2 + 1;
				}
				else{
					layer_nodes *= 2;
					layer_start -= layer_nodes;
					node_pos = node_pos * 2 - 1;
				}
			}
		}

		if (idx < text_size){
			if (seg_tree[idx] == 0)
				pos[idx] = 0;
			else
				pos[idx] = idx - final_node + 1;
		}

	}
}

__global__ void count_word(thrust::device_ptr<const int> pos_d, thrust::device_ptr<int> word_d, int text_size) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	//use share maybe TODO
	if (idx == text_size)
		word_d[idx] = pos_d[idx];
	else if (pos_d[idx] != 0 && pos_d[idx + 1] == 0)
		word_d[idx] = pos_d[idx];
	else
		word_d[idx] = 0;
}

/*void CountPosition(const char *text, int *pos, int text_size)
{
	int last_start_position = 0;
	int tree_size = 0;
	int expand_text_size = ((text_size - 1) / 512 + 1) * 512;
	int start_position = expand_text_size;

	thrust::device_ptr<bool> seg_tree = thrust::device_malloc<bool>(expand_text_size * 2 - expand_text_size / 512);
	init_tree << <(text_size / 512 + 1), 512 >> >(text, seg_tree, expand_text_size);
	for (int i = 1; i <= 9; i++){
		build_tree << <text_size / (512 * pow(2, i)) + 1, 512 >> >(seg_tree, pow(2, i), expand_text_size / pow(2, i), start_position, last_start_position);
		last_start_position = start_position;
		start_position += (expand_text_size / pow(2, i));
	}
	tree_size = expand_text_size * 2 - expand_text_size / 512;
	std::cout << "expand_text_size: " << expand_text_size << "tree_size:" << tree_size << std::endl;

	count_p << <expand_text_size / 512, 512 >> >(pos, seg_tree, expand_text_size, tree_size, text_size);
}*/






struct head_functor
{
	head_functor(){}
	__host__ __device__ float operator()(const float& x, const float& y) const
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
	free(h_newtext);
	cudaFree(d_newtext);
}

