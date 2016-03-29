#include "counting.h"
#include <cstdio>
#include <iostream>
#include <cassert>
#include <thrust/scan.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h> // add 
#include <thrust/execution_policy.h>

struct head_functor
{
  __host__ __device__
  int operator()(const int& x, const int& y) const { 
        if(x == 1)
            return y;
        else
            return -1; 
    }
};

struct head_3_functor
{
  __host__ __device__
  int operator()(const int& x, const int& y) const { 
        if((x == 1) || (x == 2) || (x == 3))
            return y;
        else
            return -1; 
    }
};

__device__ __host__ int CeilDiv(int a, int b) { return (a-1)/b + 1; }
__device__ __host__ int CeilAlign(int a, int b) { return CeilDiv(a, b) * b; }

__global__ void init_tree(const char *text, thrust::device_ptr<bool> seg_tree, int text_size) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < text_size && text[idx] != '\n') 
		seg_tree[idx] = 1;
  else
    seg_tree[idx] = 0;
}

__global__ void build_tree(thrust::device_ptr<bool> seg_tree ,int num,int nodes, int start, int last_start) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
  //use share maybe TODO
  if (idx < nodes && seg_tree[last_start+2*idx] != 0 && seg_tree[last_start+2*idx+1] != 0) 
		seg_tree[start+idx] = 1;
  else
    seg_tree[start+idx] = 0;
}

__global__ void count_p(int *pos, thrust::device_ptr<bool> seg_tree, int expand_text_size ,int tree_size ,int text_size) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int previous_node = idx;
  int up_search = 0;
  int pow_2_i = 1;
  int final_node = 0;
  int layer_start = 0;
  int node_pos = 0;
  int layer_nodes = expand_text_size;
  //use share maybe TODO
  
  if(idx == 0){
    pos[0] = seg_tree[0];
  }
  else{
    //climb up seg_tree
    for (int i = 0; i<10; i++){
      if(idx/pow_2_i == 0){
        previous_node = 1;
        break;
      }
      if(seg_tree[up_search+idx/pow_2_i-1] == 0){
        if(pow_2_i == 1)
          layer_start = 0;
        else{
          layer_start = up_search - expand_text_size/(pow_2_i/2);
          layer_nodes *= 2;
        }
        break;
      }
      else{
        previous_node = up_search+idx/pow_2_i-1;
        up_search += expand_text_size/pow_2_i;
        pow_2_i *= 2;
        layer_nodes /= 2;
      }
    }
    
    node_pos = previous_node - 1 - layer_start;
    //go down seg_tree
    for (int i = 0; i<10; i++){
      if(layer_start == 0){
        if(seg_tree[node_pos] == 0)
          final_node = node_pos+1;
        else
          final_node = node_pos;
        break;
      }
      else{
        if(seg_tree[layer_start+node_pos] == 0){
          layer_nodes *=2;
          layer_start -= layer_nodes;
          node_pos = node_pos*2+1; 
        }
          //inv_count = inv_count*2+1;
        else{
          layer_nodes *=2;
          layer_start -= layer_nodes;
          node_pos = node_pos*2-1; 
        }
      }
    }
    
    if(idx < text_size){
      if(seg_tree[idx] == 0)
        pos[idx] = 0;
      else
        pos[idx] = idx-final_node+1;
    }
    
  }
}

__global__ void count_word(thrust::device_ptr<const int> pos_d ,thrust::device_ptr<int> word_d,int text_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
  //use share maybe TODO
  if (idx == text_size)
    word_d[idx] = pos_d[idx];
  else if (pos_d[idx] != 0 && pos_d[idx+1] == 0) 
    word_d[idx] = pos_d[idx];
  else
    word_d[idx] = 0;
}

void CountPosition(const char *text, int *pos, int text_size)
{
		int last_start_position = 0;
    int tree_size = 0;
    int expand_text_size = ((text_size-1)/512+1)*512;
    int start_position = expand_text_size;
    
    thrust::device_ptr<bool> seg_tree = thrust::device_malloc<bool>(expand_text_size*2-expand_text_size/512);
    init_tree<<<(text_size/512+1), 512>>>(text, seg_tree, expand_text_size);
    for(int i = 1; i <=9; i++ ){
      build_tree<<<text_size/(512*pow(2,i))+1, 512>>>(seg_tree, pow(2,i),expand_text_size/pow(2,i),start_position,last_start_position);
      last_start_position = start_position;
      start_position += (expand_text_size/pow(2,i));
    }
    tree_size = expand_text_size*2-expand_text_size/512;
    std::cout << "expand_text_size: " <<expand_text_size <<"tree_size:" << tree_size<< std::endl;
    
    
    count_p<<<expand_text_size/512, 512>>>( pos, seg_tree, expand_text_size ,tree_size ,text_size);

}

int ExtractHead(const int *pos, int *head, int text_size)
{
	//int *buffer;
	int nhead;
	//cudaMalloc(&buffer, sizeof(int)*text_size*2); // this is enough
	thrust::device_ptr<const int> pos_d(pos);
	thrust::device_ptr<int> head_d(head);//, flag_d(buffer), cumsum_d(buffer+text_size);
    thrust::device_vector<int> flag(text_size);
    thrust::device_vector<int> head_temp(text_size);
	// TODO
    thrust::sequence(flag.begin(), flag.end());
    //thrust::copy(pos_d.begin(), pos_d.end(), head_d.begin());
    nhead = thrust::count(pos_d, pos_d+text_size, 1);
    thrust::transform(pos_d, pos_d+text_size,flag.begin(), head_temp.begin() , head_functor());
    //head_d = thrust::remove(head_temp.begin(), head_temp.end(), 0);

    thrust::remove_copy(head_temp.begin(), head_temp.end(), head_d, -1);

	return nhead;
}

void Part3(char *text, int *pos, int *head, int text_size, int n_head)
{
	thrust::device_ptr<const int> pos_d(pos);
	//thrust::device_ptr<int> text_d(text);//, flag_d(buffer), cumsum_d(buffer+text_size);
    thrust::device_ptr<int> word_d = thrust::device_malloc<int>(text_size);
    thrust::device_vector<int> word_length(text_size);

    count_word<<<text_size/512+1, 512>>>(pos_d, word_d,text_size);
    thrust::remove_copy(word_d, word_d+text_size, word_length.begin(), 0);
    for(int i=0; i< 100; i++){
      std::cout << word_length[i] << " ";
    }
}