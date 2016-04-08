#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include "MyFunctions.h"
#include "CharacterList.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>

using namespace std;

static const unsigned W = 1920;
static const unsigned H = 1080;
static const int char_w = 12;
static const int char_h = 24;
static const int w_num = 160;
static const int h_num = 45;

class Rain
{
public:
	int pos;
	int leng;

	Rain()
	{
		pos = rand() % w_num;
		leng = (rand() % 10) + 5;
	}
};

__global__ void render(uint8_t *yuv, bool d_pos_occu[], int d_bright_time[], bool d_not_empty[], Character d_character_i[])
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int col = idx % w_num;
	int row = idx / w_num;

	if (col < w_num && row < h_num)
	{
		int pixel_idx_start = (char_h * W * row) + (char_w * col);
		int color_idx_start = (W * H) + (char_h/2 * W/2 * row) + (char_w/2 * col);
		//	render

		for (int i = 0; i < char_h; i++)
		{
			for (int j = 0; j < char_w; j++)
			{
				if(d_character_i[idx].bitmap[i][j] == true)
					yuv[pixel_idx_start + (W * i) + j] = 255 * d_bright_time[idx] / 20;
				else
					yuv[pixel_idx_start + (W * i) + j] = 0;
			}
		}
		for (int i = 0; i < char_h/2; i++)
		{
			for (int j = 0; j < char_w/2; j++)
			{
				if(d_character_i[idx].bitmap[i*2][j*2])
					yuv[color_idx_start + (W/2 * i) + j] = 128 - (0.331 * 255 * d_bright_time[idx] / 20) ;
				else
					yuv[color_idx_start + (W/2 * i) + j] = 128;
			}
		}
		for (int i = 0; i < char_h / 2; i++)
		{
			for (int j = 0; j < char_w / 2; j++)
			{
				if(d_character_i[idx].bitmap[i * 2][j * 2])
					yuv[color_idx_start + (W * H / 4) + (W/2 * i) + j] = 128 - (0.419 * 255 * d_bright_time[idx] / 20);
				else
					yuv[color_idx_start + (W * H / 4) + (W/2 * i) + j] = 128;
			}
		}
		
		//	prepare next round
		int temp = d_bright_time[idx];
		if (row >= 0 && row < (h_num - 1))
		{
			d_bright_time[idx + w_num] = temp;
		}

		if (row == 0)
		{
			
			if (d_bright_time[idx] > 0)
			{
				
				d_bright_time[idx] = d_bright_time[idx] - 1;
			}
		}

		//	set occupancy of every position
		if (d_bright_time[idx] > 0)
		{
			d_not_empty[col] = true;
		}

		if (d_not_empty[col] == true)
		{
			d_pos_occu[col] = true;
		}
		else
		{
			d_pos_occu[col] = false;
		}
	}
}



void RainFall(uint8_t *yuv)
{
	static bool pos_occu[w_num] = { false };
	static int bright_time[w_num * h_num] = { 0 };
	static int char_id[w_num * h_num];
	static Character character_i[w_num * h_num];
	for (int i = 0; i < (w_num * h_num); i++)
	{
		char_id[i] = rand() % 10;
		Character character_temp(char_id[i]);
		character_i[i] = character_temp;
	}
	

	int blockNum = ((w_num * h_num + 1) / 512) + 1;
	Rain *rain_i = new Rain[2];

	for (int i = 0; i < 2; i++)
	{
		if (pos_occu[rain_i[i].pos] == false)
		{
			bright_time[rain_i[i].pos] = rain_i[i].leng;
		}
	}

	 bool *d_pos_occu;
	 int *d_bright_time;
	 bool *d_not_empty;
	 Character *d_character_i;

	 cudaMalloc(&d_pos_occu, w_num * sizeof(bool));
	 cudaMalloc(&d_bright_time, w_num * h_num * sizeof(int));
	 cudaMalloc(&d_not_empty, w_num * sizeof(bool));
	 cudaMalloc(&d_character_i, w_num * h_num * sizeof(Character));

	 cudaMemset(d_pos_occu, false, w_num * sizeof(bool));
	 cudaMemset(d_bright_time, 0, w_num * h_num * sizeof(int));
	 cudaMemset(d_not_empty, false, w_num * sizeof(bool));

	 cudaMemcpy(d_pos_occu, pos_occu, w_num * sizeof(bool), cudaMemcpyHostToDevice);
	 cudaMemcpy(d_bright_time, bright_time, w_num * h_num * sizeof(int), cudaMemcpyHostToDevice);
	 cudaMemcpy(d_character_i, character_i, w_num * h_num * sizeof(Character), cudaMemcpyHostToDevice);

	render << < blockNum, 512 >> > (yuv, d_pos_occu, d_bright_time, d_not_empty, d_character_i);

	cudaMemcpy(pos_occu, d_pos_occu, w_num * sizeof(bool), cudaMemcpyDeviceToHost);
	cudaMemcpy(bright_time, d_bright_time, w_num * h_num * sizeof(int), cudaMemcpyDeviceToHost);

	cudaFree(d_pos_occu);
	cudaFree(d_bright_time);
	cudaFree(d_not_empty);
	cudaFree(d_character_i);
}