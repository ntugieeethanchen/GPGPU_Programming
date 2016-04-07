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

class Rain
{
public:
	int pos;
	int leng;

	Rain()
	{
		pos = rand() % 53;
		leng = (rand() % 10) + 5;
	}
};

__global__ void render(uint8_t *yuv, bool d_pos_occu[], int d_bright_time[], bool d_not_empty[], Character d_character_i[])
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int col = idx % 53;
	int row = idx / 53;

	if (col < 53 && row < 20)
	{
		int pixel_idx_start = (24 * 640 * row) + (12 * col);
		int color_idx_start = (640 * 480) + (12 * 320 * row) + (6 * col);
		//	render

		for (int i = 0; i < 24; i++)
		{
			for (int j = 0; j < 12; j++)
			{
				if(d_character_i[idx].bitmap[i][j] == true)
					yuv[pixel_idx_start + (640 * i) + j] = 255 * d_bright_time[idx] / 20;
				else
					yuv[pixel_idx_start + (640 * i) + j] = 0;
			}
		}
		for (int i = 0; i < 12; i++)
		{
			for (int j = 0; j < 6; j++)
			{
				if(d_character_i[idx].bitmap[i*2][j*2])
					yuv[color_idx_start + (320 * i) + j] = 128 - (0.331 * 255 * d_bright_time[idx] / 20) ;
				else
					yuv[color_idx_start + (320 * i) + j] = 128;
			}
		}
		for (int i = 0; i < 12; i++)
		{
			for (int j = 0; j < 6; j++)
			{
				if(d_character_i[idx].bitmap[i * 2][j * 2])
					yuv[color_idx_start + (640 * 480 / 4) + (320 * i) + j] = 128 - (0.419 * 255 * d_bright_time[idx] / 20) ;
				else
					yuv[color_idx_start + (640 * 480 / 4) + (320 * i) + j] = 128;
			}
		}
		
		//	prepare next round
		int temp = d_bright_time[idx];
		if (row >= 0 && row < 19)
		{
			d_bright_time[idx + 53] = temp;
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



void RainFall(uint8_t *yuv, unsigned w, unsigned h)
{
	static bool pos_occu[53] = { false };
	static int bright_time[53 * 20] = { 0 };
	static int char_id[53*20];
	static Character character_i[53*20];
	for (int i = 0; i < (53 * 20); i++)
	{
		char_id[i] = rand() % 10;
		Character character_temp(char_id[i]);
		character_i[i] = character_temp;
	}
	

	int blockNum = ((53 * 20 + 1) / 512) + 1;
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

	cudaMalloc(&d_pos_occu, 53 * sizeof(bool));
	cudaMalloc(&d_bright_time, 53 * 20 * sizeof(int));
	cudaMalloc(&d_not_empty, 53 * sizeof(bool));
	cudaMalloc(&d_character_i, 53 * 20 * sizeof(Character));

	cudaMemset(d_pos_occu, false, 53 * sizeof(bool));
	cudaMemset(d_bright_time, 0, 53 * 20 * sizeof(int));
	cudaMemset(d_not_empty, false, 53 * sizeof(bool));

	cudaMemcpy(d_pos_occu, pos_occu, 53 * sizeof(bool), cudaMemcpyHostToDevice);
	cudaMemcpy(d_bright_time, bright_time, 53 * 20 * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_character_i, character_i, 53 * 20 * sizeof(Character), cudaMemcpyHostToDevice);

	render << < blockNum, 512 >> > (yuv, d_pos_occu, d_bright_time, d_not_empty, d_character_i);

	cudaMemcpy(pos_occu, d_pos_occu, 53 * sizeof(bool), cudaMemcpyDeviceToHost);
	cudaMemcpy(bright_time, d_bright_time, 53 * 20 * sizeof(int), cudaMemcpyDeviceToHost);

	cudaFree(d_pos_occu);
	cudaFree(d_bright_time);
	cudaFree(d_not_empty);
	cudaFree(d_character_i);
}