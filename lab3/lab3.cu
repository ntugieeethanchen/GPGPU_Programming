#include <iso646.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include "lab3.h"
#include <cstdio>

__device__ __host__ int CeilDiv(int a, int b) { return (a-1)/b + 1; }
__device__ __host__ int CeilAlign(int a, int b) { return CeilDiv(a, b) * b; }

__global__ void SimpleClone(
	const float *background,
	const float *target,
	const float *mask,
	float *output,
	const int wb, const int hb, const int wt, const int ht,
	const int oy, const int ox
)
{
	const int yt = blockIdx.y * blockDim.y + threadIdx.y;
	const int xt = blockIdx.x * blockDim.x + threadIdx.x;
	const int curt = wt*yt+xt;
	if (yt < ht and xt < wt and mask[curt] > 127.0f) {
		const int yb = oy+yt, xb = ox+xt;
		const int curb = wb*yb+xb;
		if (0 <= yb and yb < hb and 0 <= xb and xb < wb) {
			output[curb*3+0] = target[curt*3+0];
			output[curb*3+1] = target[curt*3+1];
			output[curb*3+2] = target[curt*3+2];
/*			output[curb * 3 + 0] = 255;
			output[curb * 3 + 1] = 255;
			output[curb * 3 + 2] = 255;*/
		}
	}
}

__global__ void CalculateFixed(
	const float *background, 
	const float *target, 
	const float *mask,
	float *fixed,
	const int wb, const int hb, const int wt, const int ht, 
	const int oy, const int ox
	)
{
	const int yt = blockIdx.y * blockDim.y + threadIdx.y;
	const int xt = blockIdx.x * blockDim.x + threadIdx.x;
	const int curt = wt*yt + xt;
	if (yt < ht and xt < wt and mask[curt] > 127.0f) {
		const int yb = oy + yt, xb = ox + xt;
		const int curb = wb*yb + xb;
		if (0 <= yb and yb < hb and 0 <= xb and xb < wb) {
			if (yt == 0 || xt == 0 || yt == (ht - 1) || xt == (wt - 1))
			{
				fixed[curt * 3 + 0] = 0;
				fixed[curt * 3 + 1] = 0;
				fixed[curt * 3 + 2] = 0;
			}
			else
			{
				fixed[curt * 3 + 0] = 4 * target[curt * 3 + 0]
					- (target[(curt - wt) * 3 + 0] + target[(curt - 1) * 3 + 0]
					+ target[(curt + wt) * 3 + 0] + target[(curt + 1) * 3 + 0]);
				fixed[curt * 3 + 1] = 4 * target[curt * 3 + 1]
					- (target[(curt - wt) * 3 + 1] + target[(curt - 1) * 3 + 1]
					+ target[(curt + wt) * 3 + 1] + target[(curt + 1) * 3 + 1]);
				fixed[curt * 3 + 2] = 4 * target[curt * 3 + 2]
					- (target[(curt - wt) * 3 + 2] + target[(curt - 1) * 3 + 2]
					+ target[(curt + wt) * 3 + 2] + target[(curt + 1) * 3 + 2]);
			}

			if (yt == 0 || mask[curt - wt] != 255.0f)
			{
				fixed[curt * 3 + 0] += background[(curb - wb) * 3 + 0];
				fixed[curt * 3 + 1] += background[(curb - wb) * 3 + 1];
				fixed[curt * 3 + 2] += background[(curb - wb) * 3 + 2];
			}
			if (xt == 0 || mask[curt - 1] != 255.0f)
			{
				fixed[curt * 3 + 0] += background[(curb - 1) * 3 + 0];
				fixed[curt * 3 + 1] += background[(curb - 1) * 3 + 1];
				fixed[curt * 3 + 2] += background[(curb - 1) * 3 + 2];
			}
			if (yt == (ht - 1) || mask[curt + wt] != 255.0f)
			{
				fixed[curt * 3 + 0] += background[(curb + wb) * 3 + 0];
				fixed[curt * 3 + 1] += background[(curb + wb) * 3 + 1];
				fixed[curt * 3 + 2] += background[(curb + wb) * 3 + 2];
			}
			if (xt == (wt - 1) || mask[curt + 1] != 255.0f)
			{
				fixed[curt * 3 + 0] += background[(curb + 1) * 3 + 0];
				fixed[curt * 3 + 1] += background[(curb + 1) * 3 + 1];
				fixed[curt * 3 + 2] += background[(curb + 1) * 3 + 2];
			}

		}
	}
}

__global__ void PoissonImageCloningIteration(
	const float *fixed,
	const float *mask,
	const float *buf1,
	float *buf2,
	const int wt, const int ht
	)
{
	const int yt = blockIdx.y * blockDim.y + threadIdx.y;
	const int xt = blockIdx.x * blockDim.x + threadIdx.x;
	const int curt = wt*yt + xt;
	if (yt < ht and xt < wt and mask[curt] > 127.0f) {
		buf2[curt * 3 + 0] = fixed[curt * 3 + 0];
		buf2[curt * 3 + 1] = fixed[curt * 3 + 1];
		buf2[curt * 3 + 2] = fixed[curt * 3 + 2];
		if (yt != 0 && mask[curt - wt] == 255.0f)
		{
			buf2[curt * 3 + 0] += buf1[(curt - wt) * 3 + 0];
			buf2[curt * 3 + 1] += buf1[(curt - wt) * 3 + 1];
			buf2[curt * 3 + 2] += buf1[(curt - wt) * 3 + 2];
		}
		if (xt != 0 && mask[curt - 1] == 255.0f)
		{
			buf2[curt * 3 + 0] += buf1[(curt - 1) * 3 + 0];
			buf2[curt * 3 + 1] += buf1[(curt - 1) * 3 + 1];
			buf2[curt * 3 + 2] += buf1[(curt - 1) * 3 + 2];
		}
		if (yt != (ht - 1) && mask[curt + wt] == 255.0f)
		{
			buf2[curt * 3 + 0] += buf1[(curt + wt) * 3 + 0];
			buf2[curt * 3 + 1] += buf1[(curt + wt) * 3 + 1];
			buf2[curt * 3 + 2] += buf1[(curt + wt) * 3 + 2];
		}
		if (xt != (wt - 1) && mask[curt + 1] == 255.0f)
		{
			buf2[curt * 3 + 0] += buf1[(curt + 1) * 3 + 0];
			buf2[curt * 3 + 1] += buf1[(curt + 1) * 3 + 1];
			buf2[curt * 3 + 2] += buf1[(curt + 1) * 3 + 2];
		}

		buf2[curt * 3 + 0] /= 4;
		buf2[curt * 3 + 1] /= 4;
		buf2[curt * 3 + 2] /= 4;
	}
}

__global__ void ImageShrinking(
	const float *src,
	float *dst,
	const int ws, const int hs
	)
{
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (y < ((hs + 1) / 2) and x < ((ws + 1) / 2))
	{
		dst[(((ws + 1) /2)*y + x) * 3 + 0] = src[(ws*(2*y) + (2*x)) * 3 + 0];
		dst[(((ws + 1) / 2)*y + x) * 3 + 1] = src[(ws*(2 * y) + (2 * x)) * 3 + 1];
		dst[(((ws + 1) / 2)*y + x) * 3 + 2] = src[(ws*(2 * y) + (2 * x)) * 3 + 2];
	}
}

__global__ void ImageShrinkingMask(
	const float *src,
	float *dst,
	const int ws, const int hs
	)
{
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	const int x = blockIdx.x * blockDim.x + threadIdx.x;

	if (y < ((hs + 1) / 2) and x < ((ws + 1) / 2))
	{
		dst[((ws + 1) / 2)*y + x] = src[ws*(2 * y) + (2 * x)];
	}
}

__global__ void ImageUpsample(
	const float *src,
	float *dst,
	const int wd, const int hd
	)
{
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	const int x = blockIdx.x * blockDim.x + threadIdx.x;

	if (y * 2 < hd and x * 2 < wd)
	{
		dst[((wd *(y * 2)) + (x * 2)) * 3 + 0] = src[(((wd + 1) / 2) * y + x) * 3 + 0];
		dst[((wd *(y * 2)) + (x * 2)) * 3 + 1] = src[(((wd + 1) / 2) * y + x) * 3 + 1];
		dst[((wd *(y * 2)) + (x * 2)) * 3 + 2] = src[(((wd + 1) / 2) * y + x) * 3 + 2];
		if ((x * 2 + 1) < wd)
		{
			dst[((wd *(y * 2)) + (x * 2) + 1) * 3 + 0] = src[(((wd + 1) / 2) * y + x) * 3 + 0];
			dst[((wd *(y * 2)) + (x * 2) + 1) * 3 + 1] = src[(((wd + 1) / 2) * y + x) * 3 + 1];
			dst[((wd *(y * 2)) + (x * 2) + 1) * 3 + 2] = src[(((wd + 1) / 2) * y + x) * 3 + 2];
		}
		if ((y * 2 + 1) < hd)
		{
			dst[((wd *(y * 2 + 1)) + (x * 2)) * 3 + 0] = src[(((wd + 1) / 2) * y + x) * 3 + 0];
			dst[((wd *(y * 2 + 1)) + (x * 2)) * 3 + 1] = src[(((wd + 1) / 2) * y + x) * 3 + 1];
			dst[((wd *(y * 2 + 1)) + (x * 2)) * 3 + 2] = src[(((wd + 1) / 2) * y + x) * 3 + 2];
		}
		if ((x * 2 + 1) < wd and (y * 2 + 1) < hd)
		{
			dst[((wd *(y * 2 + 1)) + (x * 2) + 1) * 3 + 0] = src[(((wd + 1) / 2) * y + x) * 3 + 0];
			dst[((wd *(y * 2 + 1)) + (x * 2) + 1) * 3 + 1] = src[(((wd + 1) / 2) * y + x) * 3 + 1];
			dst[((wd *(y * 2 + 1)) + (x * 2) + 1) * 3 + 2] = src[(((wd + 1) / 2) * y + x) * 3 + 2];
		}
	}
}

void PoissonImageCloning(
	const float *background,
	const float *target,
	const float *mask,
	float *output,
	const int wb, const int hb, const int wt, const int ht,
	const int oy, const int ox
	)
{
	dim3 gdim(CeilDiv(wt, 32), CeilDiv(ht, 16)), bdim(32, 16);
	//set up
	/*	float *fixed;
		float *buf1, *buf2;
		cudaMalloc(&fixed, 3 * wt*ht*sizeof(float));
		cudaMalloc(&buf1, 3 * wt*ht*sizeof(float));
		cudaMalloc(&buf2, 3 * wt*ht*sizeof(float));

		//initialize the iteration

		CalculateFixed <<< gdim, bdim >>>(
		background, target, mask, fixed,
		wb, hb, wt, ht, oy, ox
		);
		cudaMemcpy(buf1, target, sizeof(float) * 3 * wt*ht, cudaMemcpyDeviceToDevice);

		//iterate
		//Original
		for (int i = 0; i < 10000; ++i) {
		PoissonImageCloningIteration <<<gdim, bdim >>>(
		fixed, mask, buf1, buf2, wt, ht
		);
		PoissonImageCloningIteration <<<gdim, bdim >>>(
		fixed, mask, buf2, buf1, wt, ht
		);
		}*/
	//Hierachical
	//declare
	float *background_hier[4];
	float *fixed_hier[4];
	float *mask_hier[4];
	float *buf1_hier[4];
	float *buf2_hier[4];
	int wbs[4];
	int hbs[4];
	int ws[4];
	int hs[4];

	for (int i = 0; i < 4; i++)
	{
		if (i == 0)
		{
			ws[i] = wt;
			hs[i] = ht;
			wbs[i] = wb;
			hbs[i] = hb;
		}
		else
		{
			ws[i] = (ws[i-1] + 1) / 2;
			hs[i] = (hs[i-1] + 1) / 2;
			wbs[i] = (wbs[i-1] + 1) / 2;
			hbs[i] = (hbs[i-1] + 1) / 2;
		}
	}
	//Malloc
	for (int i = 0; i < 4; i++)
	{
		cudaMalloc(&background_hier[i], 3 * wbs[i] * hbs[i] * sizeof(float));
		cudaMalloc(&fixed_hier[i], 3 * ws[i] * hs[i] * sizeof(float));
		cudaMalloc(&mask_hier[i], ws[i] * hs[i] * sizeof(float));
		cudaMalloc(&buf1_hier[i], 3 * ws[i] * hs[i] * sizeof(float));
		cudaMalloc(&buf2_hier[i], 3 * ws[i] * hs[i] * sizeof(float));
	}
	//initialize
	cudaMemcpy(background_hier[0], background, sizeof(float) * 3 * wb * hb, cudaMemcpyDeviceToDevice);
	cudaMemcpy(mask_hier[0], mask, sizeof(float) * wt*ht, cudaMemcpyDeviceToDevice);
	cudaMemcpy(buf1_hier[0], target, sizeof(float) * 3 * wt*ht, cudaMemcpyDeviceToDevice);

	for (int i = 1; i < 4; i++)
	{
		ImageShrinking << < dim3(CeilDiv(wbs[i], 32), CeilDiv(hbs[i], 16)), dim3(32, 16) >> > (background_hier[i-1], background_hier[i], wbs[i-1], hbs[i-1]);
		ImageShrinkingMask << < dim3(CeilDiv(ws[i], 32), CeilDiv(hs[i], 16)), dim3(32, 16) >> > (mask_hier[i-1], mask_hier[i], ws[i-1], hs[i-1]);
		ImageShrinking << < dim3(CeilDiv(ws[i], 32), CeilDiv(hs[i], 16)), dim3(32, 16) >> > (buf1_hier[i-1], buf1_hier[i], ws[i-1], hs[i-1]);
	}
	//fixed
	for (int i = 0; i < 4; i++)
	{
		CalculateFixed << < dim3(CeilDiv(ws[i], 32), CeilDiv(hs[i], 16)), dim3(32, 16) >> >(
			background_hier[i], buf1_hier[i], mask_hier[i], fixed_hier[i],
			wbs[i], hbs[i], ws[i], hs[i], (oy / pow(2, i)), (ox / pow(2, i))
			);
	}
	//iteration
	for (int i = 0; i < 4; ++i) {
		for (int j = 0; j < 500; ++j)
		{
			PoissonImageCloningIteration << < dim3(CeilDiv(ws[3 - i], 32), CeilDiv(hs[3 - i], 16)), dim3(32, 16) >> >(
				fixed_hier[3 - i], mask_hier[3 - i], buf1_hier[3 - i], buf2_hier[3 - i], ws[3 - i], hs[3 - i]
				);
			PoissonImageCloningIteration << < dim3(CeilDiv(ws[3 - i], 32), CeilDiv(hs[3 - i], 16)), dim3(32, 16) >> >(
				fixed_hier[3 - i], mask_hier[3 - i], buf2_hier[3 - i], buf1_hier[3 - i], ws[3 - i], hs[3 - i]
				);
		}
		if (i < 3)
		{
			ImageUpsample << < dim3(CeilDiv(ws[3 - i], 32), CeilDiv(hs[3 - i], 16)), dim3(32, 16) >> > (buf1_hier[3 - i], buf1_hier[2 - i], ws[2 - i], hs[2 - i]);
			ImageUpsample << < dim3(CeilDiv(ws[3 - i], 32), CeilDiv(hs[3 - i], 16)), dim3(32, 16) >> > (buf2_hier[3 - i], buf2_hier[2 - i], ws[2 - i], hs[2 - i]);
		}
	}

	//copy the image back
	cudaMemcpy(output, background, wb*hb*sizeof(float)*3, cudaMemcpyDeviceToDevice);
/*	SimpleClone <<< gdim, bdim >>>(
		background, buf1, mask, output,
		wb, hb, wt, ht, oy, ox
		);*/
	SimpleClone << < gdim, bdim >> >(
		background, buf1_hier[0], mask, output,
		wb, hb, wt, ht, oy, ox
		);

	//clean up
/*	cudaFree(fixed);
	cudaFree(buf1);
	cudaFree(buf2);*/
	for (int i = 0; i < 4; i++)
	{
		cudaFree(fixed_hier[i]);
		cudaFree(mask_hier[i]);
		cudaFree(buf1_hier[i]);
		cudaFree(buf2_hier[i]);
	}
}
