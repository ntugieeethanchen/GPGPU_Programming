#include "lab2.h"

#include "MyFunctions.h"

static const unsigned W = 640;
static const unsigned H = 480;
static const unsigned NFRAME = 240;

struct Lab2VideoGenerator::Impl {
	int t = 0;
};

Lab2VideoGenerator::Lab2VideoGenerator(): impl(new Impl) {
}

Lab2VideoGenerator::~Lab2VideoGenerator() {}

void Lab2VideoGenerator::get_info(Lab2VideoInfo &info) {
	info.w = W;
	info.h = H;
	info.n_frame = NFRAME;
	// fps = 24/1 = 24
	info.fps_n = 24;
	info.fps_d = 1;
};


void Lab2VideoGenerator::Generate(uint8_t *yuv) {
/*	cudaMemset(yuv, (impl->t)*255/NFRAME, W*H);
	cudaMemset(yuv+W*H, 128, W*H/2);
	++(impl->t);*/
	static bool first_time = true;
	if (first_time)
	{
		cudaMemset(yuv, 0, W*H);
		cudaMemset(yuv + W*H, 128, W*H / 4);
		cudaMemset(yuv + (W*H) + (W*H / 4), 128, W*H / 4);
		RainFall(yuv, W, H);
		first_time = false;
	}
	else
	{
		RainFall(yuv, W, H);
	}
}
