#include <opencv/cv.hpp>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <omp.h>

#define PATH_ROOT "e:/video/test/"
#define SINGLE_MODE 0

#define OUTPUT_WIDTH 1920
#define OUTPUT_HEIGHT 1080
#define CHANNEL 3
#define pow2(a) a*a

#define CORE_NUMBER 4

using namespace cv;

class LUTdata
{
public:
	double* ref_x;
	double* ref_y;

	LUTdata()
	{
		ref_x = NULL;
		ref_y = NULL;
	}

	LUTdata(int patch_x_size, int patch_y_size)
	{
		ref_x = new double[patch_x_size * patch_y_size];
		ref_y = new double[patch_x_size * patch_y_size];
	}

	~LUTdata()
	{
		delete [] ref_x;
		delete [] ref_y;
	}

};




void CreateVRHalfpatch_LUT(LUTdata LUT_table, Mat input_image, int patch_x_size, int patch_y_size, double alpha)
{
	/*
	LUT_table: LUT for the results. it must be pre-initialized
	input_image: the reference_image. it is only used for getting the reference dimensions
	patch_size : must match the LUT size	
	alpha: the distortion parameter	
	*/

	// this is set as static for optimization (there is no case that this will change in a single video)
	static int height = input_image.rows;
	static int width = input_image.cols;


	static double ratio_x_size = (double)patch_x_size / width;
	static int new_y_size = floor(ratio_x_size * height); //의도적인 floor function

	static double half_x = patch_x_size / 2;
	static double half_y = patch_y_size / 2;

	// the resized image (the source) will have the same width (to keep the aspect ratio) but have a different height
	static double half_resized_x = patch_x_size / 2;
	static double half_resized_y = new_y_size / 2;

	//loop

	for (int x = 0; x < patch_x_size; x++)
	{
		for (int y = 0; y < patch_y_size; y++)
		{
			double norm_x_coor, norm_y_coor;

			norm_x_coor = (x - half_x) / half_x;
			norm_y_coor = (y - half_y) / half_x; //dividing by x to make a square coordinate system

			double current_radius = sqrt(pow2(norm_x_coor) + pow2(norm_y_coor));


			double radius_ratio = (1 + alpha * pow2(current_radius));


			double new_x_coor, new_y_coor;


			new_x_coor = radius_ratio * norm_x_coor * half_x;
			new_y_coor = radius_ratio * norm_y_coor * half_x; //dividing by x to make a square coordinate system


			double ref_x, ref_y, 

			ref_x = new_x_coor + half_resized_x;
			ref_y = new_y_coor + half_resized_y;

			if (ref_x > 0 && ref_x < width && ref_y > 0 && ref_y < height)
			{
				LUT_table.ref_x[y*patch_x_size + x] = ref_x;
				LUT_table.ref_y[y*patch_x_size + x] = ref_y;
			}
			else
			{ //incase of an invalid range of the reference, a "-1 (or any negative number)" is set to show this area is a background 
				LUT_table.ref_x[y*patch_x_size + x] = -1;
				LUT_table.ref_y[y*patch_x_size + x] = -1;
			}

		}
	}


	return;
}


Mat CreateVRHalfpatch(Mat input_image, int patch_x_size, int patch_y_size, double alpha)
{
	//the new
	int height = input_image.rows;
	int width = input_image.cols;

	Mat Output_image = Mat::zeros(patch_y_size, patch_x_size, CV_8UC3);

	double ratio_x_size = (double)patch_x_size / width;
	int new_y_size = floor(ratio_x_size * height); //의도적인 floor function


	Mat resize_image;

	resize(input_image, resize_image, Size(patch_x_size, new_y_size), 0, 0, INTER_LANCZOS4);


	double half_x = patch_x_size / 2;
	double half_y = patch_y_size / 2;

	double half_resized_x = patch_x_size / 2;
	double half_resized_y = new_y_size / 2;

	//loop

	for (int x = 0; x < patch_x_size; x++) 
	{
		for (int y = 0; y < patch_y_size; y++)
		{
			double norm_x_coor, norm_y_coor;
			
			norm_x_coor = (x - half_x) / half_x;
			norm_y_coor = (y - half_y) / half_x; //dividing by x to make a square coordinate system

			double current_radius = sqrt(pow2(norm_x_coor) + pow2(norm_y_coor));


			double radius_ratio = (1 + alpha * pow2(current_radius));


			double new_x_coor, new_y_coor;


			new_x_coor = radius_ratio * norm_x_coor * half_x;
			new_y_coor = radius_ratio * norm_y_coor * half_x; //dividing by x to make a square coordinate system


			double ref_x, ref_y, ref_x_float, ref_y_float;
			int ref_x_dec, ref_y_dec;

			ref_x = new_x_coor + half_resized_x;
			ref_y = new_y_coor + half_resized_y;



			if (ref_x >= 0 && ref_x < patch_x_size - 1 && ref_y >= 0 && ref_y < new_y_size - 1)
			{
				ref_x_dec = floor(ref_x);
				ref_y_dec = floor(ref_y);

				ref_x_float = ref_x - ref_x_dec;
				ref_y_float = ref_y - ref_y_dec;

				for (int ch = 0; ch < CHANNEL; ch++)
				{
					int step_ch, step_cols;
					step_ch = resize_image.channels();
					step_cols = resize_image.cols;

					double px_data =  (1 - ref_y_float)*(1 - ref_x_float) * (double)resize_image.data[step_ch * (ref_y_dec*		step_cols + ref_x_dec		) + ch]
									+ (ref_y_float)*(1 - ref_x_float) *		(double)resize_image.data[step_ch * ((ref_y_dec+1)*	step_cols + ref_x_dec		) + ch]
									+ (1 - ref_y_float)*(ref_x_float)*		(double)resize_image.data[step_ch * (ref_y_dec*		step_cols + (ref_x_dec + 1) )	 + ch]
									+ (ref_y_float)*(ref_x_float)*			(double)resize_image.data[step_ch * ((ref_y_dec + 1)*	step_cols + (ref_x_dec + 1)) + ch];

									

					int step2_cols = Output_image.cols; 
					int step2_ch = Output_image.channels();

					
					Output_image.data[step2_ch * (y*step2_cols + x) + ch] = (uchar)round(px_data);


				}
			}
		}
	}


	return Output_image;
}

Mat VR_duplicate(Mat input_image)
{
	Mat output_image = Mat::zeros(OUTPUT_HEIGHT, OUTPUT_WIDTH, CV_8UC3);


	int half_width = OUTPUT_WIDTH / 2;

	for (int x = 0; x < half_width; x++)
	{
		for (int y = 0; y < OUTPUT_HEIGHT; y++)
		{
			for (int ch = 0; ch < CHANNEL; ch++)
			{
				output_image.data[(y*OUTPUT_WIDTH + x)*CHANNEL + ch] = input_image.data[(y*half_width + x)*CHANNEL + ch];
				output_image.data[(y*OUTPUT_WIDTH + x + half_width)*CHANNEL + ch] = input_image.data[(y*half_width + x)*CHANNEL + ch];
			}
		}
	}

	return output_image;
}

Mat VR_distortion(Mat input_image, int x_size, int y_size,double alpha)
{
	int rows = input_image.rows;
	int cols = input_image.cols;


	static int half_x_size = round(x_size / 2);

	Mat patch_image = CreateVRHalfpatch(input_image, half_x_size, y_size, alpha);


	return VR_duplicate(patch_image);
}

using namespace std;

int main(int argc, char* argv[])
{
	double alpha = 0.1;

#if SINGLE_MODE	// Single image test	
	//Mat input_image = imread(PATH_ROOT);
	Mat input_image = imread("test.jpg");
	Mat output_image;// = Mat::zeros(OUTPUT_WIDTH, OUTPUT_HEIGHT, CV_8UC3); //FULL HD로

	
	std::cout << " Processing " << std::endl;
	output_image = VR_distortion(input_image, OUTPUT_WIDTH, OUTPUT_HEIGHT,alpha);
		
	
	imshow("Original", input_image);
	imshow("VR", output_image);
	imwrite("result.png", output_image);
	waitKey(0);



#else
	Mat input_image, output_image;

	std::cout << " Processing " << std::endl;

	//omp_set_num_threads(4);


	for (int index = 1; index <= 191592; index+= 1)
	{
		ostringstream read_filename, write_filename;

		read_filename << "E:/video/noise_your_name_level3/" << setw(6) << setfill('0') << index << ".jpg";
		write_filename << "E:/video/VR/your_name/" << setw(6) << setfill('0') << index << ".png";

		cout << endl << index;

		input_image = imread(read_filename.str());

		output_image = VR_distortion(input_image, OUTPUT_WIDTH, OUTPUT_HEIGHT, alpha);

		imwrite(write_filename.str(), output_image);
	}

#endif
	std::cout << " Done " << std::endl;

	return 0;
}