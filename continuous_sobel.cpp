// A simple MPI code printing a message by each MPI rank

#include <iostream>
#include <mpi.h>
#include <opencv2/opencv.hpp>
#include <chrono>

using namespace cv;
using namespace std;

/* Filters */
static int sobel_x_filter[9] = { -1, 0,+1  ,-2, 0,+2  ,-1, 0,+1 };
static int sobel_y_filter[9] = { -1,-2,-1  , 0, 0, 0  , 1, 2,+1 };

/* Structure describing the image */
struct image_t {
	int xsize;
	int ysize;
	int depth;	/* bytes */
	char* pixels;
};

struct convolve_data_t {
	struct image_t* input;
	struct image_t* output;
	int* filter;
	int ystart;
	int yend;
};

struct combine_data_t {
	struct image_t* sobelx;
	struct image_t* sobely;
	struct image_t* output;
	int ystart;
	int yend;
	int rank;
};


/* very inefficient convolve code */
void generic_convolve(struct convolve_data_t* data) {
	int x, y, k, l, d;
	uint32_t color;
	int sum, depth, width;

	struct image_t* input;
	struct image_t* output;
	int* filter;
	
	int ystart, yend;

	/* Convert from void pointer to the actual data type */
	input = data->input;
	output = data->output;
	filter = data->filter;

	ystart = data->ystart;
	yend = data->yend;

	depth = input->depth;
	width = input->xsize * depth;
	//printf("here");
	//printf("start looping, start:%d, stop%d\n", ystart, yend);
	//printf("ysize: %d\n", input->ysize);
	/* handle border */
	if (ystart == 0) ystart = 1;
	if (yend == input->ysize) yend = input->ysize - 1;

	for (d = 0; d < depth; d++) {
		for (x = 1; x < input->xsize - 1; x++) {
			for (y = ystart; y < yend; y++) {
				sum = 0;
				for (k = -1; k < 2; k++) {
					for (l = -1; l < 2; l++) {
						color = input->pixels[((y + l) * width) + (x * depth + d + k * depth)];
						sum += color * filter[(l + 1) * 3 + (k + 1)];
					}
				}
				if (sum < 0) sum = 0;
				if (sum > 255) sum = 255;
				//Store the convolved date
				if (ystart == 1) output->pixels[((y)*width) + x * depth + d] = sum;
				else output->pixels[((y - ystart) * width) + x * depth + d] = sum;
			}
		}
	}

}

void combine(struct combine_data_t *data) {

	struct image_t* sobelx;
	struct image_t* sobely;
	struct image_t* output;
	int ystart;
	int yend;


	int d, y, x;
	int out;
	int index;

	sobelx = data->sobelx;
	sobely = data->sobely;
	output = data->output;
	ystart = data->ystart;
	yend = data->yend;


	int depth = output->depth;
	int width = output->xsize * depth;

	if (ystart == 0) ystart = 1;
	if (yend == sobelx->ysize) yend = sobelx->ysize - 1;

	for (y = ystart; y < yend; y++) {
		for (x = 1; x < sobelx->xsize - 1; x++) {
			for (d = 0; d < depth; d++) {
				if (ystart == 1) index = ((y)*width) + x * depth + d;
				else index = ((y - ystart) * width) + x * depth + d;

				out = (int) sqrt(
					(sobelx->pixels[index] * sobelx->pixels[index]) +
					(sobely->pixels[index] * sobely->pixels[index])
				);
				if (out > 255) out = 255;
				if (out < 0) out = 0;

				//Store the combination
				if (ystart == 1) output->pixels[((y)*width) + x * depth + d] = out;
				else output->pixels[((y - ystart) * width) + x * depth + d] = out;


			}
		}
	}
}

int main()
{
	struct image_t *image, *storagex, *storagey, *storagec, *sobel_x, *sobel_y, *new_image;
	struct convolve_data_t *convx, *convy;
	struct combine_data_t *comb;
	int world_size, my_rank;
	int* offsets, * counts;
	int first_flag = 1;
	Mat frame_in;
	int xsize = 320;
	int ysize = 240;
	int depth = 3;
	int size = xsize * ysize * depth;
	int result;
	uint64_t start, finish;

	image = (image_t*)malloc(sizeof(image_t));
	storagex = (image_t*)malloc(sizeof(image_t));
	storagey = (image_t*)malloc(sizeof(image_t));
	storagec = (image_t*)malloc(sizeof(image_t));
	sobel_x = (image_t*)malloc(sizeof(image_t));
	sobel_y = (image_t*)malloc(sizeof(image_t));
	new_image = (image_t*)malloc(sizeof(image_t));

	convx = (convolve_data_t*)malloc(sizeof(convolve_data_t));
	convy = (convolve_data_t*)malloc(sizeof(convolve_data_t));

	comb = (combine_data_t*)malloc(sizeof(combine_data_t));

	result = MPI_Init(NULL, NULL);
	if (result != MPI_SUCCESS) {
		fprintf(stderr, "Error starting MPI program!.\n");
		MPI_Abort(MPI_COMM_WORLD, result);
	}


	//Get number of tasks and our rank
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

	cout << "Hello World from process " << my_rank << " out of " << world_size << " processes!!!" << endl;
	
	/* Allocate space for storage */
	image->xsize = xsize;
	image->ysize = ysize;
	image->depth = depth;
	image->pixels = (char*)calloc(size, sizeof(char));

	storagex->xsize = xsize;
	storagex->ysize = ysize;
	storagex->depth = depth;
	storagex->pixels = (char *)calloc(size, sizeof(char));

	storagey->xsize = xsize;
	storagey->ysize = ysize;
	storagey->depth = depth;
	storagey->pixels = (char *)calloc(size, sizeof(char));

	storagec->xsize = xsize;
	storagec->ysize = ysize;
	storagec->depth = depth;
	storagec->pixels = (char *)calloc(size, sizeof(char));

	/* Allocate space for sobel_x output */
	sobel_x->xsize = xsize;
	sobel_x->ysize = ysize;
	sobel_x->depth = depth;
	sobel_x->pixels = (char *)calloc(size, sizeof(char));

	/* Allocate space for sobel_y output */
	sobel_y->xsize = xsize;
	sobel_y->ysize = ysize;
	sobel_y->depth = depth;
	sobel_y->pixels = (char *)calloc(size, sizeof(char));

	/* Allocate space for output image */
	new_image->xsize = xsize;
	new_image->ysize = ysize;
	new_image->depth = depth;
	new_image->pixels = (char *)calloc(size, sizeof(char));

	//Allocate memory for the vectorized gather's offsets and counts
	offsets = (int *)calloc(world_size, sizeof(int));
	counts = (int *)calloc(world_size, sizeof(int));

	//Fill in the offsets as the ystarts and the counts as the chunk size
	for (int i = 0; i < world_size; i++) {
		offsets[i] = i * ((ysize / world_size) * xsize * depth);
		counts[i] = ((ysize / world_size) * xsize * depth);
	}
	//Fill in the last bit of the image into the final rank
	counts[world_size - 1] += ((ysize / world_size) * xsize * depth) % world_size;
	

	VideoCapture capture("\\\\JACOB-PC\\e\\Cluster_MPI_Sobel\\x64\\Debug\\cluster_video.mp4");



	if (my_rank == 0 && !capture.isOpened()) {
		cout << "No camera attatched :(" << endl;
	}
	cout << "Start" << endl;
	if (my_rank == 0) start = std::chrono::system_clock::now().time_since_epoch().count();
	for (int i = 0; i < 50; i++) {

		cout << "start loop" << endl;

		if (my_rank == 0) {

			if (!capture.read(frame_in)) {
				cout << "Couldn't get a frame" << endl;
				return -1;
			}
			resize(frame_in, frame_in, Size(xsize, ysize));
			memcpy(image->pixels, frame_in.data, size);
			//imshow("Original", frame_in);

			//waitKey(1);
			
		}
		image->ysize = ysize;
		image->xsize = xsize;
		image->depth = depth;

		cout << "start broadcast" << endl;
		//broadcast the pixels to the other ranks
		MPI_Bcast(image->pixels,
			size,
			MPI_CHAR,
			0,
			MPI_COMM_WORLD);


		/* convolution */
		//setup struct to send to the convolve routine
		
		convx->input = image;
		convx->output = storagex;
		convx->ystart = (ysize / world_size) * my_rank;
		convx->yend = (ysize / world_size) * (my_rank + 1);
		convx->filter = sobel_x_filter;
		generic_convolve(convx);

		//All the data is in chunks in the seperate ranks
		//Gather the data from the storage into sobel_x
		MPI_Gatherv(storagex->pixels,
			counts[my_rank],
			MPI_CHAR,
			sobel_x->pixels,
			counts,
			offsets,
			MPI_CHAR,
			0,
			MPI_COMM_WORLD);
		
		//setup struct to send to the convolve routine
		convy->input = image;
		convy->output = storagey;
		convy->ystart = (ysize / world_size) * my_rank;
		convy->yend = (ysize / world_size) * (my_rank + 1);
		convy->filter = sobel_y_filter;
		generic_convolve(convy);

		//All the data is in chunks in the seperate ranks
		//Gather the data from the storage into sobel_y
		MPI_Gatherv(storagey->pixels,
			counts[my_rank],
			MPI_CHAR,
			sobel_y->pixels,
			counts,
			offsets,
			MPI_CHAR,
			0,
			MPI_COMM_WORLD);

		/* Combine to form output */
		
		comb->sobelx = storagex;
		comb->sobely = storagey;
		comb->output = storagec;
		comb->ystart = (ysize / world_size) * my_rank;
		comb->yend = (ysize / world_size) * (my_rank + 1);
		comb->rank = my_rank;
		combine(comb);

		//All the data is in chunks in the seperate ranks
		//Gather the data from the storagex into new image
		MPI_Gatherv(storagec->pixels,
			counts[my_rank],
			MPI_CHAR,
			new_image->pixels,
			counts,
			offsets,
			MPI_CHAR,
			0,
			MPI_COMM_WORLD);

		cout << "start broadcast" << endl;
		if (my_rank == 0) {
			Mat frame_out(xsize, ysize, CV_8UC3, new_image->pixels);
			resize(frame_out, frame_out, Size(640, 480));
			imshow("Active Edge Detection", frame_out);
			waitKey(1);
		}
		
	}
	if (my_rank == 0) {
		finish = std::chrono::system_clock::now().time_since_epoch().count();

		cout << "Total Time: " << ((float)(finish - start))/10000000.0 << endl;
		cout << "Frame Rate: " << 50/(((float)(finish - start)) / 10000000.0) << endl;
	}
	


	MPI_Finalize();
	return 0;
}
