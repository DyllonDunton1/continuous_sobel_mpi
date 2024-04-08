// A simple MPI code printing a message by each MPI rank

#include <iostream>
#include <mpi.h>
#include <opencv2/opencv.hpp>

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
	unsigned char* pixels;
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
static void* generic_convolve(void* argument) {

	int x, y, k, l, d;
	uint32_t color;
	int sum, depth, width;

	struct image_t* input;
	struct image_t* output;
	int* filter;
	struct convolve_data_t* data;
	int ystart, yend;

	/* Convert from void pointer to the actual data type */
	data = (struct convolve_data_t*)argument;
	input = data->input;
	output = data->output;
	filter = data->filter;

	ystart = data->ystart;
	yend = data->yend;

	depth = input->depth;
	width = input->xsize * depth;
	//printf("start looping, start:%d, stop%d\n", ystart, yend);
	//printf("ysize: %d\n", input->ysize);
	/* handle border */
	if (ystart == 0) ystart = 1;
	if (yend == input->ysize) yend = input->ysize - 1;

	printf("start looping, start:%d, stop%d\n", ystart, yend);

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

	return NULL;
}

static void combine(void* argument) {

	struct combine_data_t* data;
	struct image_t* sobelx;
	struct image_t* sobely;
	struct image_t* output;
	int ystart;
	int yend;


	int d, y, x;
	int out;
	int index;

	data = (struct combine_data_t*)argument;
	sobelx = data->sobelx;
	sobely = data->sobely;
	output = data->output;
	ystart = data->ystart;
	yend = data->yend;


	int depth = output->depth;
	int width = output->xsize * depth;

	if (ystart == 0) ystart = 1;
	if (yend == sobelx->ysize) yend = sobelx->ysize - 1;
	printf("start combining, start:%d, stop%d\n", ystart, yend);

	for (y = ystart; y < yend; y++) {
		for (x = 1; x < sobelx->xsize - 1; x++) {
			for (d = 0; d < depth; d++) {
				if (ystart == 1) index = ((y)*width) + x * depth + d;
				else index = ((y - ystart) * width) + x * depth + d;

				out = sqrt(
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
	struct image_t image, storagex, storagey, storagec, sobel_x, sobel_y, new_image;
	int result;
	int num_ranks, rank;
	int sizes[3];
	unsigned char* pixels;
	int size;
	int* offsets, * counts;
	int first_flag = 1;

	int my_rank;
	int world_size;

	cv::VideoCapture capture(0);

	if (!capture.isOpened()) {
		cout << "No camera attatched :(" << endl;
	}

	namedWindow("Active Edge Detection");
	resizeWindow("Image", 600, 600);

	Mat frame;
	unsigned char* image;

	MPI_Init(NULL, NULL);

	MPI_Comm_size(MPI_COMM_WORLD, &world_size);
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

	cout << "Hello World from process " << my_rank << " out of " << world_size << " processes!!!" << endl;

	//Take first frame and get necessary data
	if (my_rank == 0) {
		if (!capture.read(frame)) {
			cout << "Couldn't get a frame" << endl;
			return -1;
		}
		else {
			image.pixels = frame.data;
			image.ysize = frame.rows;
			image.xsize = frame.cols;
			image.depth = frame.channels();
			sizes[0] = image.ysize;
			sizes[1] = image.xsize;
			sizes[2] = image.depth;
		}
	}
	MPI_Bcast(sizes,
		3,
		MPI_INT,
		0,
		MPI_COMM_WORLD);
	if (my_rank == 0) {
		size = image.ysize * image.xsize * image.depth;
		cout << "Size: " << size << endl;
	}
	else {
		//allocate memory for the incoming pixels and store the dimensions from the previous broadcast
		pixels = (unsigned char*)calloc(sizes[0] * sizes[1] * sizes[2], sizeof(unsigned char));
		image.ysize = sizes[0];
		image.xsize = sizes[1];
		image.depth = sizes[2];
		size = image.ysize * image.xsize * image.depth;
		image.pixels = pixels;
	}

	/* Allocate space for storage */
	storagex.xsize = image.xsize;
	storagex.ysize = image.ysize;
	storagex.depth = image.depth;
	storagex.pixels = calloc(image.xsize * image.ysize * image.depth, sizeof(char));

	storagey.xsize = image.xsize;
	storagey.ysize = image.ysize;
	storagey.depth = image.depth;
	storagey.pixels = calloc(image.xsize * image.ysize * image.depth, sizeof(char));

	storagec.xsize = image.xsize;
	storagec.ysize = image.ysize;
	storagec.depth = image.depth;
	storagec.pixels = calloc(image.xsize * image.ysize * image.depth, sizeof(char));

	/* Allocate space for sobel_x output */
	sobel_x.xsize = image.xsize;
	sobel_x.ysize = image.ysize;
	sobel_x.depth = image.depth;
	sobel_x.pixels = calloc(image.xsize * image.ysize * image.depth, sizeof(char));

	/* Allocate space for sobel_y output */
	sobel_y.xsize = image.xsize;
	sobel_y.ysize = image.ysize;
	sobel_y.depth = image.depth;
	sobel_y.pixels = calloc(image.xsize * image.ysize * image.depth, sizeof(char));

	/* Allocate space for output image */
	new_image.xsize = image.xsize;
	new_image.ysize = image.ysize;
	new_image.depth = image.depth;
	new_image.pixels = calloc(image.xsize * image.ysize * image.depth, sizeof(char));

	//Allocate memory for the vectorized gather's offsets and counts
	offsets = calloc(num_ranks, sizeof(int));
	counts = calloc(num_ranks, sizeof(int));

	//Fill in the offsets as the ystarts and the counts as the chunk size
	for (int i = 0; i < num_ranks; i++) {
		offsets[i] = i * ((image.ysize / num_ranks) * image.xsize * image.depth);
		counts[i] = ((image.ysize / num_ranks) * image.xsize * image.depth);
	}
	//Fill in the last bit of the image into the final rank
	counts[num_ranks - 1] += ((image.ysize / num_ranks) * image.xsize * image.depth) % num_ranks;

	while (1) {

		if (my_rank == 0){
			if (!capture.read(frame)) {
				cout << "Couldn't get a frame" << endl;
				return -1;
			}
			pixels = frame.data;
			image.ysize = frame.rows;
			image.xsize = frame.cols;
			image.depth = frame.channels();
		}
		
		
		//broadcast the pixels to the other ranks
		MPI_Bcast(image.pixels,
			size,
			MPI_CHAR,
			0,
			MPI_COMM_WORLD);

		/* convolution */
		//setup struct to send to the convolve routine
		struct convolve_data_t convx;
		convx.input = &image;
		convx.output = &storagex;
		convx.ystart = (image.ysize / num_ranks) * rank;
		convx.yend = (image.ysize / num_ranks) * (rank + 1);
		convx.filter = sobel_x_filter;
		generic_convolve((void*)&convx);

		//All the data is in chunks in the seperate ranks
		//Gather the data from the storage into sobel_x
		MPI_Gatherv(storagex.pixels,
			counts[rank],
			MPI_CHAR,
			sobel_x.pixels,
			counts,
			offsets,
			MPI_CHAR,
			0,
			MPI_COMM_WORLD);

		//setup struct to send to the convolve routine
		struct convolve_data_t convy;
		convy.input = &image;
		convy.output = &storagey;
		convy.ystart = (image.ysize / num_ranks) * rank;
		convy.yend = (image.ysize / num_ranks) * (rank + 1);
		convy.filter = sobel_y_filter;
		generic_convolve((void*)&convy);


		//All the data is in chunks in the seperate ranks
		//Gather the data from the storage into sobel_y
		MPI_Gatherv(storagey.pixels,
			counts[rank],
			MPI_CHAR,
			sobel_y.pixels,
			counts,
			offsets,
			MPI_CHAR,
			0,
			MPI_COMM_WORLD);

		/* Combine to form output */
		struct combine_data_t comb;
		comb.sobelx = &storagex;
		comb.sobely = &storagey;
		comb.output = &storagec;
		comb.ystart = (image.ysize / num_ranks) * rank;
		comb.yend = (image.ysize / num_ranks) * (rank + 1);
		comb.rank = rank;
		combine((void*)&comb);

		//All the data is in chunks in the seperate ranks
		//Gather the data from the storagex into new image
		MPI_Gatherv(storagec.pixels,
			counts[rank],
			MPI_CHAR,
			new_image.pixels,
			counts,
			offsets,
			MPI_CHAR,
			0,
			MPI_COMM_WORLD);

		frame.data = new_image.pixels;

		imshow("Active Edge Detection", frame);

		waitkey(0);
	}



	MPI_Finalize();
	return 0;
}
