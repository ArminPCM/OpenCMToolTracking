// Edited by: Mostafa Parchami
#include "CMT.h"
#include "gui.h"

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <fstream>

cv::Mat deinterlace(cv::Mat& gray)
{
	cv::Mat im_deinterlace(gray.rows / 2, gray.cols, CV_8UC1);

	#pragma omp parallel for
	for(int i = 0; i<gray.rows; i+=2)
	{
		memcpy(im_deinterlace.data + i*gray.step/2, gray.data + i*gray.step, gray.step);
	}

	return im_deinterlace;
}

int main(int argc, char **argv)
{
    //Create a CMT object
    cmt::CMT cmt;

    //Initialization bounding box
    cv::Rect rect;

    std::string input_path("G:\\Prostate_1_23_13_Sequence_1\\Prost012413_S2_N_LEFT_%04d.png");
	std::string windowname("CMT");

    //cmt.str_detector = optarg; //Default is FAST
    //cmt.str_descriptor = optarg; //Default is BRISK
    cmt.consensus.estimate_scale = true;
    cmt.consensus.estimate_rotation = true;

    //Create window
    cv::namedWindow(windowname);

    cv::VideoCapture cap;

    bool show_preview = true;

    //If no input was specified
    if (input_path.length() == 0)
    {
        cap.open(0); //Open default camera device
    }
    //Else open the video specified by input_path
    else
    {
        cap.open(input_path);
        show_preview = false;
    }

    //If it doesn't work, stop
    if(!cap.isOpened())
    {
        std::cerr << "Unable to open video capture." << std::endl;
        return -1;
    }

    //Get initial image
    Mat im, im_gray, im_deinterlace;
    cap >> im;


    cv::cvtColor(im, im_gray, CV_BGR2GRAY);

	im_deinterlace = deinterlace(im_gray);
    rect = getRect(im_deinterlace, windowname);

    //Initialize CMT
    cmt.initialize(im_deinterlace, rect);
	
    int frame = 0;

    //Main loop
    while (true)
    {
        ++frame;

        //If loop flag is set, reuse initial image (for debugging purposes)
		//for(int i = 0; i<5; ++i)
			cap >> im; //Else use next image in stream

        cv::cvtColor(im, im_gray, CV_BGR2GRAY);

		im_deinterlace = deinterlace(im_gray);

        //Let CMT process the frame
        cmt.processFrame(im_deinterlace);

		//Visualize the output
		//It is ok to draw on im itself, as CMT only uses the grayscale image
		for(size_t i = 0; i < cmt.points_active.size(); i++)
		{
			circle(im_deinterlace, cmt.points_active[i], 2, cv::Scalar(255));
		}

		Point2f vertices[4];
		cmt.bb_rot.points(vertices);
		for (int i = 0; i < 4; i++)
		{
			line(im_deinterlace, vertices[i], vertices[(i+1)%4], cv::Scalar(255));
		}

		cv::imshow(windowname, im_deinterlace);

        char key = cv::waitKey(5);

        if(key == 'q') break;

        std::cout << "#" << frame << " active: " << cmt.points_active.size();
    }

    return 0;
}
