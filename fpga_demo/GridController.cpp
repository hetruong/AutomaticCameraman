#include "GridController.h"
#include "FramePool.h"
#include <iostream>
#include <cmath>

using std::cout;
using std::endl;


static void * controller_function(void * args);


GridSquare::GridSquare
(
    int x0,
    int y0,
    int w,
    int h,
    int nframes
)
: _x0(x0), _y0(y0), _w(w), _h(h), runningMeanR(0.0), runningMeanSquaredR(0.0),
  runningMeanG(0.0), runningMeanSquaredG(0.0), runningMeanB(0.0), runningMeanSquaredB(0.0),
  meanShortR(0.0), meanShortG(0.0), meanShortB(0.0), occupied(false)
{}

void
GridSquare::update_index
(
    void
)
{
    _idx = (_idx + 1) % _nframes;
}


GridController::GridController
(
    int upper_x,
    int upper_y,
    int x_step,
    int y_step,
    int dim,
    int img_w,
    int img_h,
	FramePool *videoPool
)
: _upper_x(upper_x), _upper_y(upper_y), _x_step(x_step), _y_step(y_step),
  _dim(dim), _img_w(img_w), _img_h(img_h), videoPool(videoPool)
{
    // Create grid of squares
    int x = _upper_x;
    while (x + _dim < _img_w)
    {
        int y = _upper_y;
        while (y + _dim < _img_h)
        {
            _squares.push_back(GridSquare(x, y, _dim, _dim, _nframes));

            y += _y_step;
        }
        x += _x_step;
    }

    cout << "Image dimensions are " << _img_w << "x" << _img_h << endl;
    cout << "Number of squares is " << _squares.size() << endl;
}


void* trackerFunc(void *arg) {

	GridController *gridController = (GridController *)arg;
	Frame *frame = NULL;
	int frameNum = 0;
	while (true) {
		gridController->videoPool->acquire(&frame, frameNum, false, true);

		cv::Mat img(frame->_bgr);
		//cv::Mat img;
		//cv::cvtColor(imgColor, img, CV_RGB2GRAY);

		// compute new average value for each square
		for (int i = 0; i < gridController->_squares.size(); i++) {
			GridSquare &gs = gridController->_squares[i];
			double averageR = 0.0;
			double averageG = 0.0;
			double averageB = 0.0;
			for (int j = gs._x0; j < gs._x0 + gs._w; j++) {
				for (int k = gs._y0; k < gs._y0 + gs._h; k++) {
					//cout << "start j: " << j << " k: " << k << endl;
					cv::Vec3b pixel = img.at<cv::Vec3b>(k, j);
					//cout << "end\n";
					averageB += pixel[0];
					averageG += pixel[1];
					averageR += pixel[2];
				}
			}
			averageR /= (double) (gs._w * gs._h);
			averageG /= (double) (gs._w * gs._h);
			averageB /= (double) (gs._w * gs._h);


			if (frameNum == 0) {
				gs.runningMeanB = averageB;
				gs.runningMeanR = averageR;
				gs.runningMeanG = averageG;
				gs.meanShortB = averageB;
				gs.meanShortR = averageR;
				gs.meanShortG = averageG;
				gs.runningMeanSquaredB = averageB * averageB;
				gs.runningMeanSquaredR = averageR * averageR;
				gs.runningMeanSquaredG = averageG * averageG;
			} else {
				double alpha = .1;
				gs.runningMeanR = (1.0 - alpha) * gs.runningMeanR + alpha * averageR;
				gs.runningMeanSquaredR = (1.0 - alpha) * gs.runningMeanSquaredR + alpha * averageR * averageR;
				gs.runningMeanG = (1.0 - alpha) * gs.runningMeanG + alpha * averageG;
				gs.runningMeanSquaredG = (1.0 - alpha) * gs.runningMeanSquaredG + alpha * averageG * averageG;
				gs.runningMeanB = (1.0 - alpha) * gs.runningMeanB + alpha * averageB;
				gs.runningMeanSquaredB = (1.0 - alpha) * gs.runningMeanSquaredB + alpha * averageB * averageB;

				double smallAlpha = .9;
				gs.meanShortB = (1.0 - smallAlpha) * gs.meanShortB + smallAlpha * averageB;
				gs.meanShortG = (1.0 - smallAlpha) * gs.meanShortG + smallAlpha * averageG;
				gs.meanShortR = (1.0 - smallAlpha) * gs.meanShortR + smallAlpha * averageR;

				double stdR = std::sqrt(gs.runningMeanSquaredR - gs.runningMeanR * gs.runningMeanR);
				double stdG = std::sqrt(gs.runningMeanSquaredG - gs.runningMeanG * gs.runningMeanG);
				double stdB = std::sqrt(gs.runningMeanSquaredB - gs.runningMeanB * gs.runningMeanB);

				double diffR = std::abs(gs.meanShortR - gs.runningMeanR);
				double diffG = std::abs(gs.meanShortG - gs.runningMeanG);
				double diffB = std::abs(gs.meanShortB - gs.runningMeanB);
				//cout << "long: " << gs.runningMeanR << " short: " << gs.meanShortR << " average: " << averageR << " std: " << stdR << endl;
				if (diffR > 20 || diffG > 20 || diffB > 20) {
					gs.occupied = true;
				} else {
					gs.occupied = false;
				}
			}
		}

		frameNum++;

		gridController->videoPool->release(frame);
	}

	return NULL;
}

void
GridController::start
(
    void
)
{
	pthread_create(&_thread, NULL, trackerFunc, this);

}


void
GridController::stop
(
    void
)
{
    _end = true;
}


static void *
controller_function
(
    void * args
)
{

    GridController * gc = reinterpret_cast<GridController *>(args);
    IplImage *gray = cvCreateImage(cvSize(gc->_img_w, gc->_img_h),
                                   IPL_DEPTH_8U, 1);
    IppiSize roi;
    roi.width = gc->_img_w;
    roi.height = gc->_img_h;

    float one_over_nframes = 1.f / gc->_nframes;

    while(!gc->_end)
    {
        //-------------------------------------------------------------
        // Get frame and process it
        Frame * frame = 0;
        int frameNum = 0;
        bool process_frame = false;

        frameNum = gc->_frame_pool->acquire(&frame,
                                            gc->_last_frame_num,
                                            false,
                                            true);

        if (frame && frameNum > 0 && frameNum != gc->_last_frame_num)
        {
            //-------------------------------------------------------------
            // Convert the frame to gray scale
            if (!gray)
            {
                IplImage *gray = cvCreateImage(cvSize(frame->_bgr->width, frame->_bgr->height),
                                               IPL_DEPTH_8U, 1);
            }

            ippiRGBToGray_8u_C3C1R((Ipp8u*)frame->_bgr->imageData,
                                   frame->_bgr->widthStep,
                                   (Ipp8u*)gray->imageData,
                                   gray->width*1,
                                   roi);
            process_frame = true;
        }

        // Release image
        gc->_frame_pool->release(frame);

        if (process_frame)
        {
            SquareIter end = gc->_squares.end();
            for (SquareIter it = gc->_squares.begin();
                 end != it; ++it)
            {
                GridSquare & square= *it;
                int sum       = 0;
                int x0        = square._x0;
                int x1        = square._x0 + square._w;
                int y0        = square._y0;
                int y1        = square._y0 + square._h;
                float size    = square._w * square._h;

                int widthStep = gray->widthStep;

                //---------------------------------------------------------
                // Compute val of this square (val is the average intensity
                // of all pixels in the square).
                unsigned char * data =
                    reinterpret_cast<unsigned char *>(gray->imageData);

                for (int r = y0; r < y1; ++r)
                {
                    for (int c = x0; c < x1; ++c)
                    {
                        sum += data[r*widthStep +c];
                    }
                }
                float val = sum / size;

                //---------------------------------------------------------
                // Compute the mean of the vals
                float mean = one_over_nframes *
                             std::accumulate(square._vals.begin(),
                                             square._vals.end(),
                                             0.f);

                //---------------------------------------------------------
                // Compute Variance and standard deviation
                float variance = 0.f;
                for (unsigned int i = 0; i < square._vals.size(); ++i)
                {
                    float tmp = square._vals[i] - mean;
                    variance += tmp*tmp;
                }
                variance /= square._vals.size();
                float std_dev = sqrtf(variance);

                //---------------------------------------------------------
                // Compute the difference
                float diff = fabs(val - mean);

                // Flag this square if the difference from mean is greater
                // that 3 standard deviations
                square._marked = false;
                if (diff > 6*std_dev)
                {
                    square._marked = true;
                }

                //---------------------------------------------------------
                // Save this value
                square._vals[square._idx] = val;
                square.update_index();

                gc->_last_frame_num = frameNum;
            }
        } // if (process_frame)

        usleep(30*1000);

    } // while(!gc->_end)
}


