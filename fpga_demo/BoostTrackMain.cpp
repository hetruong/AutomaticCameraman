#include "Public.h"
#include "FramePool.h"
#include "VideoImages.h"
#include "FrameAnnotator.h"
#include "FrameDisplayer.h"
#include "GridAnnotator.h"
#include "GridController.h"
#include "DebugLogger.h"
#include "FrameSaver.h"
#include <riffa.h>
#include <execinfo.h>
#include <signal.h>
#include <sys/stat.h>


// Forward declarations for functions
void handle_stop(void);
void runtime_logic(FrameAnnotator * annotator,
                   VideoImages *    video,
                   FramePool *      videoPool,
                   GridController * grid_controller);


// Global variables
int end = 0;
static char * DISPLAYER_TITLE = "Capture Video";
char prefix[] = "../../milboosttracker-master/sylv/imgs/";
char stem[] = "%simg%05d.png";
int startFrame = 1;
int endFrame = 1344;
int startCenterX = 146;
int startCenterY = 84;
int startDim = 50; // with TRAIN_THRESH 7
int frameW = 320;
int frameH = 240;
int save_images = 0;
char * images_folder = "images";


//*****************************************************************************
// * main
//*****************************************************************************
int
main
(
   int argc,
   const char** argv
)
{
	randinitalize(0);

    if (argc > 1 && !strcmp("-s", argv[1])) {
        save_images = 1;
    }

    //----------------------------------------------------------------
    // Create a debug logger
    DebugLogger * debug_logger = new DebugLogger();
    //debug_logger->start();

	//-----------------------------------------------------------------
	// Open the FPGA for capture
	int fpgaIdCapture = 0; // for now assume FPGA0 is capture FPGA
	fpga_t *fpgaCapture;
    fpgaCapture = fpga_open(fpgaIdCapture);

    if (fpgaCapture == NULL) {
        printf("ERROR: Could not open FPGA %d.\n", fpgaIdCapture);
        return -1;
    }

    //-----------------------------------------------------------------
    // Create the FramePools

    // Make video pool size large enough to store the last 30 seconds
    // worth of video (for debugging purposes)
    int pool_size = 10;

	FramePool *videoPool = new FramePool(pool_size);
	FramePool *annotationPool = new FramePool(pool_size);


    //-----------------------------------------------------------------
	// Create Frame saver for saving video or for dumping the last n frames
	// upon program termination
	FrameSaver * frame_saver = new FrameSaver(0,
	                                          true);
    frame_saver->SetBGPool(videoPool);
    frame_saver->SetFGPool(annotationPool);
    frame_saver->start();

	//-----------------------------------------------------------------
	// Create the VideoImages source
	//VideoImages *video = new DiskImages(prefix, stem, startFrame, endFrame, frameW, frameH, 30, videoPool);
	VideoImages *video = new LiveImages(fpgaCapture,
			                            0,
			                            7000,
			                            1280,
			                            720,
			                            60,
			                            videoPool);
    if (!video->init()) {
        printf("ERROR: Could not initialize the VideoImages.\n");
        return -1;
    }

    //-----------------------------------------------------------------
	// Create the FrameAnnotator
	FrameAnnotator *annotator = new FrameAnnotator(video->_width,
	                                               video->_height,
		                                           NULL,
		                                           annotationPool);
    if (!annotator->init()) {
        printf("ERROR: Could not initialize the FrameAnnotator.\n");
        return -1;
    }

    //-----------------------------------------------------------------
	// Create the FrameDisplayer
	FrameDisplayer *displayer = new FrameDisplayer(DISPLAYER_TITLE,
	                                               video->_width,
	                                               video->_height,
		                                           videoPool,
		                                           annotationPool,
		                                           handle_stop);
    if (!displayer->init()) {
        printf("ERROR: Could not initialize the FrameDisplayer.\n");
        return -1;
    }

    //-----------------------------------------------------------------
    // Create controller object
    GridController * grid_controller = new GridController(36,
                                                          36,
                                                          80,
                                                          80,
                                                          12,
                                                          video->_width,
                                                          video->_height,
                                                          videoPool);

    //debug_logger->log("Creating new grid controller");

    //-----------------------------------------------------------------
	// Start the various threads
	video->start();
	displayer->start();

	//-----------------------------------------------------------------
	// Enter runtime logic
	runtime_logic(annotator,
	              video,
	              videoPool,
	              grid_controller);

	//-----------------------------------------------------------------
	// Cleanup Code
	frame_saver->stop();
	videoPool->destroy();
	annotationPool->destroy();

	// Stop the various threads (annotator last)
	displayer->stop();
	video->stop();
	annotator->stop();

	// Close the FPGA
	fpga_close(fpgaCapture);

	// Stop logs
    //debug_logger->stop();

	return 0;
}


//*****************************************************************************
// * runtime_logic
//*****************************************************************************
void
runtime_logic
(
    FrameAnnotator * annotator,
    VideoImages *    video,
    FramePool *      videoPool,
    GridController * grid_controller
)
{

    // For the time being, just get an image from the videoPool and initialize
    // on a specific location there. In time, we should replace this logic with
    // a proper target registration routine.
    usleep(3*1000*1000);

    //-----------------------------------------------------------------
    // Get a Frame
    Frame *frame;
    if (videoPool->acquire(&frame, 0, true, true) < 0) {
        printf("ERROR: Could not acquire the initial Frame.\n");
        return;
    }

    //-----------------------------------------------------------------
    // Convert the frame to gray scale
    IplImage *gray = cvCreateImage(cvSize(video->_width, video->_height), IPL_DEPTH_8U, 1);
    IppiSize roi;
    roi.width = video->_width;
    roi.height = video->_height;
    ippiRGBToGray_8u_C3C1R((Ipp8u*)frame->_bgr->imageData,
                           frame->_bgr->width*3,
                           (Ipp8u*)gray->imageData,
                           gray->width*1,
                           roi);

    //-----------------------------------------------------------------
    // Create controller
    GridAnnotator * grid_annotator = new GridAnnotator(0, grid_controller);
    annotator->add(grid_annotator);

    //-----------------------------------------------------------------
    // Release the frame
    videoPool->release(frame);
    grid_controller->start();
    annotator->start();     // DELAY STARTING THESE THINGS UNTIL NOW TO
    grid_controller->start();

    // Just wait for the end
    while (end == 0)
        usleep(30*1000);
}


//*****************************************************************************
// * handle_stop
//*****************************************************************************
void
handle_stop()
{
	end = 1;
}


