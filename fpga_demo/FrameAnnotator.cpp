#include "FrameAnnotator.h"

TS_INIT(annotator, 2);
TC_INIT(annotator, 1);


//////////////////////////////////////////////////////////////////////////////////////////////////////////

void Annotator::draw_rect(IplImage *img, int x, int y, int w, int h, 
	int lineWidth, int R, int G, int B, int alpha) {
	CvPoint p1, p2;
	p1 = cvPoint(x, y); 
	p2 = cvPoint(x+w, y+h);
	// Setting alpha to 255 means fully visible (no transparency)
	cvRectangle(img, p1, p2, cvScalar(B, G, R, alpha), lineWidth);
}


void Annotator::draw_line(IplImage *img, int x0, int y0, int x1, int y1, 
	int lineWidth, int R, int G, int B, int alpha) {
	CvPoint p1, p2;
	p1 = cvPoint(x0, y0); 
	p2 = cvPoint(x1, y1);
	// Setting alpha to 255 means fully visible (no transparency)
	cvLine(img, p1, p2, cvScalar(B, G, R, alpha), lineWidth);
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////

void *annotator_fxn(void *arg) {
	FrameAnnotator *fa = (FrameAnnotator *)arg;
	Frame *frame = NULL;
	int x, y;
	bool clear = true;

	// Create the 4 channel temp image
	IplImage *longTerm = cvCreateImage(cvSize(fa->_width, fa->_height), IPL_DEPTH_8U, 4);
	cvSet(longTerm, cvScalar(0, 0, 0, 0));

	for (int c = 0; fa->_end == 0; c++) {
TS_STAMP(annotator, 0);
		// Update the pool
		frame = fa->_pool->update_next(frame, c, false);

		if (NULL != fa->_controller)
		{
		    // Wait for changes
		    fa->_lastTracked = fa->_controller->tracked(fa->_lastTracked, &x, &y, true);
		    if (fa->_lastTracked < 0)
			    break;
		}

		// Clear the long term image (if necessary)
		if (clear)
			cvSet(longTerm, cvScalar(0, 0, 0, 0));
		clear = false;

		// Get our spinlock
		while (__sync_lock_test_and_set(&fa->_spinlock, 1) != 0);

		// Let the Annotators draw
		for (map<int, Annotator *>::iterator it=fa->_annotators.begin(); it != fa->_annotators.end(); ++it)
			clear = clear || ((*it).second)->draw_long_term(longTerm);
		cvCopy(longTerm, frame->_bgr);
		for (map<int, Annotator *>::iterator it=fa->_annotators.begin(); it != fa->_annotators.end(); ++it)
			((*it).second)->draw_short_term(frame->_bgr);
		
		// Release our spinlock
		__sync_lock_release(&fa->_spinlock);

TS_STAMP(annotator, 1);
TC_ACCRUE(annotator, 0, 1, 0);
	}

	// Cleanup
	cvReleaseImage(&longTerm);

	return NULL;
}


FrameAnnotator::FrameAnnotator(int w, int h, TrackerController *controller, 
	FramePool *pool) {
	// Save
	_width = w;
	_height = h;
	_controller = controller;
	_pool = pool;
}


bool FrameAnnotator::init() {
	// Initialize locals
	_end = 0;
	_lastTracked = 0;
	_spinlock = 0;

	// Initialize the FramePool
	IplImage *img;
	for (int i = 0; i < _pool->_size; i++) {
		img = cvCreateImage(cvSize(_width, _height), IPL_DEPTH_8U, 4);
		cvSet(img, cvScalar(0, 0, 0, 255));
		_pool->add_frame(img);
	}

	return true;
}


void FrameAnnotator::start() {
	// Create and start the thread
	pthread_create(&_thread, NULL, annotator_fxn, this);
}


void FrameAnnotator::stop() {
printf("annotator: %f (%ld)\n", TC_ITER_AVG(annotator, 0), TC_ITERS(annotator, 0));
	// Signal end
	_end = 1;

	// Wait until the thread exits
    pthread_join(_thread, NULL);
}


void FrameAnnotator::add(Annotator *a) {
	// Get our spinlock
	while (__sync_lock_test_and_set(&_spinlock, 1) != 0);

	// Update the annotators
	_annotators[a->_id] = a;

	// Release our spinlock
	__sync_lock_release(&_spinlock);
}


void FrameAnnotator::remove(Annotator *a) {
	// Get our spinlock
	while (__sync_lock_test_and_set(&_spinlock, 1) != 0);

	// Update the annotators
	_annotators.erase(a->_id);

	// Release our spinlock
	__sync_lock_release(&_spinlock);
}



