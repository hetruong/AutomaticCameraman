#ifndef FRAME_ANNOTATOR_PUBLIC
#define FRAME_ANNOTATOR_PUBLIC

#include "Public.h"
#include "FramePool.h"
#include "Annotator.h"
#include "TrackerController.h"
#include <pthread.h>



//////////////////////////////////////////////////////////////////////////////////////////////////////////
class FrameAnnotator
{
public:
	FrameAnnotator(int w, int h, TrackerController *controller, FramePool *pool);
	~FrameAnnotator(){};
	bool init();
	void start();
	void stop();
	void add(Annotator *a);
	void remove(Annotator *a);

	pthread_t _thread; 									// Thread for creating annotations
	TrackerController *_controller;						// TrackerController to indicate when tracking takes place
	FramePool *_pool;									// FramePool to add annotated images to
	map<int, Annotator *> _annotators;					// Map of Annotators
	int _spinlock;										// Spinlock to proctect the Annotators map
	int _width;											// Width of images in the FramePool
	int _height;										// Height of images in the FramePool
	int _lastTracked;									// Last frame tracked
	int _end;											// End signal
};


#endif



