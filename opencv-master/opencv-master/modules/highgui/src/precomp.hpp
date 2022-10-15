/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                        Intel License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of Intel Corporation may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#ifndef __HIGHGUI_H_
#define __HIGHGUI_H_

#if defined(__OPENCV_BUILD) && defined(BUILD_PLUGIN)
#undef __OPENCV_BUILD  // allow public API only
#endif

#include "opencv2/highgui.hpp"
#if !defined(BUILD_PLUGIN)
#include "opencv_highgui_config.hpp"  // generated by CMake
#endif

#include "opencv2/core/utility.hpp"
#if defined(__OPENCV_BUILD)
#include "opencv2/core/private.hpp"
#endif

#include "opencv2/imgproc.hpp"
#include "opencv2/imgproc/imgproc_c.h"
#include "opencv2/highgui/highgui_c.h"

#include "opencv2/imgcodecs.hpp"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <limits.h>
#include <ctype.h>
#include <assert.h>

#if defined _WIN32 || defined WINCE
    #include <windows.h>
    #undef small
    #undef min
    #undef max
    #undef abs
#endif

/* Errors */
#define HG_OK          0 /* Don't bet on it! */
#define HG_BADNAME    -1 /* Bad window or file name */
#define HG_INITFAILED -2 /* Can't initialize HigHGUI */
#define HG_WCFAILED   -3 /* Can't create a window */
#define HG_NULLPTR    -4 /* The null pointer where it should not appear */
#define HG_BADPARAM   -5

#define __BEGIN__ __CV_BEGIN__
#define __END__  __CV_END__
#define EXIT __CV_EXIT__

#define CV_WINDOW_MAGIC_VAL     0x00420042
#define CV_TRACKBAR_MAGIC_VAL   0x00420043

//Yannick Verdie 2010, Max Kostin 2015
void cvSetModeWindow_W32(const char* name, double prop_value);
void cvSetModeWindow_GTK(const char* name, double prop_value);
void cvSetModeWindow_COCOA(const char* name, double prop_value);
void cvSetModeWindow_WinRT(const char* name, double prop_value);

CvRect cvGetWindowRect_W32(const char* name);
CvRect cvGetWindowRect_GTK(const char* name);
CvRect cvGetWindowRect_COCOA(const char* name);

double cvGetModeWindow_W32(const char* name);
double cvGetModeWindow_GTK(const char* name);
double cvGetModeWindow_COCOA(const char* name);
double cvGetModeWindow_WinRT(const char* name);

double cvGetPropWindowAutoSize_W32(const char* name);
double cvGetPropWindowAutoSize_GTK(const char* name);

double cvGetRatioWindow_W32(const char* name);
double cvGetRatioWindow_GTK(const char* name);

double cvGetOpenGlProp_W32(const char* name);
double cvGetOpenGlProp_GTK(const char* name);

double cvGetPropVisible_W32(const char* name);

double cvGetPropTopmost_W32(const char* name);
double cvGetPropTopmost_COCOA(const char* name);

void cvSetPropTopmost_W32(const char* name, const bool topmost);
void cvSetPropTopmost_COCOA(const char* name, const bool topmost);

double cvGetPropVsync_W32(const char* name);
void cvSetPropVsync_W32(const char* name, const bool enabled);

//for QT
#if defined (HAVE_QT)
CvRect cvGetWindowRect_QT(const char* name);
double cvGetModeWindow_QT(const char* name);
void cvSetModeWindow_QT(const char* name, double prop_value);

double cvGetPropWindow_QT(const char* name);
void cvSetPropWindow_QT(const char* name,double prop_value);

double cvGetRatioWindow_QT(const char* name);
void cvSetRatioWindow_QT(const char* name,double prop_value);

double cvGetOpenGlProp_QT(const char* name);
double cvGetPropVisible_QT(const char* name);
#endif

inline void convertToShow(const cv::Mat &src, cv::Mat &dst, bool toRGB = true)
{
    const int src_depth = src.depth();
    CV_Assert(src_depth != CV_16F && src_depth != CV_32S);
    cv::Mat tmp;
    switch(src_depth)
    {
    case CV_8U:
        tmp = src;
        break;
    case CV_8S:
        cv::convertScaleAbs(src, tmp, 1, 127);
        break;
    case CV_16S:
        cv::convertScaleAbs(src, tmp, 1/255., 127);
        break;
    case CV_16U:
        cv::convertScaleAbs(src, tmp, 1/255.);
        break;
    case CV_32F:
    case CV_64F: // assuming image has values in range [0, 1)
        src.convertTo(tmp, CV_8U, 255., 0.);
        break;
    }
    cv::cvtColor(tmp, dst, toRGB ? cv::COLOR_BGR2RGB : cv::COLOR_BGRA2BGR, dst.channels());
}

inline void convertToShow(const cv::Mat &src, const CvMat* arr, bool toRGB = true)
{
    cv::Mat dst = cv::cvarrToMat(arr);
    convertToShow(src, dst, toRGB);
    CV_Assert(dst.data == arr->data.ptr);
}


namespace cv {

CV_EXPORTS Mutex& getWindowMutex();
static inline Mutex& getInitializationMutex() { return getWindowMutex(); }

}  // namespace

#endif /* __HIGHGUI_H_ */