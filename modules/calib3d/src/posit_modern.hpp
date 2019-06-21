// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#ifndef OPENCV_CALIB3D_POSIT_MODERN_HPP
#define OPENCV_CALIB3D_POSIT_MODERN_HPP

#include <opencv2/core.hpp>

namespace cv {
namespace POSIT {

void posit(const Mat& opoints, const Mat& ipoints, Matx31d& rvec, Matx31d& tvec);

void positPlanar();

}
} //namespace cv
#endif
