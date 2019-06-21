/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
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
//   * The name of the copyright holders may not be used to endorse or promote products
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

#include "test_precomp.hpp"

namespace opencv_test { namespace {

void computePose(const Matx41d& I, const Matx41d& J, Matx33d& R, Matx31d& t)
{
    double scale = sqrt(I(0)*I(0) + I(1)*I(1) + I(2)*I(2));

//    if ((normI3 < 1e-10) || (normJ3 < 1e-10)) {
//        // vpERROR_TRACE(" normI+normJ = 0, division par zero " ) ;
//        throw(vpException(vpException::divideByZeroError,
//                          "Division by zero in Dementhon pose computation: normI or normJ = 0"));
//    }

    double  Z0 = 1.0 / scale;
    Vec3d I_normalized(I(0)*Z0, I(1)*Z0, I(2)*Z0);
    Vec3d J_normalized(J(0)*Z0, J(1)*Z0, J(2)*Z0);
    Matx31d K = I_normalized.cross(J_normalized);
    double K_norm = sqrt( K(0)*K(0) + K(1)*K(1) + K(2)*K(2) );
    Vec3d K_normalized(K(0)/K_norm, K(1)/K_norm, K(2)/K_norm);

    std::cout << "I: " << I.t() << std::endl;
    std::cout << "J: " << J.t() << std::endl;
    std::cout << "I_normalized: " << I_normalized.t() << std::endl;
    std::cout << "J_normalized: " << J_normalized.t() << std::endl;
    std::cout << "K: " << K.t() << std::endl;
    std::cout << "K_normalized: " << K_normalized.t() << std::endl;

    for (int i = 0; i < 3; i++)
    {
        R(0,i) = I_normalized(i);
        R(1,i) = J_normalized(i);
        R(2,i) = K_normalized(i);
    }

    std::cout << "\nR:\n" << R << std::endl;

    t(0,0) = I(3) * Z0;
    t(1,0) = J(3) * Z0;
    t(2,0) = Z0;
}

/*fonction POS (approximation SOP) pour points COPLanaires. Retourne une translation et deux rotations.*/
/*Le premier element des matrices de rotation est mis a 2 en cas de pose impossible*/
/*(points objets derriere le plan image)*/
void PosCopl(const std::vector<Point3d>& objectPointsCentered, const std::vector<Point2d>& imagePoints, const Mat& objectMatrix,
             const Matx41d& U, Matx33d& rotation1, Matx31d& translation1, Matx33d& rotation2, Matx31d& translation2)
{
    Mat xprime(static_cast<int>(objectPointsCentered.size()), 1, CV_64FC1);
    Mat yprime(static_cast<int>(objectPointsCentered.size()), 1, CV_64FC1);

    for (int i = 0; i < xprime.rows; i++) {
        xprime.at<double>(i,0) = imagePoints[i].x;
        yprime.at<double>(i,0) = imagePoints[i].y;
    }

    //TODO: debug
    std::cout << "objectPointsCentered: " << objectPointsCentered.size() << std::endl;
    std::cout << "objectMatrix:\n" << objectMatrix << std::endl;
    std::cout << "xprime:\n" << xprime << std::endl;
    std::cout << "yprime:\n" << yprime << std::endl;

    Matx41d I4_0 = Matx41d(Mat(objectMatrix * xprime));
    Matx41d J4_0 = Matx41d(Mat(objectMatrix * yprime));

    //TODO: debug
    std::cout << "I4_0: " << I4_0.t() << std::endl;
    std::cout << "J4_0: " << J4_0.t() << std::endl;

    Matx31d I0(I4_0(0), I4_0(1), I4_0(2));
    Matx31d J0(J4_0(0), J4_0(1), J4_0(2));

    double J0sq_I0sq = J0.dot(J0) - I0.dot(I0);
    double I0J0_2 = 2 * I0.dot(J0);

    //TODO: debug
    std::cout << "J0sq_I0sq: " << J0sq_I0sq << std::endl;
    std::cout << "I0J0_2: " << I0J0_2 << std::endl;

    double rho = 0.0, theta = 0.0;
    if (std::fabs(J0sq_I0sq) > std::numeric_limits<double>::epsilon())
    {
        rho = sqrt( sqrt(J0sq_I0sq*J0sq_I0sq + I0J0_2*I0J0_2) );
        theta = atan2(-I0J0_2, J0sq_I0sq) / 2;
    }
    else
    {
        rho = sqrt(std::fabs(I0J0_2));
        theta = I0J0_2 >= 0 ? -M_PI_2 : M_PI_2;
    }

    double costheta = cos(theta);
    double sintheta = sin(theta);

    //TODO:
    std::cout << "rho: " << rho << " ; theta: " << theta << std::endl;

    double lambda = rho * costheta;
    double mu = rho * sintheta;

    //First solution
    Matx41d I = I4_0 + lambda*U;
    Matx41d J = J4_0 + mu*U;

    //TODO: debug
    std::cout << "lambda: " << lambda << std::endl;
    std::cout << "mu: " << mu << std::endl;
    std::cout << "I4_0: " << I4_0.t() << std::endl;
    std::cout << "J4_0: " << J4_0.t() << std::endl;
    std::cout << "I: " << I.t() << std::endl;
    std::cout << "J: " << J.t() << std::endl;

    computePose(I, J, rotation1, translation1);

    //Second solution
    I = I4_0 - lambda*U;
    J = J4_0 - mu*U;

    computePose(I, J, rotation2, translation2);
}

double homogeneousTransformZ(const Point3d& pt, const Matx33d& R, const Matx31d& t)
{
    return R(2,0)*pt.x + R(2,1)*pt.y + R(2,2)*pt.z + t(2,0);
}

double computeResidual(const vector<Point3d>& objectPoints, const vector<Point2d>& imagePoints, const Matx33d& R, const Matx31d& t)
{
    vector<Point2d> projectedPoints;
    Matx31d rvec;
    Rodrigues(R, rvec);
    projectPoints(objectPoints, rvec, t, Matx33d::eye(), noArray(), projectedPoints);

    return cv::norm(projectedPoints, imagePoints, NORM_L2SQR) / (2*projectedPoints.size());
}

void checkNotBehindCamera(const vector<Point3d>& objectPointsCentered, const Matx33d& Rot1, const Matx31d& Trans1,
                          const Matx33d& Rot2, const Matx31d& Trans2, int& errorCode1, int& errorCode2)
{
    for (size_t i = 0; i < objectPointsCentered.size(); i++)
    {
        const Point3d& pt = objectPointsCentered[i];
        double Z1 = homogeneousTransformZ(pt, Rot1, Trans1);
        if (Z1 < 0)
        {
            errorCode1 = -1;
        }
        double Z2 = homogeneousTransformZ(pt, Rot2, Trans2);
        if (Z2 < 0)
        {
            errorCode2 = -1;
        }
    }
}

void Transf(const vector<Point3d>& objectPointsCentered, const vector<Point2d>& imagePoints, const Mat& objectMatrix,
            const Matx41d& U, Matx33d& R, Matx31d& t, double& residual)
{
    double residual_prev = 1e6;
    residual = computeResidual(objectPointsCentered, imagePoints, R, t);

    const int nbPts = static_cast<int>(objectPointsCentered.size());
    Mat xprime(nbPts, 1, CV_64FC1);
    Mat yprime(nbPts, 1, CV_64FC1);

    Matx41d I4_0, J4_0;
    Matx33d rotation1, rotation2, rotation_best;
    Matx31d translation1, translation2, translation_best;

    const int max_iter = 20;
    for (int iter = 0; iter < max_iter && residual < residual_prev; iter++)
    {
        residual_prev = residual;
        rotation_best = R;
        translation_best = t;

        for (size_t i = 0; i < objectPointsCentered.size(); i++)
        {
            const Point3d& pt = objectPointsCentered[i];
            double epsilon = (R(2,0)*pt.x + R(2,1)*pt.y + R(2,2)*pt.z) / t(2,0);

            const Point2d& imPt = imagePoints[i];
            xprime.at<double>(static_cast<int>(i),0) = (1 + epsilon)*imPt.x;
            yprime.at<double>(static_cast<int>(i),0) = (1 + epsilon)*imPt.y;
        }

        I4_0 = Matx41d(Mat(objectMatrix * xprime));
        J4_0 = Matx41d(Mat(objectMatrix * yprime));

        PosCopl(objectPointsCentered, imagePoints, objectMatrix, U, rotation1, translation1, rotation2, translation2);

        int errorCode1 = 0, errorCode2 = 0;
        checkNotBehindCamera(objectPointsCentered, rotation1, translation1, rotation2, translation2, errorCode1, errorCode2);

        if (errorCode1 == -1 && errorCode2 == -1)
        {
            R = rotation_best;
            t = translation_best;
            break;
        }

        if (errorCode1 == 0 && errorCode2 == -1)
        {
            residual = computeResidual(objectPointsCentered, imagePoints, rotation1, translation1);
            R = rotation1;
            t = translation1;
        }
        else if (errorCode1 == -1 && errorCode2 == 0)
        {
            residual = computeResidual(objectPointsCentered, imagePoints, rotation1, translation1);
            R = rotation2;
            t = translation2;
        }
        else
        {
            double res1 = computeResidual(objectPointsCentered, imagePoints, rotation1, translation1);
            double res2 = computeResidual(objectPointsCentered, imagePoints, rotation2, translation2);

            if (res1 < res2)
            {
                residual = res1;
                R = rotation1;
                t = translation1;
            }
            else
            {
                residual = res2;
                R = rotation2;
                t = translation2;
            }
        }

        if (residual > residual_prev)
        {
            R = rotation_best;
            t = translation_best;
            residual = residual_prev;
            break;
        }
    }
}

/*retourne les DEUX poses resultant de la convergence des deux branches de POSIT, sans les juger*/
void PositCopl(const std::vector<Point3d>& objectPointsCentered, const std::vector<Point2d>& imagePoints, const Mat& objectMatrix,
               const Matx41d& U, Matx33d& Rot1, Matx31d& Trans1, Matx33d& Rot2, Matx31d& Trans2, int& nbSol)
{
    PosCopl(objectPointsCentered, imagePoints, objectMatrix, U, Rot1, Trans1, Rot2, Trans2);

    //Check that all object points are in front of the camera
    Matx31d rvec1, rvec2;
    Rodrigues(Rot1, rvec1);
    Rodrigues(Rot2, rvec2);

    int errorCode1 = 0, errorCode2 = 0;
    checkNotBehindCamera(objectPointsCentered, Rot1, Trans1, Rot2, Trans2, errorCode1, errorCode2);

    double residual1 = 1e6, residual2 = 1e6;
    if (errorCode1 == -1 && errorCode2 == -1)
    {
        nbSol = 0;
    }
    else if (errorCode1 == 0 && errorCode2 == -1)
    {
        nbSol = 1;
        Transf(objectPointsCentered, imagePoints, objectMatrix, U, Rot1, Trans1, residual1);
    }
    else if (errorCode1 == -1 && errorCode2 == 0)
    {
        nbSol = 1;
        Transf(objectPointsCentered, imagePoints, objectMatrix, U, Rot1, Trans1, residual1);
    }
    else
    {
        nbSol = 2;

        Transf(objectPointsCentered, imagePoints, objectMatrix, U, Rot1, Trans1, residual1);
        Transf(objectPointsCentered, imagePoints, objectMatrix, U, Rot2, Trans2, residual2);

        if (residual1 > residual2)
        {
            Matx33d R_tmp = Rot2;
            Matx31d Trans_tmp = Trans2;

            Rot2 = Rot1;
            Trans2 = Trans1;

            Rot1 = R_tmp;
            Trans1 = Trans_tmp;
        }
    }
}

void Composit(const vector<Point3d>& objectPoints, const vector<Point2d>& imagePoints,
              Matx33d& R1, Matx31d& t1, Matx33d& R2, Matx31d& t2, int& nbSol)
{
    // compute the cog of the object points
    Point3d cog;
    for (size_t i = 0; i < objectPoints.size(); i++)
    {
        const Point3d& pt = objectPoints[i];
        cog.x += pt.x;
        cog.y += pt.y;
        cog.z += pt.z;
    }
    cog.x /= objectPoints.size();
    cog.y /= objectPoints.size();
    cog.z /= objectPoints.size();

    int np = static_cast<int>(objectPoints.size());
    vector<Point3d> objectPointsCentered(objectPoints.size());
    Mat coplVectors(np, 4, CV_64FC1);
    for (int i = 0; i < np; i++)
    {
        Point3d pt = objectPoints[i] - cog;
        coplVectors.at<double>(i,0) = pt.x;
        coplVectors.at<double>(i,1) = pt.y;
        coplVectors.at<double>(i,2) = pt.z;
        coplVectors.at<double>(i,3) = 1;

        objectPointsCentered[i] = pt;
    }

    //TODO: debug
    std::cout << "coplVectors:\n" << coplVectors << std::endl;

    Mat w_, u, vt;
    SVDecomp(coplVectors, w_, u, vt, SVD::FULL_UV);
    Mat w_reci = Mat::zeros(coplVectors.rows, coplVectors.cols, CV_64FC1);
    int min_sz = std::min(w_reci.rows, w_reci.cols);
    for (int i = 0; i < min_sz; i++)
    {
        w_reci.at<double>(i,i) = std::fabs(w_.at<double>(i,0)) > std::numeric_limits<double>::epsilon() ? 1 / w_.at<double>(i,0) : 0;
    }

    // pseudo-inverse
    Mat coplMatrix = (u * w_reci * vt).t();

    //TODO: debug
    std::cout << "u:\n" << u << std::endl;
    std::cout << "w:\n" << w_ << std::endl;
    std::cout << "w_reci:\n" << w_reci << std::endl;
    std::cout << "vt:\n" << vt << std::endl;
    std::cout << "coplMatrix:\n" << coplMatrix << std::endl;

    Matx41d U = vt.col(w_.rows - 1);
    std::cout << "U: " << U.t() << std::endl;

    /*retourne les DEUX poses resultant de la convergence des deux branches de POSIT, sans les juger*/
    PositCopl(objectPointsCentered, imagePoints, coplMatrix, U, R1, t1, R2, t2, nbSol);

    if (nbSol == 0)
    {
        std::cerr << "NO SOLUTION" << std::endl;
        //TODO:
    }
    else
    {
        t1(0,0) -= R1(0,0)*cog.x + R1(0,1)*cog.y + R1(0,2)*cog.z;
        t1(1,0) -= R1(1,0)*cog.x + R1(1,1)*cog.y + R1(1,2)*cog.z;
        t1(2,0) -= R1(2,0)*cog.x + R1(2,1)*cog.y + R1(2,2)*cog.z;

        if (nbSol == 2)
        {
            t2(0,0) -= R2(0,0)*cog.x + R2(0,1)*cog.y + R2(0,2)*cog.z;
            t2(1,0) -= R2(1,0)*cog.x + R2(1,1)*cog.y + R2(1,2)*cog.z;
            t2(2,0) -= R2(2,0)*cog.x + R2(2,1)*cog.y + R2(2,2)*cog.z;
        }
    }
}

//Statistics Helpers
struct ErrorInfo
{
    ErrorInfo(double errT, double errR) : errorTrans(errT), errorRot(errR)
    {
    }

    bool operator<(const ErrorInfo& e) const
    {
        return sqrt(errorTrans*errorTrans + errorRot*errorRot) <
                sqrt(e.errorTrans*e.errorTrans + e.errorRot*e.errorRot);
    }

    double errorTrans;
    double errorRot;
};

//Try to find the translation and rotation thresholds to achieve a predefined percentage of success.
//Since a success is defined by error_trans < trans_thresh && error_rot < rot_thresh
//this just gives an idea of the values to use
static void findThreshold(const std::vector<double>& v_trans, const std::vector<double>& v_rot, double percentage,
                          double& transThresh, double& rotThresh)
{
    if (v_trans.empty() || v_rot.empty() || v_trans.size() != v_rot.size())
    {
        transThresh = -1;
        rotThresh = -1;
        return;
    }

    std::vector<ErrorInfo> error_info;
    error_info.reserve(v_trans.size());
    for (size_t i = 0; i < v_trans.size(); i++)
    {
        error_info.push_back(ErrorInfo(v_trans[i], v_rot[i]));
    }

    std::sort(error_info.begin(), error_info.end());
    size_t idx = static_cast<size_t>(error_info.size() * percentage);
    transThresh = error_info[idx].errorTrans;
    rotThresh = error_info[idx].errorRot;
}

static double getMax(const std::vector<double>& v)
{
    return *std::max_element(v.begin(), v.end());
}

static double getMean(const std::vector<double>& v)
{
    if (v.empty())
    {
        return 0.0;
    }

    double sum = std::accumulate(v.begin(), v.end(), 0.0);
    return sum / v.size();
}

static double getMedian(const std::vector<double>& v)
{
    if (v.empty())
    {
        return 0.0;
    }

    std::vector<double> v_copy = v;
    size_t size = v_copy.size();

    size_t n = size / 2;
    std::nth_element(v_copy.begin(), v_copy.begin() + n, v_copy.end());
    double val_n = v_copy[n];

    if (size % 2 == 1)
    {
        return val_n;
    } else
    {
        std::nth_element(v_copy.begin(), v_copy.begin() + n - 1, v_copy.end());
        return 0.5 * (val_n + v_copy[n - 1]);
    }
}

static void generatePose(const vector<Point3d>& points, Mat& rvec, Mat& tvec, RNG& rng, int nbTrials=10)
{
    const double minVal = 1.0e-3;
    const double maxVal = 1.0;
    rvec.create(3, 1, CV_64FC1);
    tvec.create(3, 1, CV_64FC1);

    bool validPose = false;
    for (int trial = 0; trial < nbTrials && !validPose; trial++)
    {
        for (int i = 0; i < 3; i++)
        {
            rvec.at<double>(i,0) = rng.uniform(minVal, maxVal);
            tvec.at<double>(i,0) = (i == 2) ? rng.uniform(minVal*10, maxVal) : rng.uniform(-maxVal, maxVal);
        }

        Mat R;
        cv::Rodrigues(rvec, R);
        bool positiveDepth = true;
        for (size_t i = 0; i < points.size() && positiveDepth; i++)
        {
            Matx31d objPts(points[i].x, points[i].y, points[i].z);
            Mat camPts = R*objPts + tvec;
            if (camPts.at<double>(2,0) <= 0)
            {
                positiveDepth = false;
            }
        }
        validPose = positiveDepth;
    }
}

static void generatePose(const vector<Point3f>& points, Mat& rvec, Mat& tvec, RNG& rng, int nbTrials=10)
{
    vector<Point3d> points_double(points.size());

    for (size_t i = 0; i < points.size(); i++)
    {
        points_double[i] = Point3d(points[i].x, points[i].y, points[i].z);
    }

    generatePose(points_double, rvec, tvec, rng, nbTrials);
}

static std::string printMethod(int method)
{
    switch (method) {
    case 0:
        return "SOLVEPNP_ITERATIVE";
    case 1:
        return "SOLVEPNP_EPNP";
    case 2:
        return "SOLVEPNP_P3P";
    case 3:
        return "SOLVEPNP_DLS (remaped to SOLVEPNP_EPNP)";
    case 4:
        return "SOLVEPNP_UPNP (remaped to SOLVEPNP_EPNP)";
    case 5:
        return "SOLVEPNP_AP3P";
    case 6:
        return "SOLVEPNP_IPPE";
    case 7:
        return "SOLVEPNP_IPPE_SQUARE";
    default:
        return "Unknown value";
    }
}

class CV_solvePnPRansac_Test : public cvtest::BaseTest
{
public:
    CV_solvePnPRansac_Test(bool planar_=false, bool planarTag_=false) : planar(planar_), planarTag(planarTag_)
    {
        eps[SOLVEPNP_ITERATIVE] = 1.0e-2;
        eps[SOLVEPNP_EPNP] = 1.0e-2;
        eps[SOLVEPNP_P3P] = 1.0e-2;
        eps[SOLVEPNP_AP3P] = 1.0e-2;
        eps[SOLVEPNP_DLS] = 1.0e-2;
        eps[SOLVEPNP_UPNP] = 1.0e-2;
        totalTestsCount = 10;
        pointsCount = 500;
    }
    ~CV_solvePnPRansac_Test() {}
protected:
    void generate3DPointCloud(vector<Point3f>& points,
                              Point3f pmin = Point3f(-1, -1, 5),
                              Point3f pmax = Point3f(1, 1, 10))
    {
        RNG& rng = theRNG(); // fix the seed to use "fixed" input 3D points

        for (size_t i = 0; i < points.size(); i++)
        {
            float _x = rng.uniform(pmin.x, pmax.x);
            float _y = rng.uniform(pmin.y, pmax.y);
            float _z = rng.uniform(pmin.z, pmax.z);
            points[i] = Point3f(_x, _y, _z);
        }
    }

    void generatePlanarPointCloud(vector<Point3f>& points,
                                  Point2f pmin = Point2f(-1, -1),
                                  Point2f pmax = Point2f(1, 1))
    {
        RNG& rng = theRNG(); // fix the seed to use "fixed" input 3D points

        if (planarTag)
        {
            const float squareLength_2 = rng.uniform(0.01f, pmax.x) / 2;
            points.clear();
            points.push_back(Point3f(-squareLength_2, squareLength_2, 0));
            points.push_back(Point3f(squareLength_2, squareLength_2, 0));
            points.push_back(Point3f(squareLength_2, -squareLength_2, 0));
            points.push_back(Point3f(-squareLength_2, -squareLength_2, 0));
        }
        else
        {
            Mat rvec_double, tvec_double;
            generatePose(points, rvec_double, tvec_double, rng);

            Mat rvec, tvec, R;
            rvec_double.convertTo(rvec, CV_32F);
            tvec_double.convertTo(tvec, CV_32F);
            cv::Rodrigues(rvec, R);

            for (size_t i = 0; i < points.size(); i++)
            {
                float x = rng.uniform(pmin.x, pmax.x);
                float y = rng.uniform(pmin.y, pmax.y);
                float z = 0;

                Matx31f pt(x, y, z);
                Mat pt_trans = R * pt + tvec;
                points[i] = Point3f(pt_trans.at<float>(0,0), pt_trans.at<float>(1,0), pt_trans.at<float>(2,0));
            }
        }
    }

    void generateCameraMatrix(Mat& cameraMatrix, RNG& rng)
    {
        const double fcMinVal = 1e-3;
        const double fcMaxVal = 100;
        cameraMatrix.create(3, 3, CV_64FC1);
        cameraMatrix.setTo(Scalar(0));
        cameraMatrix.at<double>(0,0) = rng.uniform(fcMinVal, fcMaxVal);
        cameraMatrix.at<double>(1,1) = rng.uniform(fcMinVal, fcMaxVal);
        cameraMatrix.at<double>(0,2) = rng.uniform(fcMinVal, fcMaxVal);
        cameraMatrix.at<double>(1,2) = rng.uniform(fcMinVal, fcMaxVal);
        cameraMatrix.at<double>(2,2) = 1;
    }

    void generateDistCoeffs(Mat& distCoeffs, RNG& rng)
    {
        distCoeffs = Mat::zeros(4, 1, CV_64FC1);
        for (int i = 0; i < 3; i++)
            distCoeffs.at<double>(i,0) = rng.uniform(0.0, 1.0e-6);
    }

    virtual bool runTest(RNG& rng, int mode, int method, const vector<Point3f>& points, double& errorTrans, double& errorRot)
    {
        if ((!planar && method == SOLVEPNP_IPPE) || method == SOLVEPNP_IPPE_SQUARE)
        {
            return true;
        }

        Mat rvec, tvec;
        vector<int> inliers;
        Mat trueRvec, trueTvec;
        Mat intrinsics, distCoeffs;
        generateCameraMatrix(intrinsics, rng);
        //UPnP is mapped to EPnP
        //Uncomment this when UPnP is fixed
//        if (method == SOLVEPNP_UPNP)
//        {
//            intrinsics.at<double>(1,1) = intrinsics.at<double>(0,0);
//        }
        if (mode == 0)
        {
            distCoeffs = Mat::zeros(4, 1, CV_64FC1);
        }
        else
        {
            generateDistCoeffs(distCoeffs, rng);
        }

        generatePose(points, trueRvec, trueTvec, rng);

        vector<Point2f> projectedPoints;
        projectedPoints.resize(points.size());
        projectPoints(points, trueRvec, trueTvec, intrinsics, distCoeffs, projectedPoints);
        for (size_t i = 0; i < projectedPoints.size(); i++)
        {
            if (i % 20 == 0)
            {
                projectedPoints[i] = projectedPoints[rng.uniform(0,(int)points.size()-1)];
            }
        }

        solvePnPRansac(points, projectedPoints, intrinsics, distCoeffs, rvec, tvec, false, pointsCount, 0.5f, 0.99, inliers, method);

        bool isTestSuccess = inliers.size() >= points.size()*0.95;

        double rvecDiff = cvtest::norm(rvec, trueRvec, NORM_L2), tvecDiff = cvtest::norm(tvec, trueTvec, NORM_L2);
        isTestSuccess = isTestSuccess && rvecDiff < eps[method] && tvecDiff < eps[method];
        errorTrans = tvecDiff;
        errorRot = rvecDiff;

        return isTestSuccess;
    }

    virtual void run(int)
    {
        ts->set_failed_test_info(cvtest::TS::OK);

        vector<Point3f> points, points_dls;
        points.resize(static_cast<size_t>(pointsCount));

        if (planar || planarTag)
        {
            generatePlanarPointCloud(points);
        }
        else
        {
            generate3DPointCloud(points);
        }

        RNG& rng = ts->get_rng();

        for (int mode = 0; mode < 2; mode++)
        {
            for (int method = 0; method < SOLVEPNP_MAX_COUNT; method++)
            {
                //To get the same input for each methods
                RNG rngCopy = rng;
                std::vector<double> vec_errorTrans, vec_errorRot;
                vec_errorTrans.reserve(static_cast<size_t>(totalTestsCount));
                vec_errorRot.reserve(static_cast<size_t>(totalTestsCount));

                int successfulTestsCount = 0;
                for (int testIndex = 0; testIndex < totalTestsCount; testIndex++)
                {
                    double errorTrans, errorRot;
                    if (runTest(rngCopy, mode, method, points, errorTrans, errorRot))
                    {
                        successfulTestsCount++;
                    }
                    vec_errorTrans.push_back(errorTrans);
                    vec_errorRot.push_back(errorRot);
                }

                double maxErrorTrans = getMax(vec_errorTrans);
                double maxErrorRot = getMax(vec_errorRot);
                double meanErrorTrans = getMean(vec_errorTrans);
                double meanErrorRot = getMean(vec_errorRot);
                double medianErrorTrans = getMedian(vec_errorTrans);
                double medianErrorRot = getMedian(vec_errorRot);

                if (successfulTestsCount < 0.7*totalTestsCount)
                {
                    ts->printf(cvtest::TS::LOG, "Invalid accuracy for %s, failed %d tests from %d, %s, "
                                                "maxErrT: %f, maxErrR: %f, "
                                                "meanErrT: %f, meanErrR: %f, "
                                                "medErrT: %f, medErrR: %f\n",
                               printMethod(method).c_str(), totalTestsCount - successfulTestsCount, totalTestsCount, printMode(mode).c_str(),
                               maxErrorTrans, maxErrorRot, meanErrorTrans, meanErrorRot, medianErrorTrans, medianErrorRot);
                    ts->set_failed_test_info(cvtest::TS::FAIL_BAD_ACCURACY);
                }
                cout << "mode: " << printMode(mode) << ", method: " << printMethod(method) << " -> "
                     << ((double)successfulTestsCount / totalTestsCount) * 100 << "%"
                     << " (maxErrT: " << maxErrorTrans << ", maxErrR: " << maxErrorRot
                     << ", meanErrT: " << meanErrorTrans << ", meanErrR: " << meanErrorRot
                     << ", medErrT: " << medianErrorTrans << ", medErrR: " << medianErrorRot << ")" << endl;
                double transThres, rotThresh;
                findThreshold(vec_errorTrans, vec_errorRot, 0.7, transThres, rotThresh);
                cout << "approximate translation threshold for 0.7: " << transThres
                     << ", approximate rotation threshold for 0.7: " << rotThresh << endl;
            }
            cout << endl;
        }
    }
    std::string printMode(int mode)
    {
        switch (mode) {
        case 0:
            return "no distortion";
        case 1:
        default:
            return "distorsion";
        }
    }
    double eps[SOLVEPNP_MAX_COUNT];
    int totalTestsCount;
    int pointsCount;
    bool planar;
    bool planarTag;
};

class CV_solvePnP_Test : public CV_solvePnPRansac_Test
{
public:
    CV_solvePnP_Test(bool planar_=false, bool planarTag_=false) : CV_solvePnPRansac_Test(planar_, planarTag_)
    {
        eps[SOLVEPNP_ITERATIVE] = 1.0e-6;
        eps[SOLVEPNP_EPNP] = 1.0e-6;
        eps[SOLVEPNP_P3P] = 2.0e-4;
        eps[SOLVEPNP_AP3P] = 1.0e-4;
        eps[SOLVEPNP_DLS] = 1.0e-6; //DLS is remapped to EPnP, so we use the same threshold
        eps[SOLVEPNP_UPNP] = 1.0e-6; //UPnP is remapped to EPnP, so we use the same threshold
        eps[SOLVEPNP_IPPE] = 1.0e-6;
        eps[SOLVEPNP_IPPE_SQUARE] = 1.0e-6;

        totalTestsCount = 1000;

        if (planar || planarTag)
        {
            if (planarTag)
            {
                pointsCount = 4;
            }
            else
            {
                pointsCount = 30;
            }
        }
        else
        {
            pointsCount = 500;
        }
    }

    ~CV_solvePnP_Test() {}
protected:
    virtual bool runTest(RNG& rng, int mode, int method, const vector<Point3f>& points, double& errorTrans, double& errorRot)
    {
        if ((!planar && (method == SOLVEPNP_IPPE || method == SOLVEPNP_IPPE_SQUARE)) ||
            (!planarTag && method == SOLVEPNP_IPPE_SQUARE))
        {
            errorTrans = -1;
            errorRot = -1;
            //SOLVEPNP_IPPE and SOLVEPNP_IPPE_SQUARE need planar object
            return true;
        }

        //Tune thresholds...
        double epsilon_trans[SOLVEPNP_MAX_COUNT];
        memcpy(epsilon_trans, eps, SOLVEPNP_MAX_COUNT * sizeof(*epsilon_trans));

        double epsilon_rot[SOLVEPNP_MAX_COUNT];
        memcpy(epsilon_rot, eps, SOLVEPNP_MAX_COUNT * sizeof(*epsilon_rot));

        if (planar)
        {
            if (mode == 0)
            {
                epsilon_trans[SOLVEPNP_EPNP] = 5.0e-3;
                epsilon_trans[SOLVEPNP_DLS] = 5.0e-3;
                epsilon_trans[SOLVEPNP_UPNP] = 5.0e-3;

                epsilon_rot[SOLVEPNP_EPNP] = 5.0e-3;
                epsilon_rot[SOLVEPNP_DLS] = 5.0e-3;
                epsilon_rot[SOLVEPNP_UPNP] = 5.0e-3;
            }
            else
            {
                epsilon_trans[SOLVEPNP_ITERATIVE] = 1e-4;
                epsilon_trans[SOLVEPNP_EPNP] = 5e-3;
                epsilon_trans[SOLVEPNP_DLS] = 5e-3;
                epsilon_trans[SOLVEPNP_UPNP] = 5e-3;
                epsilon_trans[SOLVEPNP_P3P] = 1e-4;
                epsilon_trans[SOLVEPNP_AP3P] = 1e-4;
                epsilon_trans[SOLVEPNP_IPPE] = 1e-4;
                epsilon_trans[SOLVEPNP_IPPE_SQUARE] = 1e-4;

                epsilon_rot[SOLVEPNP_ITERATIVE] = 1e-4;
                epsilon_rot[SOLVEPNP_EPNP] = 5e-3;
                epsilon_rot[SOLVEPNP_DLS] = 5e-3;
                epsilon_rot[SOLVEPNP_UPNP] = 5e-3;
                epsilon_rot[SOLVEPNP_P3P] = 1e-4;
                epsilon_rot[SOLVEPNP_AP3P] = 1e-4;
                epsilon_rot[SOLVEPNP_IPPE] = 1e-4;
                epsilon_rot[SOLVEPNP_IPPE_SQUARE] = 1e-4;
            }
        }

        Mat trueRvec, trueTvec;
        Mat intrinsics, distCoeffs;
        generateCameraMatrix(intrinsics, rng);
        //UPnP is mapped to EPnP
        //Uncomment this when UPnP is fixed
//        if (method == SOLVEPNP_UPNP)
//        {
//            intrinsics.at<double>(1,1) = intrinsics.at<double>(0,0);
//        }
        if (mode == 0)
        {
            distCoeffs = Mat::zeros(4, 1, CV_64FC1);
        }
        else
        {
            generateDistCoeffs(distCoeffs, rng);
        }

        generatePose(points, trueRvec, trueTvec, rng);

        std::vector<Point3f> opoints;
        switch(method)
        {
            case SOLVEPNP_P3P:
            case SOLVEPNP_AP3P:
                opoints = std::vector<Point3f>(points.begin(), points.begin()+4);
                break;
                //UPnP is mapped to EPnP
                //Uncomment this when UPnP is fixed
//            case SOLVEPNP_UPNP:
//                if (points.size() > 50)
//                {
//                    opoints = std::vector<Point3f>(points.begin(), points.begin()+50);
//                }
//                else
//                {
//                    opoints = points;
//                }
//                break;
            default:
                opoints = points;
                break;
        }

        vector<Point2f> projectedPoints;
        projectedPoints.resize(opoints.size());
        projectPoints(opoints, trueRvec, trueTvec, intrinsics, distCoeffs, projectedPoints);

        Mat rvec, tvec;
        bool isEstimateSuccess = solvePnP(opoints, projectedPoints, intrinsics, distCoeffs, rvec, tvec, false, method);

        if (!isEstimateSuccess)
        {
            return false;
        }

        double rvecDiff = cvtest::norm(rvec, trueRvec, NORM_L2), tvecDiff = cvtest::norm(tvec, trueTvec, NORM_L2);
        bool isTestSuccess = rvecDiff < epsilon_rot[method] && tvecDiff < epsilon_trans[method];

        errorTrans = tvecDiff;
        errorRot = rvecDiff;

        return isTestSuccess;
    }
};

class CV_solveP3P_Test : public CV_solvePnPRansac_Test
{
public:
    CV_solveP3P_Test()
    {
        eps[SOLVEPNP_P3P] = 2.0e-4;
        eps[SOLVEPNP_AP3P] = 1.0e-4;
        totalTestsCount = 1000;
    }

    ~CV_solveP3P_Test() {}
protected:
    virtual bool runTest(RNG& rng, int mode, int method, const vector<Point3f>& points, double& errorTrans, double& errorRot)
    {
        std::vector<Mat> rvecs, tvecs;
        Mat trueRvec, trueTvec;
        Mat intrinsics, distCoeffs;
        generateCameraMatrix(intrinsics, rng);
        if (mode == 0)
        {
            distCoeffs = Mat::zeros(4, 1, CV_64FC1);
        }
        else
        {
            generateDistCoeffs(distCoeffs, rng);
        }
        generatePose(points, trueRvec, trueTvec, rng);

        std::vector<Point3f> opoints;
        opoints = std::vector<Point3f>(points.begin(), points.begin()+3);

        vector<Point2f> projectedPoints;
        projectedPoints.resize(opoints.size());
        projectPoints(opoints, trueRvec, trueTvec, intrinsics, distCoeffs, projectedPoints);

        int num_of_solutions = solveP3P(opoints, projectedPoints, intrinsics, distCoeffs, rvecs, tvecs, method);
        if (num_of_solutions != (int) rvecs.size() || num_of_solutions != (int) tvecs.size() || num_of_solutions == 0)
        {
            return false;
        }

        bool isTestSuccess = false;
        for (size_t i = 0; i < rvecs.size() && !isTestSuccess; i++) {
            double rvecDiff = cvtest::norm(rvecs[i], trueRvec, NORM_L2);
            double tvecDiff = cvtest::norm(tvecs[i], trueTvec, NORM_L2);
            isTestSuccess = rvecDiff < eps[method] && tvecDiff < eps[method];

            errorTrans = std::min(errorTrans, tvecDiff);
            errorRot = std::min(errorRot, rvecDiff);
        }

        return isTestSuccess;
    }

    virtual void run(int)
    {
        ts->set_failed_test_info(cvtest::TS::OK);

        vector<Point3f> points;
        points.resize(static_cast<size_t>(pointsCount));
        generate3DPointCloud(points);

        const int methodsCount = 2;
        int methods[] = {SOLVEPNP_P3P, SOLVEPNP_AP3P};
        RNG rng = ts->get_rng();

        for (int mode = 0; mode < 2; mode++)
        {
            //To get the same input for each methods
            RNG rngCopy = rng;
            for (int method = 0; method < methodsCount; method++)
            {
                std::vector<double> vec_errorTrans, vec_errorRot;
                vec_errorTrans.reserve(static_cast<size_t>(totalTestsCount));
                vec_errorRot.reserve(static_cast<size_t>(totalTestsCount));

                int successfulTestsCount = 0;
                for (int testIndex = 0; testIndex < totalTestsCount; testIndex++)
                {
                    double errorTrans = 0, errorRot = 0;
                    if (runTest(rngCopy, mode, methods[method], points, errorTrans, errorRot))
                    {
                        successfulTestsCount++;
                    }
                    vec_errorTrans.push_back(errorTrans);
                    vec_errorRot.push_back(errorRot);
                }

                double maxErrorTrans = getMax(vec_errorTrans);
                double maxErrorRot = getMax(vec_errorRot);
                double meanErrorTrans = getMean(vec_errorTrans);
                double meanErrorRot = getMean(vec_errorRot);
                double medianErrorTrans = getMedian(vec_errorTrans);
                double medianErrorRot = getMedian(vec_errorRot);

                if (successfulTestsCount < 0.7*totalTestsCount)
                {
                    ts->printf(cvtest::TS::LOG, "Invalid accuracy for %s, failed %d tests from %d, %s, "
                                                "maxErrT: %f, maxErrR: %f, "
                                                "meanErrT: %f, meanErrR: %f, "
                                                "medErrT: %f, medErrR: %f\n",
                               printMethod(methods[method]).c_str(), totalTestsCount - successfulTestsCount, totalTestsCount, printMode(mode).c_str(),
                               maxErrorTrans, maxErrorRot, meanErrorTrans, meanErrorRot, medianErrorTrans, medianErrorRot);
                    ts->set_failed_test_info(cvtest::TS::FAIL_BAD_ACCURACY);
                }
                cout << "mode: " << printMode(mode) << ", method: " << printMethod(methods[method]) << " -> "
                     << ((double)successfulTestsCount / totalTestsCount) * 100 << "%"
                     << " (maxErrT: " << maxErrorTrans << ", maxErrR: " << maxErrorRot
                     << ", meanErrT: " << meanErrorTrans << ", meanErrR: " << meanErrorRot
                     << ", medErrT: " << medianErrorTrans << ", medErrR: " << medianErrorRot << ")" << endl;
                double transThres, rotThresh;
                findThreshold(vec_errorTrans, vec_errorRot, 0.7, transThres, rotThresh);
                cout << "approximate translation threshold for 0.7: " << transThres
                     << ", approximate rotation threshold for 0.7: " << rotThresh << endl;
            }
        }
    }
};


TEST(Calib3d_SolveP3P, accuracy) { CV_solveP3P_Test test; test.safe_run();}
TEST(Calib3d_SolvePnPRansac, accuracy) { CV_solvePnPRansac_Test test; test.safe_run(); }
TEST(Calib3d_SolvePnP, accuracy) { CV_solvePnP_Test test; test.safe_run(); }
TEST(Calib3d_SolvePnP, accuracy_planar) { CV_solvePnP_Test test(true); test.safe_run(); }
TEST(Calib3d_SolvePnP, accuracy_planar_tag) { CV_solvePnP_Test test(true, true); test.safe_run(); }

TEST(Calib3d_SolvePnPRansac, concurrency)
{
    int count = 7*13;

    Mat object(1, count, CV_32FC3);
    randu(object, -100, 100);

    Mat camera_mat(3, 3, CV_32FC1);
    randu(camera_mat, 0.5, 1);
    camera_mat.at<float>(0, 1) = 0.f;
    camera_mat.at<float>(1, 0) = 0.f;
    camera_mat.at<float>(2, 0) = 0.f;
    camera_mat.at<float>(2, 1) = 0.f;
    camera_mat.at<float>(2, 2) = 1.f;

    Mat dist_coef(1, 8, CV_32F, cv::Scalar::all(0));

    vector<cv::Point2f> image_vec;
    Mat rvec_gold(1, 3, CV_32FC1);
    randu(rvec_gold, 0, 1);
    Mat tvec_gold(1, 3, CV_32FC1);
    randu(tvec_gold, 0, 1);
    projectPoints(object, rvec_gold, tvec_gold, camera_mat, dist_coef, image_vec);

    Mat image(1, count, CV_32FC2, &image_vec[0]);

    Mat rvec1, rvec2;
    Mat tvec1, tvec2;

    int threads = getNumThreads();
    {
        // limit concurrency to get deterministic result
        theRNG().state = 20121010;
        setNumThreads(1);
        solvePnPRansac(object, image, camera_mat, dist_coef, rvec1, tvec1);
    }

    {
        setNumThreads(threads);
        Mat rvec;
        Mat tvec;
        // parallel executions
        for(int i = 0; i < 10; ++i)
        {
            cv::theRNG().state = 20121010;
            solvePnPRansac(object, image, camera_mat, dist_coef, rvec, tvec);
        }
    }

    {
        // single thread again
        theRNG().state = 20121010;
        setNumThreads(1);
        solvePnPRansac(object, image, camera_mat, dist_coef, rvec2, tvec2);
    }

    double rnorm = cvtest::norm(rvec1, rvec2, NORM_INF);
    double tnorm = cvtest::norm(tvec1, tvec2, NORM_INF);

    EXPECT_LT(rnorm, 1e-6);
    EXPECT_LT(tnorm, 1e-6);
}

TEST(Calib3d_SolvePnPRansac, input_type)
{
    const int numPoints = 10;
    Matx33d intrinsics(5.4794130238156129e+002, 0., 2.9835545700043139e+002, 0.,
                       5.4817724002728005e+002, 2.3062194051986233e+002, 0., 0., 1.);

    std::vector<cv::Point3f> points3d;
    std::vector<cv::Point2f> points2d;
    for (int i = 0; i < numPoints; i+=2)
    {
        points3d.push_back(cv::Point3i(5+i, 3, 2));
        points3d.push_back(cv::Point3i(5+i, 3+i, 2+i));
        points2d.push_back(cv::Point2i(0, i));
        points2d.push_back(cv::Point2i(-i, i));
    }
    Mat R1, t1, R2, t2, R3, t3, R4, t4;

    EXPECT_TRUE(solvePnPRansac(points3d, points2d, intrinsics, cv::Mat(), R1, t1));

    Mat points3dMat(points3d);
    Mat points2dMat(points2d);
    EXPECT_TRUE(solvePnPRansac(points3dMat, points2dMat, intrinsics, cv::Mat(), R2, t2));

    points3dMat = points3dMat.reshape(3, 1);
    points2dMat = points2dMat.reshape(2, 1);
    EXPECT_TRUE(solvePnPRansac(points3dMat, points2dMat, intrinsics, cv::Mat(), R3, t3));

    points3dMat = points3dMat.reshape(1, numPoints);
    points2dMat = points2dMat.reshape(1, numPoints);
    EXPECT_TRUE(solvePnPRansac(points3dMat, points2dMat, intrinsics, cv::Mat(), R4, t4));

    EXPECT_LE(cvtest::norm(R1, R2, NORM_INF), 1e-6);
    EXPECT_LE(cvtest::norm(t1, t2, NORM_INF), 1e-6);
    EXPECT_LE(cvtest::norm(R1, R3, NORM_INF), 1e-6);
    EXPECT_LE(cvtest::norm(t1, t3, NORM_INF), 1e-6);
    EXPECT_LE(cvtest::norm(R1, R4, NORM_INF), 1e-6);
    EXPECT_LE(cvtest::norm(t1, t4, NORM_INF), 1e-6);
}

TEST(Calib3d_SolvePnPRansac, double_support)
{
    Matx33d intrinsics(5.4794130238156129e+002, 0., 2.9835545700043139e+002, 0.,
                       5.4817724002728005e+002, 2.3062194051986233e+002, 0., 0., 1.);
    std::vector<cv::Point3d> points3d;
    std::vector<cv::Point2d> points2d;
    std::vector<cv::Point3f> points3dF;
    std::vector<cv::Point2f> points2dF;
    for (int i = 0; i < 10 ; i+=2)
    {
        points3d.push_back(cv::Point3d(5+i, 3, 2));
        points3dF.push_back(cv::Point3f(static_cast<float>(5+i), 3, 2));
        points3d.push_back(cv::Point3d(5+i, 3+i, 2+i));
        points3dF.push_back(cv::Point3f(static_cast<float>(5+i), static_cast<float>(3+i), static_cast<float>(2+i)));
        points2d.push_back(cv::Point2d(0, i));
        points2dF.push_back(cv::Point2f(0, static_cast<float>(i)));
        points2d.push_back(cv::Point2d(-i, i));
        points2dF.push_back(cv::Point2f(static_cast<float>(-i), static_cast<float>(i)));
    }
    Mat R, t, RF, tF;
    vector<int> inliers;

    solvePnPRansac(points3dF, points2dF, intrinsics, cv::Mat(), RF, tF, true, 100, 8.f, 0.999, inliers, cv::SOLVEPNP_P3P);
    solvePnPRansac(points3d, points2d, intrinsics, cv::Mat(), R, t, true, 100, 8.f, 0.999, inliers, cv::SOLVEPNP_P3P);

    EXPECT_LE(cvtest::norm(R, Mat_<double>(RF), NORM_INF), 1e-3);
    EXPECT_LE(cvtest::norm(t, Mat_<double>(tF), NORM_INF), 1e-3);
}

TEST(Calib3d_SolvePnP, input_type)
{
    Matx33d intrinsics(5.4794130238156129e+002, 0., 2.9835545700043139e+002, 0.,
                       5.4817724002728005e+002, 2.3062194051986233e+002, 0., 0., 1.);
    vector<Point3d> points3d_;
    vector<Point3f> points3dF_;
    //Cube
    const float l = -0.1f;
    //Front face
    points3d_.push_back(Point3d(-l, -l, -l));
    points3dF_.push_back(Point3f(-l, -l, -l));
    points3d_.push_back(Point3d(l, -l, -l));
    points3dF_.push_back(Point3f(l, -l, -l));
    points3d_.push_back(Point3d(l, l, -l));
    points3dF_.push_back(Point3f(l, l, -l));
    points3d_.push_back(Point3d(-l, l, -l));
    points3dF_.push_back(Point3f(-l, l, -l));
    //Back face
    points3d_.push_back(Point3d(-l, -l, l));
    points3dF_.push_back(Point3f(-l, -l, l));
    points3d_.push_back(Point3d(l, -l, l));
    points3dF_.push_back(Point3f(l, -l, l));
    points3d_.push_back(Point3d(l, l, l));
    points3dF_.push_back(Point3f(l, l, l));
    points3d_.push_back(Point3d(-l, l, l));
    points3dF_.push_back(Point3f(-l, l, l));

    Mat trueRvec = (Mat_<double>(3,1) << 0.1, -0.25, 0.467);
    Mat trueTvec = (Mat_<double>(3,1) << -0.21, 0.12, 0.746);

    for (int method = 0; method < SOLVEPNP_MAX_COUNT; method++)
    {
        vector<Point3d> points3d;
        vector<Point2d> points2d;
        vector<Point3f> points3dF;
        vector<Point2f> points2dF;

        if (method == SOLVEPNP_IPPE || method == SOLVEPNP_IPPE_SQUARE)
        {
            const float tagSize_2 = 0.05f / 2;
            points3d.push_back(Point3d(-tagSize_2,  tagSize_2, 0));
            points3d.push_back(Point3d( tagSize_2,  tagSize_2, 0));
            points3d.push_back(Point3d( tagSize_2, -tagSize_2, 0));
            points3d.push_back(Point3d(-tagSize_2, -tagSize_2, 0));

            points3dF.push_back(Point3f(-tagSize_2,  tagSize_2, 0));
            points3dF.push_back(Point3f( tagSize_2,  tagSize_2, 0));
            points3dF.push_back(Point3f( tagSize_2, -tagSize_2, 0));
            points3dF.push_back(Point3f(-tagSize_2, -tagSize_2, 0));
        }
        else if (method == SOLVEPNP_P3P || method == SOLVEPNP_AP3P)
        {
            points3d = vector<Point3d>(points3d_.begin(), points3d_.begin()+4);
            points3dF = vector<Point3f>(points3dF_.begin(), points3dF_.begin()+4);
        }
        else
        {
            points3d = points3d_;
            points3dF = points3dF_;
        }

        projectPoints(points3d, trueRvec, trueTvec, intrinsics, noArray(), points2d);
        projectPoints(points3dF, trueRvec, trueTvec, intrinsics, noArray(), points2dF);

        //solvePnP
        {
            Mat R, t, RF, tF;

            solvePnP(points3dF, points2dF, Matx33f(intrinsics), Mat(), RF, tF, false, method);
            solvePnP(points3d, points2d, intrinsics, Mat(), R, t, false, method);

            //By default rvec and tvec must be returned in double precision
            EXPECT_EQ(RF.type(), tF.type());
            EXPECT_EQ(RF.type(), CV_64FC1);

            EXPECT_EQ(R.type(), t.type());
            EXPECT_EQ(R.type(), CV_64FC1);

            EXPECT_LE(cvtest::norm(R, RF, NORM_INF), 1e-3);
            EXPECT_LE(cvtest::norm(t, tF, NORM_INF), 1e-3);

            EXPECT_LE(cvtest::norm(trueRvec, R, NORM_INF), 1e-3);
            EXPECT_LE(cvtest::norm(trueTvec, t, NORM_INF), 1e-3);
            EXPECT_LE(cvtest::norm(trueRvec, RF, NORM_INF), 1e-3);
            EXPECT_LE(cvtest::norm(trueTvec, tF, NORM_INF), 1e-3);
        }
        {
            Mat R1, t1, R2, t2;

            solvePnP(points3dF, points2d, intrinsics, Mat(), R1, t1, false, method);
            solvePnP(points3d, points2dF, intrinsics, Mat(), R2, t2, false, method);

            //By default rvec and tvec must be returned in double precision
            EXPECT_EQ(R1.type(), t1.type());
            EXPECT_EQ(R1.type(), CV_64FC1);

            EXPECT_EQ(R2.type(), t2.type());
            EXPECT_EQ(R2.type(), CV_64FC1);

            EXPECT_LE(cvtest::norm(R1, R2, NORM_INF), 1e-3);
            EXPECT_LE(cvtest::norm(t1, t2, NORM_INF), 1e-3);

            EXPECT_LE(cvtest::norm(trueRvec, R1, NORM_INF), 1e-3);
            EXPECT_LE(cvtest::norm(trueTvec, t1, NORM_INF), 1e-3);
            EXPECT_LE(cvtest::norm(trueRvec, R2, NORM_INF), 1e-3);
            EXPECT_LE(cvtest::norm(trueTvec, t2, NORM_INF), 1e-3);
        }
        {
            Mat R1(3,1,CV_32FC1), t1(3,1,CV_64FC1);
            Mat R2(3,1,CV_64FC1), t2(3,1,CV_32FC1);

            solvePnP(points3dF, points2d, intrinsics, Mat(), R1, t1, false, method);
            solvePnP(points3d, points2dF, intrinsics, Mat(), R2, t2, false, method);

            //If not null, rvec and tvec must be returned in the same precision
            EXPECT_EQ(R1.type(), CV_32FC1);
            EXPECT_EQ(t1.type(), CV_64FC1);

            EXPECT_EQ(R2.type(), CV_64FC1);
            EXPECT_EQ(t2.type(), CV_32FC1);

            EXPECT_LE(cvtest::norm(Mat_<double>(R1), R2, NORM_INF), 1e-3);
            EXPECT_LE(cvtest::norm(t1, Mat_<double>(t2), NORM_INF), 1e-3);

            EXPECT_LE(cvtest::norm(trueRvec, Mat_<double>(R1), NORM_INF), 1e-3);
            EXPECT_LE(cvtest::norm(trueTvec, t1, NORM_INF), 1e-3);
            EXPECT_LE(cvtest::norm(trueRvec, R2, NORM_INF), 1e-3);
            EXPECT_LE(cvtest::norm(trueTvec, Mat_<double>(t2), NORM_INF), 1e-3);
        }
        {
            Matx31f R1, t2;
            Matx31d R2, t1;

            solvePnP(points3dF, points2d, intrinsics, Mat(), R1, t1, false, method);
            solvePnP(points3d, points2dF, intrinsics, Mat(), R2, t2, false, method);

            Matx31d R1d(R1(0), R1(1), R1(2));
            Matx31d t2d(t2(0), t2(1), t2(2));

            EXPECT_LE(cvtest::norm(R1d, R2, NORM_INF), 1e-3);
            EXPECT_LE(cvtest::norm(t1, t2d, NORM_INF), 1e-3);

            EXPECT_LE(cvtest::norm(trueRvec, R1d, NORM_INF), 1e-3);
            EXPECT_LE(cvtest::norm(trueTvec, t1, NORM_INF), 1e-3);
            EXPECT_LE(cvtest::norm(trueRvec, R2, NORM_INF), 1e-3);
            EXPECT_LE(cvtest::norm(trueTvec, t2d, NORM_INF), 1e-3);
        }

        //solvePnPGeneric
        {
            vector<Mat> Rs, ts, RFs, tFs;

            int res1 = solvePnPGeneric(points3dF, points2dF, Matx33f(intrinsics), Mat(), RFs, tFs, false, (SolvePnPMethod)method);
            int res2 = solvePnPGeneric(points3d, points2d, intrinsics, Mat(), Rs, ts, false, (SolvePnPMethod)method);

            EXPECT_GT(res1, 0);
            EXPECT_GT(res2, 0);

            Mat R = Rs.front(), t = ts.front(), RF = RFs.front(), tF = tFs.front();

            //By default rvecs and tvecs must be returned in double precision
            EXPECT_EQ(RF.type(), tF.type());
            EXPECT_EQ(RF.type(), CV_64FC1);

            EXPECT_EQ(R.type(), t.type());
            EXPECT_EQ(R.type(), CV_64FC1);

            EXPECT_LE(cvtest::norm(R, RF, NORM_INF), 1e-3);
            EXPECT_LE(cvtest::norm(t, tF, NORM_INF), 1e-3);

            EXPECT_LE(cvtest::norm(trueRvec, R, NORM_INF), 1e-3);
            EXPECT_LE(cvtest::norm(trueTvec, t, NORM_INF), 1e-3);
            EXPECT_LE(cvtest::norm(trueRvec, RF, NORM_INF), 1e-3);
            EXPECT_LE(cvtest::norm(trueTvec, tF, NORM_INF), 1e-3);
        }
        {
            vector<Mat> R1s, t1s, R2s, t2s;

            int res1 = solvePnPGeneric(points3dF, points2d, intrinsics, Mat(), R1s, t1s, false, (SolvePnPMethod)method);
            int res2 = solvePnPGeneric(points3d, points2dF, intrinsics, Mat(), R2s, t2s, false, (SolvePnPMethod)method);

            EXPECT_GT(res1, 0);
            EXPECT_GT(res2, 0);

            Mat R1 = R1s.front(), t1 = t1s.front(), R2 = R2s.front(), t2 = t2s.front();

            //By default rvecs and tvecs must be returned in double precision
            EXPECT_EQ(R1.type(), t1.type());
            EXPECT_EQ(R1.type(), CV_64FC1);

            EXPECT_EQ(R2.type(), t2.type());
            EXPECT_EQ(R2.type(), CV_64FC1);

            EXPECT_LE(cvtest::norm(R1, R2, NORM_INF), 1e-3);
            EXPECT_LE(cvtest::norm(t1, t2, NORM_INF), 1e-3);

            EXPECT_LE(cvtest::norm(trueRvec, R1, NORM_INF), 1e-3);
            EXPECT_LE(cvtest::norm(trueTvec, t1, NORM_INF), 1e-3);
            EXPECT_LE(cvtest::norm(trueRvec, R2, NORM_INF), 1e-3);
            EXPECT_LE(cvtest::norm(trueTvec, t2, NORM_INF), 1e-3);
        }
        {
            vector<Mat_<float> > R1s, t2s;
            vector<Mat_<double> > R2s, t1s;

            int res1 = solvePnPGeneric(points3dF, points2d, intrinsics, Mat(), R1s, t1s, false, (SolvePnPMethod)method);
            int res2 = solvePnPGeneric(points3d, points2dF, intrinsics, Mat(), R2s, t2s, false, (SolvePnPMethod)method);

            EXPECT_GT(res1, 0);
            EXPECT_GT(res2, 0);

            Mat R1 = R1s.front(), t1 = t1s.front();
            Mat R2 = R2s.front(), t2 = t2s.front();

            //If not null, rvecs and tvecs must be returned in the same precision
            EXPECT_EQ(R1.type(), CV_32FC1);
            EXPECT_EQ(t1.type(), CV_64FC1);

            EXPECT_EQ(R2.type(), CV_64FC1);
            EXPECT_EQ(t2.type(), CV_32FC1);

            EXPECT_LE(cvtest::norm(Mat_<double>(R1), R2, NORM_INF), 1e-3);
            EXPECT_LE(cvtest::norm(t1, Mat_<double>(t2), NORM_INF), 1e-3);

            EXPECT_LE(cvtest::norm(trueRvec, Mat_<double>(R1), NORM_INF), 1e-3);
            EXPECT_LE(cvtest::norm(trueTvec, t1, NORM_INF), 1e-3);
            EXPECT_LE(cvtest::norm(trueRvec, R2, NORM_INF), 1e-3);
            EXPECT_LE(cvtest::norm(trueTvec, Mat_<double>(t2), NORM_INF), 1e-3);
        }
        {
            vector<Matx31f> R1s, t2s;
            vector<Matx31d> R2s, t1s;

            int res1 = solvePnPGeneric(points3dF, points2d, intrinsics, Mat(), R1s, t1s, false, (SolvePnPMethod)method);
            int res2 = solvePnPGeneric(points3d, points2dF, intrinsics, Mat(), R2s, t2s, false, (SolvePnPMethod)method);

            EXPECT_GT(res1, 0);
            EXPECT_GT(res2, 0);

            Matx31f R1 = R1s.front(), t2 = t2s.front();
            Matx31d R2 = R2s.front(), t1 = t1s.front();
            Matx31d R1d(R1(0), R1(1), R1(2)), t2d(t2(0), t2(1), t2(2));

            EXPECT_LE(cvtest::norm(R1d, R2, NORM_INF), 1e-3);
            EXPECT_LE(cvtest::norm(t1, t2d, NORM_INF), 1e-3);

            EXPECT_LE(cvtest::norm(trueRvec, R1d, NORM_INF), 1e-3);
            EXPECT_LE(cvtest::norm(trueTvec, t1, NORM_INF), 1e-3);
            EXPECT_LE(cvtest::norm(trueRvec, R2, NORM_INF), 1e-3);
            EXPECT_LE(cvtest::norm(trueTvec, t2d, NORM_INF), 1e-3);
        }

        if (method == SOLVEPNP_P3P || method == SOLVEPNP_AP3P)
        {
            //solveP3P
            {
                vector<Mat> Rs, ts, RFs, tFs;

                int res1 = solveP3P(points3dF, points2dF, Matx33f(intrinsics), Mat(), RFs, tFs, (SolvePnPMethod)method);
                int res2 = solveP3P(points3d, points2d, intrinsics, Mat(), Rs, ts, (SolvePnPMethod)method);

                EXPECT_GT(res1, 0);
                EXPECT_GT(res2, 0);

                Mat R = Rs.front(), t = ts.front(), RF = RFs.front(), tF = tFs.front();

                //By default rvecs and tvecs must be returned in double precision
                EXPECT_EQ(RF.type(), tF.type());
                EXPECT_EQ(RF.type(), CV_64FC1);

                EXPECT_EQ(R.type(), t.type());
                EXPECT_EQ(R.type(), CV_64FC1);

                EXPECT_LE(cvtest::norm(R, RF, NORM_INF), 1e-3);
                EXPECT_LE(cvtest::norm(t, tF, NORM_INF), 1e-3);

                EXPECT_LE(cvtest::norm(trueRvec, R, NORM_INF), 1e-3);
                EXPECT_LE(cvtest::norm(trueTvec, t, NORM_INF), 1e-3);
                EXPECT_LE(cvtest::norm(trueRvec, RF, NORM_INF), 1e-3);
                EXPECT_LE(cvtest::norm(trueTvec, tF, NORM_INF), 1e-3);
            }
            {
                vector<Mat> R1s, t1s, R2s, t2s;

                int res1 = solveP3P(points3dF, points2d, intrinsics, Mat(), R1s, t1s, (SolvePnPMethod)method);
                int res2 = solveP3P(points3d, points2dF, intrinsics, Mat(), R2s, t2s, (SolvePnPMethod)method);

                EXPECT_GT(res1, 0);
                EXPECT_GT(res2, 0);

                Mat R1 = R1s.front(), t1 = t1s.front(), R2 = R2s.front(), t2 = t2s.front();

                //By default rvecs and tvecs must be returned in double precision
                EXPECT_EQ(R1.type(), t1.type());
                EXPECT_EQ(R1.type(), CV_64FC1);

                EXPECT_EQ(R2.type(), t2.type());
                EXPECT_EQ(R2.type(), CV_64FC1);

                EXPECT_LE(cvtest::norm(R1, R2, NORM_INF), 1e-3);
                EXPECT_LE(cvtest::norm(t1, t2, NORM_INF), 1e-3);

                EXPECT_LE(cvtest::norm(trueRvec, R1, NORM_INF), 1e-3);
                EXPECT_LE(cvtest::norm(trueTvec, t1, NORM_INF), 1e-3);
                EXPECT_LE(cvtest::norm(trueRvec, R2, NORM_INF), 1e-3);
                EXPECT_LE(cvtest::norm(trueTvec, t2, NORM_INF), 1e-3);
            }
            {
                vector<Mat_<float> > R1s, t2s;
                vector<Mat_<double> > R2s, t1s;

                int res1 = solveP3P(points3dF, points2d, intrinsics, Mat(), R1s, t1s, (SolvePnPMethod)method);
                int res2 = solveP3P(points3d, points2dF, intrinsics, Mat(), R2s, t2s, (SolvePnPMethod)method);

                EXPECT_GT(res1, 0);
                EXPECT_GT(res2, 0);

                Mat R1 = R1s.front(), t1 = t1s.front();
                Mat R2 = R2s.front(), t2 = t2s.front();

                //If not null, rvecs and tvecs must be returned in the same precision
                EXPECT_EQ(R1.type(), CV_32FC1);
                EXPECT_EQ(t1.type(), CV_64FC1);

                EXPECT_EQ(R2.type(), CV_64FC1);
                EXPECT_EQ(t2.type(), CV_32FC1);

                EXPECT_LE(cvtest::norm(Mat_<double>(R1), R2, NORM_INF), 1e-3);
                EXPECT_LE(cvtest::norm(t1, Mat_<double>(t2), NORM_INF), 1e-3);

                EXPECT_LE(cvtest::norm(trueRvec, Mat_<double>(R1), NORM_INF), 1e-3);
                EXPECT_LE(cvtest::norm(trueTvec, t1, NORM_INF), 1e-3);
                EXPECT_LE(cvtest::norm(trueRvec, R2, NORM_INF), 1e-3);
                EXPECT_LE(cvtest::norm(trueTvec, Mat_<double>(t2), NORM_INF), 1e-3);
            }
            {
                vector<Matx31f> R1s, t2s;
                vector<Matx31d> R2s, t1s;

                int res1 = solveP3P(points3dF, points2d, intrinsics, Mat(), R1s, t1s, (SolvePnPMethod)method);
                int res2 = solveP3P(points3d, points2dF, intrinsics, Mat(), R2s, t2s, (SolvePnPMethod)method);

                EXPECT_GT(res1, 0);
                EXPECT_GT(res2, 0);

                Matx31f R1 = R1s.front(), t2 = t2s.front();
                Matx31d R2 = R2s.front(), t1 = t1s.front();
                Matx31d R1d(R1(0), R1(1), R1(2)), t2d(t2(0), t2(1), t2(2));

                EXPECT_LE(cvtest::norm(R1d, R2, NORM_INF), 1e-3);
                EXPECT_LE(cvtest::norm(t1, t2d, NORM_INF), 1e-3);

                EXPECT_LE(cvtest::norm(trueRvec, R1d, NORM_INF), 1e-3);
                EXPECT_LE(cvtest::norm(trueTvec, t1, NORM_INF), 1e-3);
                EXPECT_LE(cvtest::norm(trueRvec, R2, NORM_INF), 1e-3);
                EXPECT_LE(cvtest::norm(trueTvec, t2d, NORM_INF), 1e-3);
            }
        }
    }
}

TEST(Calib3d_SolvePnP, translation)
{
    Mat cameraIntrinsic = Mat::eye(3,3, CV_32FC1);
    vector<float> crvec;
    crvec.push_back(0.f);
    crvec.push_back(0.f);
    crvec.push_back(0.f);
    vector<float> ctvec;
    ctvec.push_back(100.f);
    ctvec.push_back(100.f);
    ctvec.push_back(0.f);
    vector<Point3f> p3d;
    p3d.push_back(Point3f(0,0,0));
    p3d.push_back(Point3f(0,0,10));
    p3d.push_back(Point3f(0,10,10));
    p3d.push_back(Point3f(10,10,10));
    p3d.push_back(Point3f(2,5,5));
    p3d.push_back(Point3f(-4,8,6));

    vector<Point2f> p2d;
    projectPoints(p3d, crvec, ctvec, cameraIntrinsic, noArray(), p2d);
    Mat rvec;
    Mat tvec;
    rvec =(Mat_<float>(3,1) << 0, 0, 0);
    tvec = (Mat_<float>(3,1) << 100, 100, 0);

    solvePnP(p3d, p2d, cameraIntrinsic, noArray(), rvec, tvec, true);
    EXPECT_TRUE(checkRange(rvec));
    EXPECT_TRUE(checkRange(tvec));

    rvec =(Mat_<double>(3,1) << 0, 0, 0);
    tvec = (Mat_<double>(3,1) << 100, 100, 0);
    solvePnP(p3d, p2d, cameraIntrinsic, noArray(), rvec, tvec, true);
    EXPECT_TRUE(checkRange(rvec));
    EXPECT_TRUE(checkRange(tvec));

    solvePnP(p3d, p2d, cameraIntrinsic, noArray(), rvec, tvec, false);
    EXPECT_TRUE(checkRange(rvec));
    EXPECT_TRUE(checkRange(tvec));
}

TEST(Calib3d_SolvePnP, iterativeInitialGuess3pts)
{
    {
        Matx33d intrinsics(605.4, 0.0, 317.35,
                           0.0, 601.2, 242.63,
                           0.0, 0.0, 1.0);

        double L = 0.1;
        vector<Point3d> p3d;
        p3d.push_back(Point3d(-L, -L, 0.0));
        p3d.push_back(Point3d(L, -L, 0.0));
        p3d.push_back(Point3d(L, L, 0.0));

        Mat rvec_ground_truth = (Mat_<double>(3,1) << 0.3, -0.2, 0.75);
        Mat tvec_ground_truth = (Mat_<double>(3,1) << 0.15, -0.2, 1.5);

        vector<Point2d> p2d;
        projectPoints(p3d, rvec_ground_truth, tvec_ground_truth, intrinsics, noArray(), p2d);

        Mat rvec_est = (Mat_<double>(3,1) << 0.2, -0.1, 0.6);
        Mat tvec_est = (Mat_<double>(3,1) << 0.05, -0.05, 1.0);

        solvePnP(p3d, p2d, intrinsics, noArray(), rvec_est, tvec_est, true, SOLVEPNP_ITERATIVE);

        cout << "rvec_ground_truth: " << rvec_ground_truth.t() << std::endl;
        cout << "rvec_est: " << rvec_est.t() << std::endl;
        cout << "tvec_ground_truth: " << tvec_ground_truth.t() << std::endl;
        cout << "tvec_est: " << tvec_est.t() << std::endl;

        EXPECT_LE(cvtest::norm(rvec_ground_truth, rvec_est, NORM_INF), 1e-6);
        EXPECT_LE(cvtest::norm(tvec_ground_truth, tvec_est, NORM_INF), 1e-6);

        EXPECT_EQ(rvec_est.type(), CV_64FC1);
        EXPECT_EQ(tvec_est.type(), CV_64FC1);
    }

    {
        Matx33f intrinsics(605.4f, 0.0f, 317.35f,
                           0.0f, 601.2f, 242.63f,
                           0.0f, 0.0f, 1.0f);

        float L = 0.1f;
        vector<Point3f> p3d;
        p3d.push_back(Point3f(-L, -L, 0.0f));
        p3d.push_back(Point3f(L, -L, 0.0f));
        p3d.push_back(Point3f(L, L, 0.0f));

        Mat rvec_ground_truth = (Mat_<float>(3,1) << -0.75f, 0.4f, 0.34f);
        Mat tvec_ground_truth = (Mat_<float>(3,1) << -0.15f, 0.35f, 1.58f);

        vector<Point2f> p2d;
        projectPoints(p3d, rvec_ground_truth, tvec_ground_truth, intrinsics, noArray(), p2d);

        Mat rvec_est = (Mat_<float>(3,1) << -0.5f, 0.2f, 0.2f);
        Mat tvec_est = (Mat_<float>(3,1) << 0.0f, 0.2f, 1.0f);

        solvePnP(p3d, p2d, intrinsics, noArray(), rvec_est, tvec_est, true, SOLVEPNP_ITERATIVE);

        cout << "rvec_ground_truth: " << rvec_ground_truth.t() << std::endl;
        cout << "rvec_est: " << rvec_est.t() << std::endl;
        cout << "tvec_ground_truth: " << tvec_ground_truth.t() << std::endl;
        cout << "tvec_est: " << tvec_est.t() << std::endl;

        EXPECT_LE(cvtest::norm(rvec_ground_truth, rvec_est, NORM_INF), 1e-6);
        EXPECT_LE(cvtest::norm(tvec_ground_truth, tvec_est, NORM_INF), 1e-6);

        EXPECT_EQ(rvec_est.type(), CV_32FC1);
        EXPECT_EQ(tvec_est.type(), CV_32FC1);
    }
}

TEST(Calib3d_SolvePnP, iterativeInitialGuess)
{
    {
        Matx33d intrinsics(605.4, 0.0, 317.35,
                           0.0, 601.2, 242.63,
                           0.0, 0.0, 1.0);

        double L = 0.1;
        vector<Point3d> p3d;
        p3d.push_back(Point3d(-L, -L, 0.0));
        p3d.push_back(Point3d(L, -L, 0.0));
        p3d.push_back(Point3d(L, L, 0.0));
        p3d.push_back(Point3d(-L, L, L/2));
        p3d.push_back(Point3d(0, 0, -L/2));

        Mat rvec_ground_truth = (Mat_<double>(3,1) << 0.3, -0.2, 0.75);
        Mat tvec_ground_truth = (Mat_<double>(3,1) << 0.15, -0.2, 1.5);

        vector<Point2d> p2d;
        projectPoints(p3d, rvec_ground_truth, tvec_ground_truth, intrinsics, noArray(), p2d);

        Mat rvec_est = (Mat_<double>(3,1) << 0.1, -0.1, 0.1);
        Mat tvec_est = (Mat_<double>(3,1) << 0.0, -0.5, 1.0);

        solvePnP(p3d, p2d, intrinsics, noArray(), rvec_est, tvec_est, true, SOLVEPNP_ITERATIVE);

        cout << "rvec_ground_truth: " << rvec_ground_truth.t() << std::endl;
        cout << "rvec_est: " << rvec_est.t() << std::endl;
        cout << "tvec_ground_truth: " << tvec_ground_truth.t() << std::endl;
        cout << "tvec_est: " << tvec_est.t() << std::endl;

        EXPECT_LE(cvtest::norm(rvec_ground_truth, rvec_est, NORM_INF), 1e-6);
        EXPECT_LE(cvtest::norm(tvec_ground_truth, tvec_est, NORM_INF), 1e-6);

        EXPECT_EQ(rvec_est.type(), CV_64FC1);
        EXPECT_EQ(tvec_est.type(), CV_64FC1);
    }

    {
        Matx33f intrinsics(605.4f, 0.0f, 317.35f,
                           0.0f, 601.2f, 242.63f,
                           0.0f, 0.0f, 1.0f);

        float L = 0.1f;
        vector<Point3f> p3d;
        p3d.push_back(Point3f(-L, -L, 0.0f));
        p3d.push_back(Point3f(L, -L, 0.0f));
        p3d.push_back(Point3f(L, L, 0.0f));
        p3d.push_back(Point3f(-L, L, L/2));
        p3d.push_back(Point3f(0, 0, -L/2));

        Mat rvec_ground_truth = (Mat_<float>(3,1) << -0.75f, 0.4f, 0.34f);
        Mat tvec_ground_truth = (Mat_<float>(3,1) << -0.15f, 0.35f, 1.58f);

        vector<Point2f> p2d;
        projectPoints(p3d, rvec_ground_truth, tvec_ground_truth, intrinsics, noArray(), p2d);

        Mat rvec_est = (Mat_<float>(3,1) << -0.1f, 0.1f, 0.1f);
        Mat tvec_est = (Mat_<float>(3,1) << 0.0f, 0.0f, 1.0f);

        solvePnP(p3d, p2d, intrinsics, noArray(), rvec_est, tvec_est, true, SOLVEPNP_ITERATIVE);

        cout << "rvec_ground_truth: " << rvec_ground_truth.t() << std::endl;
        cout << "rvec_est: " << rvec_est.t() << std::endl;
        cout << "tvec_ground_truth: " << tvec_ground_truth.t() << std::endl;
        cout << "tvec_est: " << tvec_est.t() << std::endl;

        EXPECT_LE(cvtest::norm(rvec_ground_truth, rvec_est, NORM_INF), 1e-6);
        EXPECT_LE(cvtest::norm(tvec_ground_truth, tvec_est, NORM_INF), 1e-6);

        EXPECT_EQ(rvec_est.type(), CV_32FC1);
        EXPECT_EQ(tvec_est.type(), CV_32FC1);
    }
}

TEST(Calib3d_SolvePnP, generic)
{
    {
        Matx33d intrinsics(605.4, 0.0, 317.35,
                           0.0, 601.2, 242.63,
                           0.0, 0.0, 1.0);

        double L = 0.1;
        vector<Point3d> p3d_;
        p3d_.push_back(Point3d(-L, L, 0));
        p3d_.push_back(Point3d(L, L, 0));
        p3d_.push_back(Point3d(L, -L, 0));
        p3d_.push_back(Point3d(-L, -L, 0));
        p3d_.push_back(Point3d(-L, L, L/2));
        p3d_.push_back(Point3d(0, 0, -L/2));

        const int ntests = 10;
        for (int numTest = 0; numTest < ntests; numTest++)
        {
            Mat rvec_ground_truth;
            Mat tvec_ground_truth;
            generatePose(p3d_, rvec_ground_truth, tvec_ground_truth, theRNG());

            vector<Point2d> p2d_;
            projectPoints(p3d_, rvec_ground_truth, tvec_ground_truth, intrinsics, noArray(), p2d_);

            for (int method = 0; method < SOLVEPNP_MAX_COUNT; method++)
            {
                vector<Mat> rvecs_est;
                vector<Mat> tvecs_est;

                vector<Point3d> p3d;
                vector<Point2d> p2d;
                if (method == SOLVEPNP_P3P || method == SOLVEPNP_AP3P ||
                    method == SOLVEPNP_IPPE || method == SOLVEPNP_IPPE_SQUARE)
                {
                    p3d = vector<Point3d>(p3d_.begin(), p3d_.begin()+4);
                    p2d = vector<Point2d>(p2d_.begin(), p2d_.begin()+4);
                }
                else
                {
                    p3d = p3d_;
                    p2d = p2d_;
                }

                vector<double> reprojectionErrors;
                solvePnPGeneric(p3d, p2d, intrinsics, noArray(), rvecs_est, tvecs_est, false, (SolvePnPMethod)method,
                                noArray(), noArray(), reprojectionErrors);

                EXPECT_TRUE(!rvecs_est.empty());
                EXPECT_TRUE(rvecs_est.size() == tvecs_est.size() && tvecs_est.size() == reprojectionErrors.size());

                for (size_t i = 0; i < reprojectionErrors.size()-1; i++)
                {
                    EXPECT_GE(reprojectionErrors[i+1], reprojectionErrors[i]);
                }

                bool isTestSuccess = false;
                for (size_t i = 0; i < rvecs_est.size() && !isTestSuccess; i++) {
                    double rvecDiff = cvtest::norm(rvecs_est[i], rvec_ground_truth, NORM_L2);
                    double tvecDiff = cvtest::norm(tvecs_est[i], tvec_ground_truth, NORM_L2);
                    const double threshold = method == SOLVEPNP_P3P ? 1e-2 : 1e-4;
                    isTestSuccess = rvecDiff < threshold && tvecDiff < threshold;
                }

                EXPECT_TRUE(isTestSuccess);
            }
        }
    }

    {
        Matx33f intrinsics(605.4f, 0.0f, 317.35f,
                           0.0f, 601.2f, 242.63f,
                           0.0f, 0.0f, 1.0f);

        float L = 0.1f;
        vector<Point3f> p3f_;
        p3f_.push_back(Point3f(-L, L, 0));
        p3f_.push_back(Point3f(L, L, 0));
        p3f_.push_back(Point3f(L, -L, 0));
        p3f_.push_back(Point3f(-L, -L, 0));
        p3f_.push_back(Point3f(-L, L, L/2));
        p3f_.push_back(Point3f(0, 0, -L/2));

        const int ntests = 10;
        for (int numTest = 0; numTest < ntests; numTest++)
        {
            Mat rvec_ground_truth;
            Mat tvec_ground_truth;
            generatePose(p3f_, rvec_ground_truth, tvec_ground_truth, theRNG());

            vector<Point2f> p2f_;
            projectPoints(p3f_, rvec_ground_truth, tvec_ground_truth, intrinsics, noArray(), p2f_);

            for (int method = 0; method < SOLVEPNP_MAX_COUNT; method++)
            {
                vector<Mat> rvecs_est;
                vector<Mat> tvecs_est;

                vector<Point3f> p3f;
                vector<Point2f> p2f;
                if (method == SOLVEPNP_P3P || method == SOLVEPNP_AP3P ||
                    method == SOLVEPNP_IPPE || method == SOLVEPNP_IPPE_SQUARE)
                {
                    p3f = vector<Point3f>(p3f_.begin(), p3f_.begin()+4);
                    p2f = vector<Point2f>(p2f_.begin(), p2f_.begin()+4);
                }
                else
                {
                    p3f = p3f_;
                    p2f = p2f_;
                }

                vector<double> reprojectionErrors;
                solvePnPGeneric(p3f, p2f, intrinsics, noArray(), rvecs_est, tvecs_est, false, (SolvePnPMethod)method,
                                noArray(), noArray(), reprojectionErrors);

                EXPECT_TRUE(!rvecs_est.empty());
                EXPECT_TRUE(rvecs_est.size() == tvecs_est.size() && tvecs_est.size() == reprojectionErrors.size());

                for (size_t i = 0; i < reprojectionErrors.size()-1; i++)
                {
                    EXPECT_GE(reprojectionErrors[i+1], reprojectionErrors[i]);
                }

                bool isTestSuccess = false;
                for (size_t i = 0; i < rvecs_est.size() && !isTestSuccess; i++) {
                    double rvecDiff = cvtest::norm(rvecs_est[i], rvec_ground_truth, NORM_L2);
                    double tvecDiff = cvtest::norm(tvecs_est[i], tvec_ground_truth, NORM_L2);
                    const double threshold = method == SOLVEPNP_P3P ? 1e-2 : 1e-4;
                    isTestSuccess = rvecDiff < threshold && tvecDiff < threshold;
                }

                EXPECT_TRUE(isTestSuccess);
            }
        }
    }
}

TEST(Calib3d_SolvePnP, refine3pts)
{
    {
        Matx33d intrinsics(605.4, 0.0, 317.35,
                           0.0, 601.2, 242.63,
                           0.0, 0.0, 1.0);

        double L = 0.1;
        vector<Point3d> p3d;
        p3d.push_back(Point3d(-L, -L, 0.0));
        p3d.push_back(Point3d(L, -L, 0.0));
        p3d.push_back(Point3d(L, L, 0.0));

        Mat rvec_ground_truth = (Mat_<double>(3,1) << 0.3, -0.2, 0.75);
        Mat tvec_ground_truth = (Mat_<double>(3,1) << 0.15, -0.2, 1.5);

        vector<Point2d> p2d;
        projectPoints(p3d, rvec_ground_truth, tvec_ground_truth, intrinsics, noArray(), p2d);

        {
            Mat rvec_est = (Mat_<double>(3,1) << 0.2, -0.1, 0.6);
            Mat tvec_est = (Mat_<double>(3,1) << 0.05, -0.05, 1.0);

            solvePnPRefineLM(p3d, p2d, intrinsics, noArray(), rvec_est, tvec_est);

            cout << "\nmethod: Levenberg-Marquardt" << endl;
            cout << "rvec_ground_truth: " << rvec_ground_truth.t() << std::endl;
            cout << "rvec_est: " << rvec_est.t() << std::endl;
            cout << "tvec_ground_truth: " << tvec_ground_truth.t() << std::endl;
            cout << "tvec_est: " << tvec_est.t() << std::endl;

            EXPECT_LE(cvtest::norm(rvec_ground_truth, rvec_est, NORM_INF), 1e-6);
            EXPECT_LE(cvtest::norm(tvec_ground_truth, tvec_est, NORM_INF), 1e-6);
        }
        {
            Mat rvec_est = (Mat_<double>(3,1) << 0.2, -0.1, 0.6);
            Mat tvec_est = (Mat_<double>(3,1) << 0.05, -0.05, 1.0);

            solvePnPRefineVVS(p3d, p2d, intrinsics, noArray(), rvec_est, tvec_est);

            cout << "\nmethod: Virtual Visual Servoing" << endl;
            cout << "rvec_ground_truth: " << rvec_ground_truth.t() << std::endl;
            cout << "rvec_est: " << rvec_est.t() << std::endl;
            cout << "tvec_ground_truth: " << tvec_ground_truth.t() << std::endl;
            cout << "tvec_est: " << tvec_est.t() << std::endl;

            EXPECT_LE(cvtest::norm(rvec_ground_truth, rvec_est, NORM_INF), 1e-6);
            EXPECT_LE(cvtest::norm(tvec_ground_truth, tvec_est, NORM_INF), 1e-6);
        }
    }

    {
        Matx33f intrinsics(605.4f, 0.0f, 317.35f,
                           0.0f, 601.2f, 242.63f,
                           0.0f, 0.0f, 1.0f);

        float L = 0.1f;
        vector<Point3f> p3d;
        p3d.push_back(Point3f(-L, -L, 0.0f));
        p3d.push_back(Point3f(L, -L, 0.0f));
        p3d.push_back(Point3f(L, L, 0.0f));

        Mat rvec_ground_truth = (Mat_<float>(3,1) << -0.75f, 0.4f, 0.34f);
        Mat tvec_ground_truth = (Mat_<float>(3,1) << -0.15f, 0.35f, 1.58f);

        vector<Point2f> p2d;
        projectPoints(p3d, rvec_ground_truth, tvec_ground_truth, intrinsics, noArray(), p2d);

        {
            Mat rvec_est = (Mat_<float>(3,1) << -0.5f, 0.2f, 0.2f);
            Mat tvec_est = (Mat_<float>(3,1) << 0.0f, 0.2f, 1.0f);

            solvePnPRefineLM(p3d, p2d, intrinsics, noArray(), rvec_est, tvec_est);

            cout << "\nmethod: Levenberg-Marquardt" << endl;
            cout << "rvec_ground_truth: " << rvec_ground_truth.t() << std::endl;
            cout << "rvec_est: " << rvec_est.t() << std::endl;
            cout << "tvec_ground_truth: " << tvec_ground_truth.t() << std::endl;
            cout << "tvec_est: " << tvec_est.t() << std::endl;

            EXPECT_LE(cvtest::norm(rvec_ground_truth, rvec_est, NORM_INF), 1e-6);
            EXPECT_LE(cvtest::norm(tvec_ground_truth, tvec_est, NORM_INF), 1e-6);
        }
        {
            Mat rvec_est = (Mat_<float>(3,1) << -0.5f, 0.2f, 0.2f);
            Mat tvec_est = (Mat_<float>(3,1) << 0.0f, 0.2f, 1.0f);

            solvePnPRefineVVS(p3d, p2d, intrinsics, noArray(), rvec_est, tvec_est);

            cout << "\nmethod: Virtual Visual Servoing" << endl;
            cout << "rvec_ground_truth: " << rvec_ground_truth.t() << std::endl;
            cout << "rvec_est: " << rvec_est.t() << std::endl;
            cout << "tvec_ground_truth: " << tvec_ground_truth.t() << std::endl;
            cout << "tvec_est: " << tvec_est.t() << std::endl;

            EXPECT_LE(cvtest::norm(rvec_ground_truth, rvec_est, NORM_INF), 1e-6);
            EXPECT_LE(cvtest::norm(tvec_ground_truth, tvec_est, NORM_INF), 1e-6);
        }
    }
}

TEST(Calib3d_SolvePnP, refine)
{
    //double
    {
        Matx33d intrinsics(605.4, 0.0, 317.35,
                           0.0, 601.2, 242.63,
                           0.0, 0.0, 1.0);

        double L = 0.1;
        vector<Point3d> p3d;
        p3d.push_back(Point3d(-L, -L, 0.0));
        p3d.push_back(Point3d(L, -L, 0.0));
        p3d.push_back(Point3d(L, L, 0.0));
        p3d.push_back(Point3d(-L, L, L/2));
        p3d.push_back(Point3d(0, 0, -L/2));

        Mat rvec_ground_truth = (Mat_<double>(3,1) << 0.3, -0.2, 0.75);
        Mat tvec_ground_truth = (Mat_<double>(3,1) << 0.15, -0.2, 1.5);

        vector<Point2d> p2d;
        projectPoints(p3d, rvec_ground_truth, tvec_ground_truth, intrinsics, noArray(), p2d);

        {
            Mat rvec_est = (Mat_<double>(3,1) << 0.1, -0.1, 0.1);
            Mat tvec_est = (Mat_<double>(3,1) << 0.0, -0.5, 1.0);

            solvePnP(p3d, p2d, intrinsics, noArray(), rvec_est, tvec_est, true, SOLVEPNP_ITERATIVE);

            cout << "\nmethod: Levenberg-Marquardt (C API)" << endl;
            cout << "rvec_ground_truth: " << rvec_ground_truth.t() << std::endl;
            cout << "rvec_est: " << rvec_est.t() << std::endl;
            cout << "tvec_ground_truth: " << tvec_ground_truth.t() << std::endl;
            cout << "tvec_est: " << tvec_est.t() << std::endl;

            EXPECT_LE(cvtest::norm(rvec_ground_truth, rvec_est, NORM_INF), 1e-6);
            EXPECT_LE(cvtest::norm(tvec_ground_truth, tvec_est, NORM_INF), 1e-6);
        }
        {
            Mat rvec_est = (Mat_<double>(3,1) << 0.1, -0.1, 0.1);
            Mat tvec_est = (Mat_<double>(3,1) << 0.0, -0.5, 1.0);

            solvePnPRefineLM(p3d, p2d, intrinsics, noArray(), rvec_est, tvec_est);

            cout << "\nmethod: Levenberg-Marquardt (C++ API)" << endl;
            cout << "rvec_ground_truth: " << rvec_ground_truth.t() << std::endl;
            cout << "rvec_est: " << rvec_est.t() << std::endl;
            cout << "tvec_ground_truth: " << tvec_ground_truth.t() << std::endl;
            cout << "tvec_est: " << tvec_est.t() << std::endl;

            EXPECT_LE(cvtest::norm(rvec_ground_truth, rvec_est, NORM_INF), 1e-6);
            EXPECT_LE(cvtest::norm(tvec_ground_truth, tvec_est, NORM_INF), 1e-6);
        }
        {
            Mat rvec_est = (Mat_<double>(3,1) << 0.1, -0.1, 0.1);
            Mat tvec_est = (Mat_<double>(3,1) << 0.0, -0.5, 1.0);

            solvePnPRefineVVS(p3d, p2d, intrinsics, noArray(), rvec_est, tvec_est);

            cout << "\nmethod: Virtual Visual Servoing" << endl;
            cout << "rvec_ground_truth: " << rvec_ground_truth.t() << std::endl;
            cout << "rvec_est: " << rvec_est.t() << std::endl;
            cout << "tvec_ground_truth: " << tvec_ground_truth.t() << std::endl;
            cout << "tvec_est: " << tvec_est.t() << std::endl;

            EXPECT_LE(cvtest::norm(rvec_ground_truth, rvec_est, NORM_INF), 1e-6);
            EXPECT_LE(cvtest::norm(tvec_ground_truth, tvec_est, NORM_INF), 1e-6);
        }
    }

    //float
    {
        Matx33f intrinsics(605.4f, 0.0f, 317.35f,
                           0.0f, 601.2f, 242.63f,
                           0.0f, 0.0f, 1.0f);

        float L = 0.1f;
        vector<Point3f> p3d;
        p3d.push_back(Point3f(-L, -L, 0.0f));
        p3d.push_back(Point3f(L, -L, 0.0f));
        p3d.push_back(Point3f(L, L, 0.0f));
        p3d.push_back(Point3f(-L, L, L/2));
        p3d.push_back(Point3f(0, 0, -L/2));

        Mat rvec_ground_truth = (Mat_<float>(3,1) << -0.75f, 0.4f, 0.34f);
        Mat tvec_ground_truth = (Mat_<float>(3,1) << -0.15f, 0.35f, 1.58f);

        vector<Point2f> p2d;
        projectPoints(p3d, rvec_ground_truth, tvec_ground_truth, intrinsics, noArray(), p2d);

        {
            Mat rvec_est = (Mat_<float>(3,1) << -0.1f, 0.1f, 0.1f);
            Mat tvec_est = (Mat_<float>(3,1) << 0.0f, 0.0f, 1.0f);

            solvePnP(p3d, p2d, intrinsics, noArray(), rvec_est, tvec_est, true, SOLVEPNP_ITERATIVE);

            cout << "\nmethod: Levenberg-Marquardt (C API)" << endl;
            cout << "rvec_ground_truth: " << rvec_ground_truth.t() << std::endl;
            cout << "rvec_est: " << rvec_est.t() << std::endl;
            cout << "tvec_ground_truth: " << tvec_ground_truth.t() << std::endl;
            cout << "tvec_est: " << tvec_est.t() << std::endl;

            EXPECT_LE(cvtest::norm(rvec_ground_truth, rvec_est, NORM_INF), 1e-6);
            EXPECT_LE(cvtest::norm(tvec_ground_truth, tvec_est, NORM_INF), 1e-6);
        }
        {
            Mat rvec_est = (Mat_<float>(3,1) << -0.1f, 0.1f, 0.1f);
            Mat tvec_est = (Mat_<float>(3,1) << 0.0f, 0.0f, 1.0f);

            solvePnPRefineLM(p3d, p2d, intrinsics, noArray(), rvec_est, tvec_est);

            cout << "\nmethod: Levenberg-Marquardt (C++ API)" << endl;
            cout << "rvec_ground_truth: " << rvec_ground_truth.t() << std::endl;
            cout << "rvec_est: " << rvec_est.t() << std::endl;
            cout << "tvec_ground_truth: " << tvec_ground_truth.t() << std::endl;
            cout << "tvec_est: " << tvec_est.t() << std::endl;

            EXPECT_LE(cvtest::norm(rvec_ground_truth, rvec_est, NORM_INF), 1e-6);
            EXPECT_LE(cvtest::norm(tvec_ground_truth, tvec_est, NORM_INF), 1e-6);
        }
        {
            Mat rvec_est = (Mat_<float>(3,1) << -0.1f, 0.1f, 0.1f);
            Mat tvec_est = (Mat_<float>(3,1) << 0.0f, 0.0f, 1.0f);

            solvePnPRefineVVS(p3d, p2d, intrinsics, noArray(), rvec_est, tvec_est);

            cout << "\nmethod: Virtual Visual Servoing" << endl;
            cout << "rvec_ground_truth: " << rvec_ground_truth.t() << std::endl;
            cout << "rvec_est: " << rvec_est.t() << std::endl;
            cout << "tvec_ground_truth: " << tvec_ground_truth.t() << std::endl;
            cout << "tvec_est: " << tvec_est.t() << std::endl;

            EXPECT_LE(cvtest::norm(rvec_ground_truth, rvec_est, NORM_INF), 1e-6);
            EXPECT_LE(cvtest::norm(tvec_ground_truth, tvec_est, NORM_INF), 1e-6);
        }
    }

    //refine after solvePnP
    {
        Matx33d intrinsics(605.4, 0.0, 317.35,
                           0.0, 601.2, 242.63,
                           0.0, 0.0, 1.0);

        double L = 0.1;
        vector<Point3d> p3d;
        p3d.push_back(Point3d(-L, -L, 0.0));
        p3d.push_back(Point3d(L, -L, 0.0));
        p3d.push_back(Point3d(L, L, 0.0));
        p3d.push_back(Point3d(-L, L, L/2));
        p3d.push_back(Point3d(0, 0, -L/2));

        Mat rvec_ground_truth = (Mat_<double>(3,1) << 0.3, -0.2, 0.75);
        Mat tvec_ground_truth = (Mat_<double>(3,1) << 0.15, -0.2, 1.5);

        vector<Point2d> p2d;
        projectPoints(p3d, rvec_ground_truth, tvec_ground_truth, intrinsics, noArray(), p2d);

        //add small Gaussian noise
        RNG& rng = theRNG();
        for (size_t i = 0; i < p2d.size(); i++)
        {
            p2d[i].x += rng.gaussian(5e-2);
            p2d[i].y += rng.gaussian(5e-2);
        }

        Mat rvec_est, tvec_est;
        solvePnP(p3d, p2d, intrinsics, noArray(), rvec_est, tvec_est, false, SOLVEPNP_EPNP);

        {

            Mat rvec_est_refine = rvec_est.clone(), tvec_est_refine = tvec_est.clone();
            solvePnP(p3d, p2d, intrinsics, noArray(), rvec_est_refine, tvec_est_refine, true, SOLVEPNP_ITERATIVE);

            cout << "\nmethod: Levenberg-Marquardt (C API)" << endl;
            cout << "rvec_ground_truth: " << rvec_ground_truth.t() << std::endl;
            cout << "rvec_est (EPnP): " << rvec_est.t() << std::endl;
            cout << "rvec_est_refine: " << rvec_est_refine.t() << std::endl;
            cout << "tvec_ground_truth: " << tvec_ground_truth.t() << std::endl;
            cout << "tvec_est (EPnP): " << tvec_est.t() << std::endl;
            cout << "tvec_est_refine: " << tvec_est_refine.t() << std::endl;

            EXPECT_LE(cvtest::norm(rvec_ground_truth, rvec_est, NORM_INF), 1e-2);
            EXPECT_LE(cvtest::norm(tvec_ground_truth, tvec_est, NORM_INF), 1e-3);

            EXPECT_LT(cvtest::norm(rvec_ground_truth, rvec_est_refine, NORM_INF), cvtest::norm(rvec_ground_truth, rvec_est, NORM_INF));
            EXPECT_LT(cvtest::norm(tvec_ground_truth, tvec_est_refine, NORM_INF), cvtest::norm(tvec_ground_truth, tvec_est, NORM_INF));
        }
        {
            Mat rvec_est_refine = rvec_est.clone(), tvec_est_refine = tvec_est.clone();
            solvePnPRefineLM(p3d, p2d, intrinsics, noArray(), rvec_est_refine, tvec_est_refine);

            cout << "\nmethod: Levenberg-Marquardt (C++ API)" << endl;
            cout << "rvec_ground_truth: " << rvec_ground_truth.t() << std::endl;
            cout << "rvec_est: " << rvec_est.t() << std::endl;
            cout << "rvec_est_refine: " << rvec_est_refine.t() << std::endl;
            cout << "tvec_ground_truth: " << tvec_ground_truth.t() << std::endl;
            cout << "tvec_est: " << tvec_est.t() << std::endl;
            cout << "tvec_est_refine: " << tvec_est_refine.t() << std::endl;

            EXPECT_LE(cvtest::norm(rvec_ground_truth, rvec_est, NORM_INF), 1e-2);
            EXPECT_LE(cvtest::norm(tvec_ground_truth, tvec_est, NORM_INF), 1e-3);

            EXPECT_LT(cvtest::norm(rvec_ground_truth, rvec_est_refine, NORM_INF), cvtest::norm(rvec_ground_truth, rvec_est, NORM_INF));
            EXPECT_LT(cvtest::norm(tvec_ground_truth, tvec_est_refine, NORM_INF), cvtest::norm(tvec_ground_truth, tvec_est, NORM_INF));
        }
        {
            Mat rvec_est_refine = rvec_est.clone(), tvec_est_refine = tvec_est.clone();
            solvePnPRefineVVS(p3d, p2d, intrinsics, noArray(), rvec_est_refine, tvec_est_refine);

            cout << "\nmethod: Virtual Visual Servoing" << endl;
            cout << "rvec_ground_truth: " << rvec_ground_truth.t() << std::endl;
            cout << "rvec_est: " << rvec_est.t() << std::endl;
            cout << "rvec_est_refine: " << rvec_est_refine.t() << std::endl;
            cout << "tvec_ground_truth: " << tvec_ground_truth.t() << std::endl;
            cout << "tvec_est: " << tvec_est.t() << std::endl;
            cout << "tvec_est_refine: " << tvec_est_refine.t() << std::endl;

            EXPECT_LE(cvtest::norm(rvec_ground_truth, rvec_est, NORM_INF), 1e-2);
            EXPECT_LE(cvtest::norm(tvec_ground_truth, tvec_est, NORM_INF), 1e-3);

            EXPECT_LT(cvtest::norm(rvec_ground_truth, rvec_est_refine, NORM_INF), cvtest::norm(rvec_ground_truth, rvec_est, NORM_INF));
            EXPECT_LT(cvtest::norm(tvec_ground_truth, tvec_est_refine, NORM_INF), cvtest::norm(tvec_ground_truth, tvec_est, NORM_INF));
        }
    }
}

TEST(Calib3d_SolvePnPRansac, minPoints)
{
    //https://github.com/opencv/opencv/issues/14423
    Mat matK = Mat::eye(3,3,CV_64FC1);
    Mat distCoeff = Mat::zeros(1,5,CV_64FC1);
    Matx31d true_rvec(0.9072420896651262, 0.09226497171882152, 0.8880772883671504);
    Matx31d true_tvec(7.376333362427632, 8.434449036856979, 13.79801619778456);

    {
        //nb points = 5 --> ransac_kernel_method = SOLVEPNP_EPNP
        Mat keypoints13D = (Mat_<float>(5, 3) << 12.00604, -2.8654366, 18.472504,
                                                 7.6863389, 4.9355154, 11.146358,
                                                 14.260933, 2.8320458, 12.582781,
                                                 3.4562225, 8.2668982, 11.300434,
                                                 15.316854, 3.7486348, 12.491116);
        vector<Point2f> imagesPoints;
        projectPoints(keypoints13D, true_rvec, true_tvec, matK, distCoeff, imagesPoints);

        Mat keypoints22D(keypoints13D.rows, 2, CV_32FC1);
        vector<Point3f> objectPoints;
        for (int i = 0; i < static_cast<int>(imagesPoints.size()); i++)
        {
            keypoints22D.at<float>(i,0) = imagesPoints[i].x;
            keypoints22D.at<float>(i,1) = imagesPoints[i].y;
            objectPoints.push_back(Point3f(keypoints13D.at<float>(i,0), keypoints13D.at<float>(i,1), keypoints13D.at<float>(i,2)));
        }

        Mat rvec = Mat::zeros(1,3,CV_64FC1);
        Mat Tvec = Mat::zeros(1,3,CV_64FC1);
        solvePnPRansac(keypoints13D, keypoints22D, matK, distCoeff, rvec, Tvec);

        Mat rvec2, Tvec2;
        solvePnP(objectPoints, imagesPoints, matK, distCoeff, rvec2, Tvec2, false, SOLVEPNP_EPNP);

        EXPECT_LE(cvtest::norm(true_rvec, rvec, NORM_INF), 1e-4);
        EXPECT_LE(cvtest::norm(true_tvec, Tvec, NORM_INF), 1e-4);
        EXPECT_LE(cvtest::norm(rvec, rvec2, NORM_INF), 1e-6);
        EXPECT_LE(cvtest::norm(Tvec, Tvec2, NORM_INF), 1e-6);
    }
    {
        //nb points = 4 --> ransac_kernel_method = SOLVEPNP_P3P
        Mat keypoints13D = (Mat_<float>(4, 3) << 12.00604, -2.8654366, 18.472504,
                                                 7.6863389, 4.9355154, 11.146358,
                                                 14.260933, 2.8320458, 12.582781,
                                                 3.4562225, 8.2668982, 11.300434);
        vector<Point2f> imagesPoints;
        projectPoints(keypoints13D, true_rvec, true_tvec, matK, distCoeff, imagesPoints);

        Mat keypoints22D(keypoints13D.rows, 2, CV_32FC1);
        vector<Point3f> objectPoints;
        for (int i = 0; i < static_cast<int>(imagesPoints.size()); i++)
        {
            keypoints22D.at<float>(i,0) = imagesPoints[i].x;
            keypoints22D.at<float>(i,1) = imagesPoints[i].y;
            objectPoints.push_back(Point3f(keypoints13D.at<float>(i,0), keypoints13D.at<float>(i,1), keypoints13D.at<float>(i,2)));
        }

        Mat rvec = Mat::zeros(1,3,CV_64FC1);
        Mat Tvec = Mat::zeros(1,3,CV_64FC1);
        solvePnPRansac(keypoints13D, keypoints22D, matK, distCoeff, rvec, Tvec);

        Mat rvec2, Tvec2;
        solvePnP(objectPoints, imagesPoints, matK, distCoeff, rvec2, Tvec2, false, SOLVEPNP_P3P);

        EXPECT_LE(cvtest::norm(true_rvec, rvec, NORM_INF), 1e-4);
        EXPECT_LE(cvtest::norm(true_tvec, Tvec, NORM_INF), 1e-4);
        EXPECT_LE(cvtest::norm(rvec, rvec2, NORM_INF), 1e-6);
        EXPECT_LE(cvtest::norm(Tvec, Tvec2, NORM_INF), 1e-6);
    }
}

TEST(Calib3d_SolvePnPRansac, inputShape)
{
    //https://github.com/opencv/opencv/issues/14423
    Mat matK = Mat::eye(3,3,CV_64FC1);
    Mat distCoeff = Mat::zeros(1,5,CV_64FC1);
    Matx31d true_rvec(0.9072420896651262, 0.09226497171882152, 0.8880772883671504);
    Matx31d true_tvec(7.376333362427632, 8.434449036856979, 13.79801619778456);

    {
        //Nx3 1-channel
        Mat keypoints13D = (Mat_<float>(6, 3) << 12.00604, -2.8654366, 18.472504,
                                                 7.6863389, 4.9355154, 11.146358,
                                                 14.260933, 2.8320458, 12.582781,
                                                 3.4562225, 8.2668982, 11.300434,
                                                 10.00604,  2.8654366, 15.472504,
                                                 -4.6863389, 5.9355154, 13.146358);
        vector<Point2f> imagesPoints;
        projectPoints(keypoints13D, true_rvec, true_tvec, matK, distCoeff, imagesPoints);

        Mat keypoints22D(keypoints13D.rows, 2, CV_32FC1);
        for (int i = 0; i < static_cast<int>(imagesPoints.size()); i++)
        {
            keypoints22D.at<float>(i,0) = imagesPoints[i].x;
            keypoints22D.at<float>(i,1) = imagesPoints[i].y;
        }

        Mat rvec, Tvec;
        solvePnPRansac(keypoints13D, keypoints22D, matK, distCoeff, rvec, Tvec);

        EXPECT_LE(cvtest::norm(true_rvec, rvec, NORM_INF), 1e-6);
        EXPECT_LE(cvtest::norm(true_tvec, Tvec, NORM_INF), 1e-6);
    }
    {
        //1xN 3-channel
        Mat keypoints13D(1, 6, CV_32FC3);
        keypoints13D.at<Vec3f>(0,0) = Vec3f(12.00604f, -2.8654366f, 18.472504f);
        keypoints13D.at<Vec3f>(0,1) = Vec3f(7.6863389f, 4.9355154f, 11.146358f);
        keypoints13D.at<Vec3f>(0,2) = Vec3f(14.260933f, 2.8320458f, 12.582781f);
        keypoints13D.at<Vec3f>(0,3) = Vec3f(3.4562225f, 8.2668982f, 11.300434f);
        keypoints13D.at<Vec3f>(0,4) = Vec3f(10.00604f,  2.8654366f, 15.472504f);
        keypoints13D.at<Vec3f>(0,5) = Vec3f(-4.6863389f, 5.9355154f, 13.146358f);

        vector<Point2f> imagesPoints;
        projectPoints(keypoints13D, true_rvec, true_tvec, matK, distCoeff, imagesPoints);

        Mat keypoints22D(keypoints13D.rows, keypoints13D.cols, CV_32FC2);
        for (int i = 0; i < static_cast<int>(imagesPoints.size()); i++)
        {
            keypoints22D.at<Vec2f>(0,i) = Vec2f(imagesPoints[i].x, imagesPoints[i].y);
        }

        Mat rvec, Tvec;
        solvePnPRansac(keypoints13D, keypoints22D, matK, distCoeff, rvec, Tvec);

        EXPECT_LE(cvtest::norm(true_rvec, rvec, NORM_INF), 1e-6);
        EXPECT_LE(cvtest::norm(true_tvec, Tvec, NORM_INF), 1e-6);
    }
    {
        //Nx1 3-channel
        Mat keypoints13D(6, 1, CV_32FC3);
        keypoints13D.at<Vec3f>(0,0) = Vec3f(12.00604f, -2.8654366f, 18.472504f);
        keypoints13D.at<Vec3f>(1,0) = Vec3f(7.6863389f, 4.9355154f, 11.146358f);
        keypoints13D.at<Vec3f>(2,0) = Vec3f(14.260933f, 2.8320458f, 12.582781f);
        keypoints13D.at<Vec3f>(3,0) = Vec3f(3.4562225f, 8.2668982f, 11.300434f);
        keypoints13D.at<Vec3f>(4,0) = Vec3f(10.00604f,  2.8654366f, 15.472504f);
        keypoints13D.at<Vec3f>(5,0) = Vec3f(-4.6863389f, 5.9355154f, 13.146358f);

        vector<Point2f> imagesPoints;
        projectPoints(keypoints13D, true_rvec, true_tvec, matK, distCoeff, imagesPoints);

        Mat keypoints22D(keypoints13D.rows, keypoints13D.cols, CV_32FC2);
        for (int i = 0; i < static_cast<int>(imagesPoints.size()); i++)
        {
            keypoints22D.at<Vec2f>(i,0) = Vec2f(imagesPoints[i].x, imagesPoints[i].y);
        }

        Mat rvec, Tvec;
        solvePnPRansac(keypoints13D, keypoints22D, matK, distCoeff, rvec, Tvec);

        EXPECT_LE(cvtest::norm(true_rvec, rvec, NORM_INF), 1e-6);
        EXPECT_LE(cvtest::norm(true_tvec, Tvec, NORM_INF), 1e-6);
    }
    {
        //vector<Point3f>
        vector<Point3f> keypoints13D;
        keypoints13D.push_back(Point3f(12.00604f, -2.8654366f, 18.472504f));
        keypoints13D.push_back(Point3f(7.6863389f, 4.9355154f, 11.146358f));
        keypoints13D.push_back(Point3f(14.260933f, 2.8320458f, 12.582781f));
        keypoints13D.push_back(Point3f(3.4562225f, 8.2668982f, 11.300434f));
        keypoints13D.push_back(Point3f(10.00604f,  2.8654366f, 15.472504f));
        keypoints13D.push_back(Point3f(-4.6863389f, 5.9355154f, 13.146358f));

        vector<Point2f> keypoints22D;
        projectPoints(keypoints13D, true_rvec, true_tvec, matK, distCoeff, keypoints22D);

        Mat rvec, Tvec;
        solvePnPRansac(keypoints13D, keypoints22D, matK, distCoeff, rvec, Tvec);

        EXPECT_LE(cvtest::norm(true_rvec, rvec, NORM_INF), 1e-6);
        EXPECT_LE(cvtest::norm(true_tvec, Tvec, NORM_INF), 1e-6);
    }
    {
        //vector<Point3d>
        vector<Point3d> keypoints13D;
        keypoints13D.push_back(Point3d(12.00604f, -2.8654366f, 18.472504f));
        keypoints13D.push_back(Point3d(7.6863389f, 4.9355154f, 11.146358f));
        keypoints13D.push_back(Point3d(14.260933f, 2.8320458f, 12.582781f));
        keypoints13D.push_back(Point3d(3.4562225f, 8.2668982f, 11.300434f));
        keypoints13D.push_back(Point3d(10.00604f,  2.8654366f, 15.472504f));
        keypoints13D.push_back(Point3d(-4.6863389f, 5.9355154f, 13.146358f));

        vector<Point2d> keypoints22D;
        projectPoints(keypoints13D, true_rvec, true_tvec, matK, distCoeff, keypoints22D);

        Mat rvec, Tvec;
        solvePnPRansac(keypoints13D, keypoints22D, matK, distCoeff, rvec, Tvec);

        EXPECT_LE(cvtest::norm(true_rvec, rvec, NORM_INF), 1e-6);
        EXPECT_LE(cvtest::norm(true_tvec, Tvec, NORM_INF), 1e-6);
    }
}

TEST(Calib3d_SolvePnP, inputShape)
{
    //https://github.com/opencv/opencv/issues/14423
    Mat matK = Mat::eye(3,3,CV_64FC1);
    Mat distCoeff = Mat::zeros(1,5,CV_64FC1);
    Matx31d true_rvec(0.407, 0.092, 0.88);
    Matx31d true_tvec(0.576, -0.43, 1.3798);

    vector<Point3d> objectPoints;
    const double L = 0.5;
    objectPoints.push_back(Point3d(-L, -L,  L));
    objectPoints.push_back(Point3d( L, -L,  L));
    objectPoints.push_back(Point3d( L,  L,  L));
    objectPoints.push_back(Point3d(-L,  L,  L));
    objectPoints.push_back(Point3d(-L, -L, -L));
    objectPoints.push_back(Point3d( L, -L, -L));

    const int methodsCount = 6;
    int methods[] = {SOLVEPNP_ITERATIVE, SOLVEPNP_EPNP, SOLVEPNP_P3P, SOLVEPNP_AP3P, SOLVEPNP_IPPE, SOLVEPNP_IPPE_SQUARE};
    for (int method = 0; method < methodsCount; method++)
    {
        if (methods[method] == SOLVEPNP_IPPE_SQUARE)
        {
            objectPoints[0] = Point3d(-L,  L,  0);
            objectPoints[1] = Point3d( L,  L,  0);
            objectPoints[2] = Point3d( L, -L,  0);
            objectPoints[3] = Point3d(-L, -L,  0);
        }

        {
            //Nx3 1-channel
            Mat keypoints13D;
            if (methods[method] == SOLVEPNP_P3P || methods[method] == SOLVEPNP_AP3P ||
                methods[method] == SOLVEPNP_IPPE || methods[method] == SOLVEPNP_IPPE_SQUARE)
            {
                keypoints13D = Mat(4, 3, CV_32FC1);
            }
            else
            {
                keypoints13D = Mat(6, 3, CV_32FC1);
            }

            for (int i = 0; i < keypoints13D.rows; i++)
            {
                keypoints13D.at<float>(i,0) = static_cast<float>(objectPoints[i].x);
                keypoints13D.at<float>(i,1) = static_cast<float>(objectPoints[i].y);
                keypoints13D.at<float>(i,2) = static_cast<float>(objectPoints[i].z);
            }

            vector<Point2f> imagesPoints;
            projectPoints(keypoints13D, true_rvec, true_tvec, matK, distCoeff, imagesPoints);

            Mat keypoints22D(keypoints13D.rows, 2, CV_32FC1);
            for (int i = 0; i < static_cast<int>(imagesPoints.size()); i++)
            {
                keypoints22D.at<float>(i,0) = imagesPoints[i].x;
                keypoints22D.at<float>(i,1) = imagesPoints[i].y;
            }

            Mat rvec, Tvec;
            solvePnP(keypoints13D, keypoints22D, matK, distCoeff, rvec, Tvec, false, methods[method]);

            EXPECT_LE(cvtest::norm(true_rvec, rvec, NORM_INF), 1e-3);
            EXPECT_LE(cvtest::norm(true_tvec, Tvec, NORM_INF), 1e-3);
        }
        {
            //1xN 3-channel
            Mat keypoints13D;
            if (methods[method] == SOLVEPNP_P3P || methods[method] == SOLVEPNP_AP3P ||
                methods[method] == SOLVEPNP_IPPE || methods[method] == SOLVEPNP_IPPE_SQUARE)
            {
                keypoints13D = Mat(1, 4, CV_32FC3);
            }
            else
            {
                keypoints13D = Mat(1, 6, CV_32FC3);
            }

            for (int i = 0; i < keypoints13D.cols; i++)
            {
                keypoints13D.at<Vec3f>(0,i) = Vec3f(static_cast<float>(objectPoints[i].x),
                                                    static_cast<float>(objectPoints[i].y),
                                                    static_cast<float>(objectPoints[i].z));
            }

            vector<Point2f> imagesPoints;
            projectPoints(keypoints13D, true_rvec, true_tvec, matK, distCoeff, imagesPoints);

            Mat keypoints22D(keypoints13D.rows, keypoints13D.cols, CV_32FC2);
            for (int i = 0; i < static_cast<int>(imagesPoints.size()); i++)
            {
                keypoints22D.at<Vec2f>(0,i) = Vec2f(imagesPoints[i].x, imagesPoints[i].y);
            }

            Mat rvec, Tvec;
            solvePnP(keypoints13D, keypoints22D, matK, distCoeff, rvec, Tvec, false, methods[method]);

            EXPECT_LE(cvtest::norm(true_rvec, rvec, NORM_INF), 1e-3);
            EXPECT_LE(cvtest::norm(true_tvec, Tvec, NORM_INF), 1e-3);
        }
        {
            //Nx1 3-channel
            Mat keypoints13D;
            if (methods[method] == SOLVEPNP_P3P || methods[method] == SOLVEPNP_AP3P ||
                methods[method] == SOLVEPNP_IPPE || methods[method] == SOLVEPNP_IPPE_SQUARE)
            {
                keypoints13D = Mat(4, 1, CV_32FC3);
            }
            else
            {
                keypoints13D = Mat(6, 1, CV_32FC3);
            }

            for (int i = 0; i < keypoints13D.rows; i++)
            {
                keypoints13D.at<Vec3f>(i,0) = Vec3f(static_cast<float>(objectPoints[i].x),
                                                    static_cast<float>(objectPoints[i].y),
                                                    static_cast<float>(objectPoints[i].z));
            }

            vector<Point2f> imagesPoints;
            projectPoints(keypoints13D, true_rvec, true_tvec, matK, distCoeff, imagesPoints);

            Mat keypoints22D(keypoints13D.rows, keypoints13D.cols, CV_32FC2);
            for (int i = 0; i < static_cast<int>(imagesPoints.size()); i++)
            {
                keypoints22D.at<Vec2f>(i,0) = Vec2f(imagesPoints[i].x, imagesPoints[i].y);
            }

            Mat rvec, Tvec;
            solvePnP(keypoints13D, keypoints22D, matK, distCoeff, rvec, Tvec, false, methods[method]);

            EXPECT_LE(cvtest::norm(true_rvec, rvec, NORM_INF), 1e-3);
            EXPECT_LE(cvtest::norm(true_tvec, Tvec, NORM_INF), 1e-3);
        }
        {
            //vector<Point3f>
            vector<Point3f> keypoints13D;
            const int nbPts = (methods[method] == SOLVEPNP_P3P || methods[method] == SOLVEPNP_AP3P ||
                               methods[method] == SOLVEPNP_IPPE || methods[method] == SOLVEPNP_IPPE_SQUARE) ? 4 : 6;
            for (int i = 0; i < nbPts; i++)
            {
                keypoints13D.push_back(Point3f(static_cast<float>(objectPoints[i].x),
                                               static_cast<float>(objectPoints[i].y),
                                               static_cast<float>(objectPoints[i].z)));
            }

            vector<Point2f> keypoints22D;
            projectPoints(keypoints13D, true_rvec, true_tvec, matK, distCoeff, keypoints22D);

            Mat rvec, Tvec;
            solvePnP(keypoints13D, keypoints22D, matK, distCoeff, rvec, Tvec, false, methods[method]);

            EXPECT_LE(cvtest::norm(true_rvec, rvec, NORM_INF), 1e-3);
            EXPECT_LE(cvtest::norm(true_tvec, Tvec, NORM_INF), 1e-3);
        }
        {
            //vector<Point3d>
            vector<Point3d> keypoints13D;
            const int nbPts = (methods[method] == SOLVEPNP_P3P || methods[method] == SOLVEPNP_AP3P ||
                               methods[method] == SOLVEPNP_IPPE || methods[method] == SOLVEPNP_IPPE_SQUARE) ? 4 : 6;
            for (int i = 0; i < nbPts; i++)
            {
                keypoints13D.push_back(objectPoints[i]);
            }

            vector<Point2d> keypoints22D;
            projectPoints(keypoints13D, true_rvec, true_tvec, matK, distCoeff, keypoints22D);

            Mat rvec, Tvec;
            solvePnP(keypoints13D, keypoints22D, matK, distCoeff, rvec, Tvec, false, methods[method]);

            EXPECT_LE(cvtest::norm(true_rvec, rvec, NORM_INF), 1e-3);
            EXPECT_LE(cvtest::norm(true_tvec, Tvec, NORM_INF), 1e-3);
        }
    }
}

TEST(Calib3d_SolvePnP, POSIT)
{
    const double L = 10;
    vector<Point3d> cube;
    cube.push_back(Point3d(0, 0, 0));
    cube.push_back(Point3d(L, 0, 0));
    cube.push_back(Point3d(L, L, 0));
    cube.push_back(Point3d(0, L, 0));
    cube.push_back(Point3d(0, 0, L));
    cube.push_back(Point3d(L, 0, L));
    cube.push_back(Point3d(L, L, L));
    cube.push_back(Point3d(0, L, L));

    Matx33d cameraMatrix(760, 0, 320,
                         0, 760, 240,
                         0, 0, 1);

    vector<Point2d> imagePoints;
    imagePoints.push_back(Point2d(320, 240));
    imagePoints.push_back(Point2d(400, 147));
    imagePoints.push_back(Point2d(565, 163));
    imagePoints.push_back(Point2d(505, 272));
    imagePoints.push_back(Point2d(352, 375));
    imagePoints.push_back(Point2d(419, 275));
    imagePoints.push_back(Point2d(567, 302));
    imagePoints.push_back(Point2d(515, 419));

    {
        Mat rvec, tvec;
        solvePnP(cube, imagePoints, cameraMatrix, noArray(), rvec, tvec, false, SOLVEPNP_POSIT);

        std::cout << "POSIT:" << std::endl;
        std::cout << "rvec: " << rvec.t() << std::endl;
        std::cout << "tvec: " << tvec.t() << std::endl;
    }
    {
        Mat rvec, tvec;
        solvePnP(cube, imagePoints, cameraMatrix, noArray(), rvec, tvec, false, SOLVEPNP_EPNP);

        std::cout << "EPnP:" << std::endl;
        std::cout << "rvec: " << rvec.t() << std::endl;
        std::cout << "tvec: " << tvec.t() << std::endl;
    }
    {
        Mat rvec, tvec;
        solvePnP(cube, imagePoints, cameraMatrix, noArray(), rvec, tvec);

        std::cout << "Iterative:" << std::endl;
        std::cout << "rvec: " << rvec.t() << std::endl;
        std::cout << "tvec: " << tvec.t() << std::endl;
    }
}

TEST(Calib3d_SolvePnP, POSIT_COPLANAR)
{
#if 0
    std::vector<Point3d> objectPoints;
    const double L = 0.1;
    objectPoints.push_back(Point3d(-L,  L, 0));
    objectPoints.push_back(Point3d( L,  L, 0));
    objectPoints.push_back(Point3d( L, -L, 0));
    objectPoints.push_back(Point3d(-L, -L, 0));

    std::vector<Point2d> imagePoints;
    Matx31d rvec(0.1, 0.1, 0.1);
    Matx31d tvec(0.1, 0.1, 0.5);

    Matx33d cameraMatrix = Matx33d::eye();
    projectPoints(objectPoints, rvec, tvec, cameraMatrix, noArray(), imagePoints);

    for (size_t i = 0; i < imagePoints.size(); i++) {
        std::cout << "imagePoints[" << i << "]: " << imagePoints[i] << std::endl;
    }

    Matx31d rvec_est, tvec_est;
    solvePnP(objectPoints, imagePoints, cameraMatrix, noArray(), rvec_est, tvec_est);

    Mat R_est;
    Rodrigues(rvec_est, R_est);
    std::cout << "rvec_est: " << rvec_est.t() << std::endl;
    std::cout << "R_est:\n" << R_est << std::endl;
    std::cout << "tvec_est: " << tvec_est.t() << std::endl;
#else
    std::vector<Point3d> objectPoints;
    objectPoints.push_back(Point3d(-15,  0, 0));
    objectPoints.push_back(Point3d( 15,  0, 0));
    objectPoints.push_back(Point3d( 15,  500, 0));
    objectPoints.push_back(Point3d(-15,  500, 0));

    std::vector<Point2d> imagePoints;
    imagePoints.push_back(Point2d(92.6, 41.38));
    imagePoints.push_back(Point2d(97.37, 34.65));
    imagePoints.push_back(Point2d(-60.59, -23.84));
    imagePoints.push_back(Point2d(-66.37, -18.24));

    Matx33d cameraMatrix(760, 0, 0,
                         0, 760, 0,
                         0, 0, 1);
#endif

    if (0)
    {
        //test pseudo inverse by SVD
        Matx23d A(2, -1, 0,
                  4, 3, -2);

        Mat u, w_, vt;
        SVDecomp(A, w_, u, vt, SVD::FULL_UV);
        Mat w = Mat::zeros(A.rows, A.cols, CV_64FC1);
        Mat w_inv = Mat::zeros(A.rows, A.cols, CV_64FC1);
        int min_sz = std::min(A.rows, A.cols);
        for (int i = 0; i < min_sz; i++) {
            w.at<double>(i,i) = w_.at<double>(i,0);
            w_inv.at<double>(i,i) = 1 / w_.at<double>(i,0);
        }

        std::cout << "A:\n" << A << std::endl;
        std::cout << "u:\n" << u << std::endl;
        std::cout << "w:\n" << w << std::endl;
        std::cout << "w.inv():\n" << w.inv(DECOMP_SVD) << std::endl;
        std::cout << "w_inv:\n" << w_inv << std::endl;
        std::cout << "vt:\n" << vt << std::endl;

        Mat Ap = (u * w_inv * vt).t(); //vt.t() * w_inv.t() * u.t();

        std::cout << "Ap:\n" << Ap << std::endl;
    }

    if (0)
    {
        //Test Matx31d.dot
        Matx31d M(1, 2, 3);
        std::cout << "Mt x M: " << M.t()*M << std::endl;
        std::cout << "M.dot(M): " << M.dot(M) << std::endl;
    }

    vector<Point2d> imagePointsNormalized;
    undistortPoints(imagePoints, imagePointsNormalized, cameraMatrix, noArray());

    Matx33d R1, R2;
    Matx31d t1, t2;
    int nbsol = 0;
    Composit(objectPoints, imagePointsNormalized, R1, t1, R2, t2, nbsol);

    std::cout << "\nnbsol: " << nbsol << std::endl;
    std::cout << "R1:\n" << R1 << std::endl;
    std::cout << "t1: " << t1.t() << std::endl;
    std::cout << "R2:\n" << R2 << std::endl;
    std::cout << "t2: " << t2.t() << std::endl;
}

}} // namespace
