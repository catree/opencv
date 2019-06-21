// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include "precomp.hpp"
#include "posit_modern.hpp"

//TODO: debug
#include <iostream>

namespace cv {
namespace POSIT {
using namespace std;

void posit(const Mat& objectPoints_, const Mat& imagePoints_, Matx31d& rvec, Matx31d& tvec)
{
    Mat objectPoints, imagePoints;
    objectPoints_.convertTo(objectPoints, CV_64F);
    imagePoints_.convertTo(imagePoints, CV_64F);
    int nbPts = objectPoints.checkVector(3);

    Mat homogeneousWorldPts = Mat::ones(nbPts, 4, CV_64FC1);
    Mat centeredImageU(nbPts, 1, CV_64FC1), centeredImageV(nbPts, 1, CV_64FC1);
    for (int i = 0; i < nbPts; i++)
    {
        const Vec3d worldPts = objectPoints.ptr<Vec3d>(0)[i];
        homogeneousWorldPts.at<double>(i,0) = worldPts(0);
        homogeneousWorldPts.at<double>(i,1) = worldPts(1);
        homogeneousWorldPts.at<double>(i,2) = worldPts(2);

        const Vec2d imgPts = imagePoints.ptr<Vec2d>(0)[i];
        centeredImageU.at<double>(i,0) = imgPts(0);
        centeredImageV.at<double>(i,0) = imgPts(1);
    }

    Mat ui = centeredImageU.clone(), vi = centeredImageV.clone(), wi = Mat::ones(nbPts, 1, CV_64FC1);
    Mat objectMat = homogeneousWorldPts.inv(DECOMP_SVD);

    Mat oldUi, oldVi;
    Mat1d old_res(1,1), res(1,1);
    old_res(0,0) = 1e6;
    res(0,0) = 1e6;

    Vec3d Txyz, oldTxyz;
    Vec3d r1, r2, r3;
    Vec3d oldr1, oldr2, oldr3;

    const int max_iter = 30;
    const double res_thresh = 1e-9;
    for (int iter = 0; iter < max_iter && res(0) > res_thresh; iter++)
    {
        old_res = res.clone();
        oldTxyz = Txyz;
        oldr1 = r1;
        oldr2 = r2;
        oldr3 = r3;

        Matx41d r1T = Matx41d(Mat(objectMat * ui));
        Matx41d r2T = Matx41d(Mat(objectMat * vi));
        double Tz1 = 1 / sqrt(r1T(0)*r1T(0) + r1T(1)*r1T(1) + r1T(2)*r1T(2));
        double Tz2 = 1 / sqrt(r2T(0)*r2T(0) + r2T(1)*r2T(1) + r2T(2)*r2T(2));
        Txyz(2) = sqrt(Tz1*Tz2);

        Matx41d r1N = r1T*Txyz(2);
        Matx41d r2N = r2T*Txyz(2);

        r1 = Vec3d(r1N(0), r1N(1), r1N(2));
        r2 = Vec3d(r2N(0), r2N(1), r2N(2));
        r3 = r1.cross(r2);

        Matx41d r3T(r3(0), r3(1), r3(2), Txyz(2));
        Txyz(0) = r1N(3);
        Txyz(1) = r2N(3);

        wi = homogeneousWorldPts * r3T / Txyz(2);

        oldUi = ui.clone();
        oldVi = vi.clone();
        ui = wi.mul(centeredImageU);
        vi = wi.mul(centeredImageV);

        Mat deltaUi = ui - oldUi;
        Mat deltaVi = vi - oldVi;

        res = (deltaUi.t() * deltaUi + deltaVi.t() * deltaVi) / (2*deltaUi.rows);
        if (res(0) > old_res(0))
        {
            Txyz = oldTxyz;
            r1 = oldr1;
            r2 = oldr2;
            r3 = oldr3;
            break;
        }
    }

    tvec = Txyz;
    Matx33d R;
    for (int j = 0; j < 3; j++)
    {
        R(0,j) = r1(j);
        R(1,j) = r2(j);
        R(2,j) = r3(j);
    }

    Rodrigues(R, rvec);
}

static void positPlanarComputePose(const Matx41d& I, const Matx41d& J, Matx33d& R, Matx31d& t)
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

static double homogeneousTransformZ(const Point3d& pt, const Matx33d& R, const Matx31d& t)
{
    return R(2,0)*pt.x + R(2,1)*pt.y + R(2,2)*pt.z + t(2,0);
}

static void checkNotBehindCamera(const vector<Point3d>& objectPointsCentered, const Matx33d& Rot1, const Matx31d& Trans1,
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

static void positPlanarPoseEstimation(const vector<Point3d>& objectPointsCentered, const vector<Point2d>& imagePoints, const Mat& objectMatrix,
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

    positPlanarComputePose(I, J, rotation1, translation1);

    //Second solution
    I = I4_0 - lambda*U;
    J = J4_0 - mu*U;

    positPlanarComputePose(I, J, rotation2, translation2);
}

static double computeResidual(const vector<Point3d>& objectPoints, const vector<Point2d>& imagePoints, const Matx33d& R, const Matx31d& t)
{
    vector<Point2d> projectedPoints;
    Matx31d rvec;
    Rodrigues(R, rvec);
    projectPoints(objectPoints, rvec, t, Matx33d::eye(), noArray(), projectedPoints);

    return cv::norm(projectedPoints, imagePoints, NORM_L2SQR) / (2*projectedPoints.size());
}

static void positPlanarTreeIteration(const vector<Point3d>& objectPointsCentered, const vector<Point2d>& imagePoints, const Mat& objectMatrix,
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

        positPlanarPoseEstimation(objectPointsCentered, imagePoints, objectMatrix, U, rotation1, translation1, rotation2, translation2);

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

static int positPlanarRoot(const vector<Point3d>& objectPointsCentered, const vector<Point2d>& imagePoints, const Mat& objectMatrix,
                           const Matx41d& U, Matx33d& Rot1, Matx31d& Trans1, Matx33d& Rot2, Matx31d& Trans2)
{
    int nbSol = 0;
    positPlanarPoseEstimation(objectPointsCentered, imagePoints, objectMatrix, U, Rot1, Trans1, Rot2, Trans2);

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
        positPlanarTreeIteration(objectPointsCentered, imagePoints, objectMatrix, U, Rot1, Trans1, residual1);
    }
    else if (errorCode1 == -1 && errorCode2 == 0)
    {
        nbSol = 1;
        positPlanarTreeIteration(objectPointsCentered, imagePoints, objectMatrix, U, Rot1, Trans1, residual1);
    }
    else
    {
        nbSol = 2;

        positPlanarTreeIteration(objectPointsCentered, imagePoints, objectMatrix, U, Rot1, Trans1, residual1);
        positPlanarTreeIteration(objectPointsCentered, imagePoints, objectMatrix, U, Rot2, Trans2, residual2);

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

    return nbSol;
}

int positPlanar(const Mat& objectPoints_, const Mat& imagePoints_,
                Matx31d& rvec1, Matx31d& tvec1, Matx31d& rvec2, Matx31d& tvec2)
{
    Mat objectPoints;
    Mat imagePoints;
    objectPoints_.convertTo(objectPoints, CV_64F);
    imagePoints_.convertTo(imagePoints, CV_64F);
    int np = objectPoints.checkVector(3);

    // compute the cog of the object points
    Point3d cog;
    for (int i = 0; i < np; i++)
    {
        Point3d pt = objectPoints.ptr<Vec3d>(0)[i];
        cog.x += pt.x;
        cog.y += pt.y;
        cog.z += pt.z;
    }

    cog.x /= np;
    cog.y /= np;
    cog.z /= np;

    vector<Point3d> objectPointsCentered(np);
    Mat coplVectors(np, 4, CV_64FC1);
    for (int i = 0; i < np; i++)
    {
        Point3d pt = objectPoints.ptr<Vec3d>(0)[i];
        pt -= cog;
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
    Matx33d R1, R2;
    int nbSol = positPlanarRoot(objectPointsCentered, imagePoints, coplMatrix, U, R1, tvec1, R2, tvec2);

    if (nbSol == 0)
    {
        std::cerr << "NO SOLUTION" << std::endl;
        //TODO:
    }
    else
    {
        tvec1(0,0) -= R1(0,0)*cog.x + R1(0,1)*cog.y + R1(0,2)*cog.z;
        tvec1(1,0) -= R1(1,0)*cog.x + R1(1,1)*cog.y + R1(1,2)*cog.z;
        tvec1(2,0) -= R1(2,0)*cog.x + R1(2,1)*cog.y + R1(2,2)*cog.z;

        Rodrigues(R1, rvec1);

        if (nbSol == 2)
        {
            tvec2(0,0) -= R2(0,0)*cog.x + R2(0,1)*cog.y + R2(0,2)*cog.z;
            tvec2(1,0) -= R2(1,0)*cog.x + R2(1,1)*cog.y + R2(1,2)*cog.z;
            tvec2(2,0) -= R2(2,0)*cog.x + R2(2,1)*cog.y + R2(2,2)*cog.z;

            Rodrigues(R2, rvec2);
        }
    }

    return nbSol;
}
}
} //namespace cv
