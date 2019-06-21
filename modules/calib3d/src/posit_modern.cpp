// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include "precomp.hpp"
#include "posit_modern.hpp"

//TODO: debug
#include <iostream>

namespace cv {
namespace POSIT {
void posit(const Mat& opoints_, const Mat& ipoints_, Matx31d& rvec, Matx31d& tvec)
{
    Mat opoints, ipoints;
    opoints_.convertTo(opoints, CV_64F);
    ipoints_.convertTo(ipoints, CV_64F);
    int nbPts = opoints_.checkVector(3);
    //TODO: assert

    Mat homogeneousWorldPts = Mat::ones(nbPts, 4, CV_64FC1);
    Mat centeredImageU(nbPts, 1, CV_64FC1), centeredImageV(nbPts, 1, CV_64FC1);
    for (int i = 0; i < nbPts; i++)
    {
        const Vec3d worldPts = opoints.ptr<Vec3d>(0)[i];
        homogeneousWorldPts.at<double>(i,0) = worldPts(0);
        homogeneousWorldPts.at<double>(i,1) = worldPts(1);
        homogeneousWorldPts.at<double>(i,2) = worldPts(2);

        const Vec2d imgPts = ipoints.ptr<Vec2d>(0)[i];
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

    const int max_iter = 20;
    const double res_thresh = 1e-6;
    for (int iter = 0; iter < max_iter && res(0) > res_thresh; iter++)
    {
        old_res = res.clone();
        oldTxyz = Txyz;
        oldr1 = r1;
        oldr2 = r2;
        oldr3 = r3;

        Matx41d r1T = Matx41d(Mat(objectMat * ui));
        Matx41d r2T = Matx41d(Mat(objectMat * vi));
        double Tz1 = 1/sqrt(r1T(0)*r1T(0) + r1T(1)*r1T(1) + r1T(2)*r1T(2));
        double Tz2 = 1/sqrt(r2T(0)*r2T(0) + r2T(1)*r2T(1) + r2T(2)*r2T(2));
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
}
} //namespace cv
