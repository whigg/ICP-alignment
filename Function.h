
#ifndef LIBIGL_TUTORIALS_FUNCTION_H
#define LIBIGL_TUTORIALS_FUNCTION_H


#include "igl/readOFF.h"
#include <ANN/ANN.h>
#include <iostream>
#include <cmath>
#include <random>

using namespace Eigen;
using Eigen::MatrixXd;
using Eigen::Vector3d;
class Function {

public:

    double Edistance(Eigen::MatrixXd V1, Eigen::MatrixXd V2);
    Eigen::MatrixXd transfomationMatrix_x(Eigen::MatrixXd V, double theta,double movex,double movey,double z);
    Eigen::MatrixXd transfomationMatrix_y(Eigen::MatrixXd V, double theta,double movex,double movey,double z);
    Eigen::MatrixXd transfomationMatrix_z(Eigen::MatrixXd V, double theta,double movex,double movey,double z);
    Eigen::MatrixXd ICP(Eigen::MatrixXd VA,Eigen::MatrixXd VB);
    Eigen::MatrixXd ICP_plane(Eigen::MatrixXd VA,Eigen::MatrixXd VB,Eigen::MatrixXd VAnormals);
};

#endif //LIBIGL_TUTORIALS_FUNCTION_H
