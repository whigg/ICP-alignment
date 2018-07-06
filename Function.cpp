

#include "Function.h"
#include "normalEstimation.h"
#include "Eigen/Dense"

using namespace Eigen;
using Eigen::MatrixXd;
using Eigen::Vector3d;
using namespace std;
using std::min;



double Function::Edistance(Eigen::MatrixXd V1, Eigen::MatrixXd V2){
    double d1,d2,d3;
    double dis=0;
    for(int i=0; i<V1.rows(); i++){
        d1=V1(i,0)-V2(i,0);
        d2=V1(i,1)-V2(i,1);
        d3=V1(i,2)-V2(i,2);
        dis += sqrt(pow(d1,2)+pow(d2,2)+pow(d3,2));
    }
    return dis;
}
Eigen::MatrixXd Function::transfomationMatrix_x(Eigen::MatrixXd V, double theta,double movex,double movey,double movez){
    Eigen::MatrixXd VT;
    Eigen::Matrix4d H;
    // Set the transformation Matrix
    H<<1,0,0,movex,
            0,cos(theta),sin(theta),movey,
            0,-sin(theta),cos(theta),movez,
            0,0,0,1;
    // Set n*4 matrix of original matrix V
    Eigen::MatrixXd Vtrans(V.rows(), 4);
    Vtrans<<V,Eigen::MatrixXd::Ones(V.rows(),1);
    // Use the transformation matrix to get the result
    Vtrans=(H*Vtrans.adjoint()).adjoint();
    // Adopt the first three cols
    VT=Vtrans.topLeftCorner(Vtrans.rows(),3);
    return VT;
}

Eigen::MatrixXd Function::transfomationMatrix_y(Eigen::MatrixXd V, double theta,double movex,double movey,double movez){
    Eigen::MatrixXd VT;
    Eigen::Matrix4d H;
    // Set the transformation Matrix
    H<<cos (theta),0,sin(theta),movex,
            0,1,0,movey,
            -sin(theta),0,cos(theta),movez,
            0,0,0,1;
    // Set n*4 matrix of original matrix V
    Eigen::MatrixXd Vtrans(V.rows(), 4);
    Vtrans<<V,Eigen::MatrixXd::Ones(V.rows(),1);
    // Use the transformation matrix to get the result
    Vtrans=(H*Vtrans.adjoint()).adjoint();
    // Adopt the first three cols
    VT=Vtrans.topLeftCorner(Vtrans.rows(),3);
    return VT;
}
Eigen::MatrixXd Function::transfomationMatrix_z(Eigen::MatrixXd V, double theta,double movex,double movey,double movez){
    Eigen::MatrixXd VT;
    Eigen::Matrix4d H;
    // Set the transformation Matrix
    H<<cos(theta),sin(theta),0,movex,
            -sin(theta),cos(theta),0,movey,
            0,0,1,movez,
            0,0,0,1;
    // Set n*4 matrix of original matrix V
    Eigen::MatrixXd Vtrans(V.rows(), 4);
    Vtrans<<V,Eigen::MatrixXd::Ones(V.rows(),1);
    // Use the transformation matrix to get the result
    Vtrans=(H*Vtrans.adjoint()).adjoint();
    // Adopt the first three cols
    VT=Vtrans.topLeftCorner(Vtrans.rows(),3);
    return VT;
}

Eigen::MatrixXd Function::ICP(Eigen::MatrixXd VA,Eigen::MatrixXd VB){
    int dim,numA,numB;
    dim=VA.cols();
    numA=VA.rows();
    numB=VB.rows();
    // Set the ANNpoint Array of matrix A and B
    ANNpointArray VA_points;
    ANNpointArray VB_points;
    VA_points=annAllocPts(numA,dim);
    VB_points=annAllocPts(numB,dim);
    // Give the value of VA and VB to ANNpointArray
    ANNpoint point;
    for (int i=0;i<numA;i++){
        point=annAllocPt(dim);
        point[0]=VA(i,0);
        point[1]=VA(i,1);
        point[2]=VA(i,2);
        VA_points[i]=point;
    }
    for (int i=0;i<numB;i++){
        point=annAllocPt(dim);
        point[0]=VB(i,0);
        point[1]=VB(i,1);
        point[2]=VB(i,2);
        VB_points[i]=point;
    }
    // Nearest neighbor index and distance
    ANNidxArray neighbourindex=new ANNidx[1];
    ANNdistArray neighbourdistance=new ANNdist[1];

    Eigen::MatrixXd MatrixA,MatrixB,RoatationMatrix;
    Eigen::Vector3d TranslationMatrix;
    int minnum=std::min(numA,numB);
    MatrixA.setZero(minnum,dim);
    MatrixB.setZero(minnum,dim);

    ANNkd_tree* kdTree;
    kdTree=new ANNkd_tree(VB_points,numB,dim);
    for(int i=0;i<minnum;i=i+1) {
        ANNpoint point = VA_points[i];
        kdTree->annkSearch(
                point,             // the query point
                1,                 // number of near neighbors to return
                neighbourindex,    // nearest neighbor indices (returned)
                neighbourdistance, // the approximate nearest neighbor
                0);                // the error bound
        MatrixA.row(i)=VA.row(i);
        MatrixB.row(i)=VB.row(*neighbourindex);
    }

    Eigen::Vector3d sumA,sumB;
    sumA=MatrixA.colwise().sum();
    sumB=MatrixB.colwise().sum();
    Eigen::MatrixXd Aa,Bb;
    Eigen::Vector3d AveA,AveB;
    AveA=(sumA/numA).transpose();
    AveB=(sumB/numB).transpose();
    Aa=MatrixA-AveA.transpose().replicate(numA,1);
    Bb=MatrixB-AveB.transpose().replicate(numB,1);
    Eigen::MatrixXd Cc;
    Cc = Aa.transpose()*Bb;
    // Use the JacobiSVD
    Eigen::MatrixXd svdU,svdV;
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(Cc,Eigen::ComputeThinU | Eigen::ComputeThinV);
    svdU=svd.matrixU();
    svdV=svd.matrixV();
    // Calculate the reotation and translation matrix
    RoatationMatrix=svdV*svdU.transpose();
    TranslationMatrix=AveB-RoatationMatrix*AveA;
    Eigen::MatrixXd V;
    V=(RoatationMatrix*VA.transpose()).transpose()+TranslationMatrix.replicate(1,numA).transpose();
    return V;
}

Eigen::MatrixXd Function::ICP_plane(Eigen::MatrixXd VA,Eigen::MatrixXd VB,Eigen::MatrixXd normal){
    int dim,numA,numB;
    dim=VA.cols();
    numA=VA.rows();
    numB=VB.rows();
    ANNpointArray VA_points;
    ANNpointArray VB_points;
    VA_points=annAllocPts(numA,dim);
    VB_points=annAllocPts(numB,dim);
    // Give the value of VA and VB to ANNpointArray
    ANNpoint point;
    for (int i=0;i<numA;i++){
        point=annAllocPt(dim);
        point[0]=VA(i,0);
        point[1]=VA(i,1);
        point[2]=VA(i,2);
        VA_points[i]=point;
    }
    for (int i=0;i<numB;i++){
        point=annAllocPt(dim);
        point[0]=VB(i,0);
        point[1]=VB(i,1);
        point[2]=VB(i,2);
        VB_points[i]=point;
    }
    // Nearest neighbor index and distance
    ANNidxArray neighbourindex=new ANNidx[1];
    ANNdistArray neighbourdistance=new ANNdist[1];
    int minnum=std::min(numA,numB);

    Eigen::MatrixXd Normals;
    Normals.setZero(minnum,3);
    Eigen::MatrixXd MatrixA,MatrixB,Cc,svdU,svdV;
    MatrixA.setZero(minnum,dim);
    MatrixB.setZero(minnum,dim);

    ANNkd_tree* kdTree;
    kdTree=new ANNkd_tree(VA_points,numA,dim);
    for(int i=0;i<minnum;i=i+1) {
        ANNpoint point = VB_points[i];
        kdTree->annkSearch(
                point,             // the query point
                1,                 // number of near neighbors to return
                neighbourindex,    // nearest neighbor indices (returned)
                neighbourdistance, // the approximate nearest neighbor
                0);                // the error bound
        Normals.row(i)=normal.row(*neighbourindex);
        MatrixA.row(i)=VB.row(i);
        MatrixB.row(i)=VA.row(*neighbourindex);

    }

    Eigen::Matrix3d RotationMatrix;
    Eigen::Vector3d TranslationMatrix;

    Eigen::MatrixXd A,B;
    A.setZero(minnum,6);
    B.setZero(minnum,1);
    for (int i=0; i<minnum; i++) {
        double n1,n2,n3,a1,a2,a3,b1,b2,b3;
        a1=MatrixA(i,0);
        a2=MatrixA(i,1);
        a3=MatrixA(i,2);
        b1=MatrixB(i,0);
        b2=MatrixB(i,1);
        b3=MatrixB(i,2);
        n1=Normals(i,0);
        n2=Normals(i,1);
        n3=Normals(i,2);
        A.row(i)<<n3*a2-a3*n2,n1*a3-a1*n3,n2*a1-n1*a2,Normals.row(i);
        B.row(i)<<n1*(b1-a1)+n2*(b2-a2)+n3*(b3-a3);
    }

    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A,Eigen::ComputeThinU|Eigen::ComputeThinV);
    svdU=svd.matrixU();
    svdV=svd.matrixV();
    Eigen::MatrixXd X;
    X.setZero(6,1);
    double alpha,beta,gamma;
    double acos,asin,bcos,bsin,gcos,gsin;
    acos=cos(alpha);
    asin=sin(alpha);
    bcos=cos(beta);
    bsin=sin(beta);
    gcos=cos(gamma);
    gsin=sin(gamma);
    RotationMatrix <<
            bcos*gcos,-gsin*acos+gcos*bsin*asin,gsin*asin+gcos*bsin*acos,
            gsin*bcos,gcos*acos+asin*bsin*gsin,-gcos*asin+gsin*bsin*acos,
            -bsin,bcos*asin,bcos*acos;

    Eigen::MatrixXd V;
    V = (RotationMatrix*VB.transpose()).transpose()+TranslationMatrix.replicate(1,numB).transpose();
    return V;
}
