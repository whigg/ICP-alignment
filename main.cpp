#include "normalEstimation.h"
#include "decoratedCloud.h"
#include "cloudManager.h"

#include "nanogui/formhelper.h"
#include "nanogui/screen.h"
#include "Function.h"
#include "igl/readOFF.h"
#include "igl/viewer/Viewer.h"

#include "nanogui/formhelper.h"
#include "nanogui/screen.h"

#include "igl/copyleft/cgal/mesh_boolean.h"
#include "igl/copyleft/cgal/intersect_other.h"

#include <iostream>
#include <cmath>
#include <random>
#include <ANN/ANN.h>


namespace acq {

/** \brief                      Re-estimate normals of cloud \p V fitting planes
 *                              to the \p kNeighbours nearest neighbours of each point.
 * \param[in ] kNeighbours      How many neighbours to use (Typiclaly: 5..15)
 * \param[in ] vertices         Input pointcloud. Nx3, where N is the number of points.
 * \param[in ] maxNeighbourDist Maximum distance between vertex and neighbour.
 * \param[out] viewer           The viewer to show the normals at.
 * \return                      The estimated normals, Nx3.
 */
    NormalsT
    recalcNormals(
            int                 const  kNeighbours,
            CloudT              const& vertices,
            float               const  maxNeighbourDist
    ) {
        NeighboursT const neighbours =
                calculateCloudNeighbours(
                        /* [in]        cloud: */ vertices,
                        /* [in] k-neighbours: */ kNeighbours,
                        /* [in]      maxDist: */ maxNeighbourDist
                );

        // Estimate normals for points in cloud vertices
        NormalsT normals =
                calculateCloudNormals(
                        /* [in]               Cloud: */ vertices,
                        /* [in] Lists of neighbours: */ neighbours
                );

        return normals;
    } //...recalcNormals()

    void setViewerNormals(
            igl::viewer::Viewer      & viewer,
            CloudT              const& vertices,
            NormalsT            const& normals
    ) {
        // [Optional] Set viewer face normals for shading
        //viewer.data.set_normals(normals);

        // Clear visualized lines (see Viewer.clear())
        viewer.data.lines = Eigen::MatrixXd(0, 9);

        // Add normals to viewer
        viewer.data.add_edges(
                /* [in] Edge starting points: */ vertices,
                /* [in]       Edge endpoints: */ vertices + normals * 0.01, // scale normals to 1% length
                /* [in]               Colors: */ Eigen::Vector3d::Zero()
        );
    }

} //...ns acq

int main(int argc, char *argv[]) {

    // How many neighbours to use for normal estimation, shown on GUI.
    int kNeighbours = 10;
    // Maximum distance between vertices to be considered neighbours (FLANN mode)
    float maxNeighbourDist = 0.15; //TODO: set to average vertex distance upon read

    // Dummy enum to demo GUI
    enum Orientation { Up=0, Down, Left, Right } dir = Up;
    // Dummy variable to demo GUI
    bool boolVariable = true;
    // Dummy variable to demo GUI
    float floatVariable = 0.1f;

    // Load a mesh in OFF format
    std::string meshPath1 = TUTORIAL_SHARED_PATH "/bunny000.off";
    std::string meshPath2 = TUTORIAL_SHARED_PATH "/bunny045.off";
    std::string meshPath3 = TUTORIAL_SHARED_PATH "/bun090.off";
    std::string meshPath4 = TUTORIAL_SHARED_PATH "/bun180.off";
    std::string meshPath5 = TUTORIAL_SHARED_PATH "/bun270.off";
    std::string meshPath6 = TUTORIAL_SHARED_PATH "/top.off";

    if (argc > 1) {
        meshPath1 = std::string(argv[1]);
        if (meshPath1.find(".off") == std::string::npos) {
            std::cerr << "Only ready for  OFF files for now...\n";
            return EXIT_FAILURE;
        }
    } else {
        std::cout << "Usage: iglFrameWork <path-to-off-mesh.off>." << "\n";
    }


    // Visualize the mesh in a viewer
    igl::viewer::Viewer viewer;
    {
        // Don't show face edges
        viewer.core.show_lines = false;
    }

    // Store cloud so we can store normals later
    acq::CloudManager cloudManager;
    // Read mesh from meshPath
    {
        // Pointcloud vertices, N rows x 3 columns.
        Eigen::MatrixXd V1,V2,V3,V4,V5,V9;
        // Face indices, M x 3 integers referring to V.
        Eigen::MatrixXi F1,F2,F3,F4,F5,F9;

        // Read mesh
        igl::readOFF(meshPath1, V1, F1);
        igl::readOFF(meshPath2, V2, F2);
        igl::readOFF(meshPath3, V3, F3);
        igl::readOFF(meshPath4, V4, F4);
        igl::readOFF(meshPath5, V5, F5);
        igl::readOFF(meshPath6, V9, F9);
        // Check, if any vertices read
        if (V1.rows() <= 0) {
            std::cerr << "Could not read mesh at " << meshPath1
                      << "...exiting...\n";
            return EXIT_FAILURE;
        } //...if vertices read

        if (V2.rows() <= 0) {
            std::cerr << "Could not read mesh at " << meshPath2
                      << "...exiting...\n";
            return EXIT_FAILURE;
        } //...if vertices read


        if (V3.rows() <= 0) {
            std::cerr << "Could not read mesh at " << meshPath3
                      << "...exiting...\n";
            return EXIT_FAILURE;
        } //...if vertices read
        if (V4.rows() <= 0) {
            std::cerr << "Could not read mesh at " << meshPath4
                      << "...exiting...\n";
            return EXIT_FAILURE;
        } //...if vertices read
        if (V5.rows() <= 0) {
            std::cerr << "Could not read mesh at " << meshPath5
                      << "...exiting...\n";
            return EXIT_FAILURE;
        } //...if vertices read
        if (V9.rows() <= 0) {
            std::cerr << "Could not read mesh at " << meshPath6
                      << "...exiting...\n";
            return EXIT_FAILURE;
        } //...if vertices read


        Eigen::MatrixXd gau(V2.rows(),V2.cols());

        // Add Gaissian noise
        for(int i = 0; i<V2.rows(); i++) {
            // A trivial random generator engine from a time-based seed
            unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
            std::default_random_engine generator (seed);
            std::normal_distribution<double> distribution (0.0,0.002);
            gau(i,1)=distribution(generator);
            gau(i,2)=distribution(generator);
            gau(i,3)=distribution(generator);
        }
        //V2=V2+gau;

        // Store read vertices and faces
        cloudManager.addCloud(acq::DecoratedCloud(V1, F1));
        cloudManager.addCloud(acq::DecoratedCloud(V2, F2));



        //***************** two meshes *****************

        double dis;
        Function fun;

        // Set the rotation angle and offset x,y,z
        double theta,movex,movey,movez;
        double radian=0.0174533;
        theta=15*radian;
        movex=0;
        movey=0;
        movez=0;
        // Original matrix
        Eigen::MatrixXd VT;
        //VT=fun.transfomationMatrix_z( cloudManager.getCloud(1).getVertices(),theta,movex,movey,movez);
        //cloudManager.getCloud(1).setVertices(VT);
        dis=fun.Edistance(cloudManager.getCloud(0).getVertices(),cloudManager.getCloud(1).getVertices());
        std::cerr << "The distance between two meshes is " <<dis << "\n";
        Eigen::MatrixXd V_icp;
        cloudManager.getCloud(0).setNormals(
                acq::recalcNormals(
                        /* [in]      k-neighbours for flann: */ 10,
                        /* [in]             vertices matrix: */ cloudManager.getCloud(1).getVertices(),
                        /* [in]      max neighbour distance: */ maxNeighbourDist
                ));
        Eigen::MatrixXd VAnormals;
        VAnormals = cloudManager.getCloud(0).getNormals();
        for(int i=0;i<20;i++){
            V_icp=fun.ICP(cloudManager.getCloud(1).getVertices(),cloudManager.getCloud(0).getVertices());

            //V_icp=fun.ICP_plane(cloudManager.getCloud(0).getVertices(),cloudManager.getCloud(1).getVertices(),VAnormals);

            cloudManager.getCloud(1).setVertices(V_icp);
            dis=fun.Edistance(cloudManager.getCloud(0).getVertices(),cloudManager.getCloud(1).getVertices());
            std::cerr << "The distance between two meshes is " <<dis << "\n";
        }


        // Calculate the overlap area
        Eigen::MatrixXi Foverlap;
        //igl::copyleft::cgal::intersect_other(cloudManager.getCloud(0).getVertices(), cloudManager.getCloud(0).getFaces(), cloudManager.getCloud(1).getVertices(), cloudManager.getCloud(1).getFaces(), false, Foverlap);

        // Concatenate two V and two F into a new one
        Eigen::MatrixXd V(cloudManager.getCloud(0).getVertices().rows()+cloudManager.getCloud(1).getVertices().rows(),cloudManager.getCloud(0).getVertices().cols());
        V<<cloudManager.getCloud(0).getVertices(),cloudManager.getCloud(1).getVertices();
        Eigen::MatrixXi F(F1.rows()+F2.rows(),F1.cols());
        F<<F1,(F2.array()+cloudManager.getCloud(0).getVertices().rows());
        // blue color for faces of first mesh, orange for second
        Eigen::MatrixXd C(F.rows(),3);
        C<<
         Eigen::RowVector3d(0.2,0.3,0.8).replicate(F1.rows(),1),
                Eigen::RowVector3d(1.0,0.7,0.2).replicate(F2.rows(),1);
        cloudManager.addCloud(acq::DecoratedCloud(V, F));
        //viewer.data.set_mesh(V,F);


        //***************** more meshes  *****************
        cloudManager.addCloud(acq::DecoratedCloud(V3, F3));//cloudManager.getCloud(3)
        cloudManager.addCloud(acq::DecoratedCloud(V4, F4));//cloudManager.getCloud(4)
        cloudManager.addCloud(acq::DecoratedCloud(V5, F5));//cloudManager.getCloud(5)

        // Bunny90.off
        theta=90*radian;
        Eigen::MatrixXd Vv3;
        Vv3=fun.transfomationMatrix_y( cloudManager.getCloud(3).getVertices(),theta,0,0,0);
        cloudManager.getCloud(3).setVertices(Vv3);
        Eigen::MatrixXd V3icp;
        for(int i=0;i<30;i=i+100) {
            V3icp=fun.ICP(cloudManager.getCloud(3).getVertices(),cloudManager.getCloud(1).getVertices());
            cloudManager.getCloud(3).setVertices(V3icp);
        }
        Eigen::MatrixXd V6(cloudManager.getCloud(2).getVertices().rows()+cloudManager.getCloud(3).getVertices().rows(),cloudManager.getCloud(2).getVertices().cols());
        V6<<cloudManager.getCloud(2).getVertices(),cloudManager.getCloud(3).getVertices();
        Eigen::MatrixXi F6(cloudManager.getCloud(2).getFaces().rows()+cloudManager.getCloud(3).getFaces().rows(),cloudManager.getCloud(2).getFaces().cols());
        F6<<cloudManager.getCloud(2).getFaces(),(cloudManager.getCloud(3).getFaces().array()+cloudManager.getCloud(2).getVertices().rows());
        cloudManager.addCloud(acq::DecoratedCloud(V6, F6));
        // Bunny180.off
        theta=180*radian;
        Eigen::MatrixXd Vv4;
        Vv4=fun.transfomationMatrix_y( cloudManager.getCloud(4).getVertices(),theta,0,0,0);
        cloudManager.getCloud(4).setVertices(Vv4);
        Eigen::MatrixXd V4icp;
        for(int i=0;i<30;i=i+100) {
        //V4icp=fun.ICP(cloudManager.getCloud(4).getVertices(),cloudManager.getCloud(3).getVertices());
        //cloudManager.getCloud(4).setVertices(V4icp);
            }
        Eigen::MatrixXd V7(cloudManager.getCloud(6).getVertices().rows()+cloudManager.getCloud(4).getVertices().rows(),cloudManager.getCloud(6).getVertices().cols());
        V7<<cloudManager.getCloud(6).getVertices(),cloudManager.getCloud(4).getVertices();
        Eigen::MatrixXi F7(cloudManager.getCloud(6).getFaces().rows()+cloudManager.getCloud(4).getFaces().rows(),cloudManager.getCloud(6).getFaces().cols());
        F7<<cloudManager.getCloud(6).getFaces(),(cloudManager.getCloud(4).getFaces().array()+cloudManager.getCloud(6).getVertices().rows());
        cloudManager.addCloud(acq::DecoratedCloud(V7, F7));
        // Bunny270.off
        theta=270*radian;
        Eigen::MatrixXd Vv5;
        Vv5=fun.transfomationMatrix_y( cloudManager.getCloud(5).getVertices(),theta,0,0,0);
        cloudManager.getCloud(5).setVertices(Vv5);
        Eigen::MatrixXd V5icp;
        for(int i=0;i<30;i=i+100) {
            //V5icp = fun.ICP(cloudManager.getCloud(5).getVertices(), cloudManager.getCloud(0).getVertices());
           // cloudManager.getCloud(5).setVertices(V5icp);
        }
        // Concatenate two V and two F into a new one
        Eigen::MatrixXd V8(cloudManager.getCloud(7).getVertices().rows()+cloudManager.getCloud(5).getVertices().rows(),cloudManager.getCloud(7).getVertices().cols());
        V8<<cloudManager.getCloud(7).getVertices(),cloudManager.getCloud(5).getVertices();
        Eigen::MatrixXi F8(cloudManager.getCloud(7).getFaces().rows()+cloudManager.getCloud(5).getFaces().rows(),cloudManager.getCloud(7).getFaces().cols());
        F8<<cloudManager.getCloud(7).getFaces(),(cloudManager.getCloud(5).getFaces().array()+cloudManager.getCloud(7).getVertices().rows());
        cloudManager.addCloud(acq::DecoratedCloud(V8, F8));
        // Bunnytop.off
        cloudManager.addCloud(acq::DecoratedCloud(V9, F9));
        theta=135*radian;
        movex=-0.055;
        movey=0.134;
        movez=0.1;
        Eigen::MatrixXd Vv6;
        Vv6=fun.transfomationMatrix_y(cloudManager.getCloud(9).getVertices(),13*radian,0,0,0);
        Vv6=fun.transfomationMatrix_x( Vv6 ,theta,movex,movey,movez);
        Vv6=fun.transfomationMatrix_z( Vv6,5*radian,0,0,0);
        cloudManager.getCloud(9).setVertices(Vv6);
        Eigen::MatrixXd V6icp;
        Eigen::MatrixXd V10(cloudManager.getCloud(8).getVertices().rows()+cloudManager.getCloud(9).getVertices().rows(),cloudManager.getCloud(8).getVertices().cols());
        V10<<cloudManager.getCloud(8).getVertices(),cloudManager.getCloud(9).getVertices();
        Eigen::MatrixXi F10(cloudManager.getCloud(8).getFaces().rows()+cloudManager.getCloud(9).getFaces().rows(),cloudManager.getCloud(8).getFaces().cols());
        F10<<cloudManager.getCloud(8).getFaces(),(cloudManager.getCloud(9).getFaces().array()+cloudManager.getCloud(8).getVertices().rows());
        cloudManager.addCloud(acq::DecoratedCloud(V10, F10));

        // Set colors
        Eigen::MatrixXd C7(F10.rows(),3);
        C7<<
          Eigen::RowVector3d(0.2,0.3,0.8).replicate(cloudManager.getCloud(0).getFaces().rows(),1),
                Eigen::RowVector3d(0.2,1.0,0.8).replicate(cloudManager.getCloud(1).getFaces().rows(),1),
                Eigen::RowVector3d(1.0,0.3,0.2).replicate(cloudManager.getCloud(3).getFaces().rows(),1),
                Eigen::RowVector3d(0.4,0.3,0.2).replicate(cloudManager.getCloud(4).getFaces().rows(),1),
                Eigen::RowVector3d(0.1,0.3,0.2).replicate(cloudManager.getCloud(5).getFaces().rows(),1),
                Eigen::RowVector3d(0.6,0.3,0.5).replicate(cloudManager.getCloud(9).getFaces().rows(),1);

        // Show mesh
        viewer.data.set_mesh(
                cloudManager.getCloud(2).getVertices(),
                cloudManager.getCloud(2).getFaces()
        );
        viewer.data.set_colors(C);
        // Calculate normals on launch
        cloudManager.getCloud(2).setNormals(
                acq::recalcNormals(
                        /* [in]      K-neighbours for FLANN: */ kNeighbours,
                        /* [in]             Vertices matrix: */ cloudManager.getCloud(2).getVertices(),
                        /* [in]      max neighbour distance: */ maxNeighbourDist
                )
        );
        // Update viewer
        acq::setViewerNormals(
                viewer,
                cloudManager.getCloud(2).getVertices(),
                cloudManager.getCloud(2).getNormals()
        );
    } //...read mesh


    // Extend viewer menu using a lambda function
    viewer.callback_init =
            [
                    &cloudManager, &kNeighbours, &maxNeighbourDist,
                    &floatVariable, &boolVariable, &dir
            ] (igl::viewer::Viewer& viewer)
            {
                // Add an additional menu window
                viewer.ngui->addWindow(Eigen::Vector2i(900,10), "Acquisition3D");

                // Add new group
                viewer.ngui->addGroup("Nearest neighbours (pointcloud, FLANN)");

                // Add k-neighbours variable to GUI
                viewer.ngui->addVariable<int>(
                        /* Displayed name: */ "k-neighbours",

                        /*  Setter lambda: */ [&] (int val) {
                            // Store reference to current cloud (id 0 for now)
                            acq::DecoratedCloud &cloud = cloudManager.getCloud(0);

                            // Store new value
                            kNeighbours = val;

                            // Recalculate normals for cloud and update viewer
                            cloud.setNormals(
                                    acq::recalcNormals(
                                            /* [in]      K-neighbours for FLANN: */ kNeighbours,
                                            /* [in]             Vertices matrix: */ cloud.getVertices(),
                                            /* [in]      max neighbour distance: */ maxNeighbourDist
                                    )
                            );

                            // Update viewer
                            acq::setViewerNormals(
                                    /* [in, out] Viewer to update: */ viewer,
                                    /* [in]            Pointcloud: */ cloud.getVertices(),
                                    /* [in] Normals of Pointcloud: */ cloud.getNormals()
                            );
                        }, //...setter lambda

                        /*  Getter lambda: */ [&]() {
                            return kNeighbours; // get
                        } //...getter lambda
                ); //...addVariable(kNeighbours)

                // Add maxNeighbourDistance variable to GUI
                viewer.ngui->addVariable<float>(
                        /* Displayed name: */ "maxNeighDist",

                        /*  Setter lambda: */ [&] (float val) {
                            // Store reference to current cloud (id 0 for now)
                            acq::DecoratedCloud &cloud = cloudManager.getCloud(0);

                            // Store new value
                            maxNeighbourDist = val;

                            // Recalculate normals for cloud and update viewer
                            cloud.setNormals(
                                    acq::recalcNormals(
                                            /* [in]      K-neighbours for FLANN: */ kNeighbours,
                                            /* [in]             Vertices matrix: */ cloud.getVertices(),
                                            /* [in]      max neighbour distance: */ maxNeighbourDist
                                    )
                            );

                            // Update viewer
                            acq::setViewerNormals(
                                    /* [in, out] Viewer to update: */ viewer,
                                    /* [in]            Pointcloud: */ cloud.getVertices(),
                                    /* [in] Normals of Pointcloud: */ cloud.getNormals()
                            );
                        }, //...setter lambda

                        /*  Getter lambda: */ [&]() {
                            return maxNeighbourDist; // get
                        } //...getter lambda
                ); //...addVariable(kNeighbours)

                // Add a button for estimating normals using FLANN as neighbourhood
                // same, as changing kNeighbours
                viewer.ngui->addButton(
                        /* displayed label: */ "Estimate normals (FLANN)",

                        /* lambda to call: */ [&]() {
                            // store reference to current cloud (id 0 for now)
                            acq::DecoratedCloud &cloud = cloudManager.getCloud(0);

                            // calculate normals for cloud and update viewer
                            cloud.setNormals(
                                    acq::recalcNormals(
                                            /* [in]      k-neighbours for flann: */ kNeighbours,
                                            /* [in]             vertices matrix: */ cloud.getVertices(),
                                            /* [in]      max neighbour distance: */ maxNeighbourDist
                                    )
                            );

                            // update viewer
                            acq::setViewerNormals(
                                    /* [in, out] viewer to update: */ viewer,
                                    /* [in]            pointcloud: */ cloud.getVertices(),
                                    /* [in] normals of pointcloud: */ cloud.getNormals()
                            );
                        } //...button push lambda
                ); //...estimate normals using FLANN

                // Add a button for orienting normals using FLANN
                viewer.ngui->addButton(
                        /* Displayed label: */ "Orient normals (FLANN)",

                        /* Lambda to call: */ [&]() {
                            // Store reference to current cloud (id 0 for now)
                            acq::DecoratedCloud &cloud = cloudManager.getCloud(0);

                            // Check, if normals already exist
                            if (!cloud.hasNormals())
                                cloud.setNormals(
                                        acq::recalcNormals(
                                                /* [in]      K-neighbours for FLANN: */ kNeighbours,
                                                /* [in]             Vertices matrix: */ cloud.getVertices(),
                                                /* [in]      max neighbour distance: */ maxNeighbourDist
                                        )
                                );

                            // Estimate neighbours using FLANN
                            acq::NeighboursT const neighbours =
                                    acq::calculateCloudNeighbours(
                                            /* [in]        Cloud: */ cloud.getVertices(),
                                            /* [in] k-neighbours: */ kNeighbours,
                                            /* [in]      maxDist: */ maxNeighbourDist
                                    );

                            // Orient normals in place using established neighbourhood
                            int nFlips =
                                    acq::orientCloudNormals(
                                            /* [in    ] Lists of neighbours: */ neighbours,
                                            /* [in,out]   Normals to change: */ cloud.getNormals()
                                    );
                            std::cout << "nFlips: " << nFlips << "/" << cloud.getNormals().size() << "\n";

                            // Update viewer
                            acq::setViewerNormals(
                                    /* [in, out] Viewer to update: */ viewer,
                                    /* [in]            Pointcloud: */ cloud.getVertices(),
                                    /* [in] Normals of Pointcloud: */ cloud.getNormals()
                            );
                        } //...lambda to call on buttonclick
                ); //...addButton(orientFLANN)


                // Add new group
                viewer.ngui->addGroup("Connectivity from faces ");

                // Add a button for estimating normals using faces as neighbourhood
                viewer.ngui->addButton(
                        /* Displayed label: */ "Estimate normals (from faces)",

                        /* Lambda to call: */ [&]() {
                            // Store reference to current cloud (id 0 for now)
                            acq::DecoratedCloud &cloud = cloudManager.getCloud(0);

                            // Check, if normals already exist
                            if (!cloud.hasNormals())
                                cloud.setNormals(
                                        acq::recalcNormals(
                                                /* [in]      K-neighbours for FLANN: */ kNeighbours,
                                                /* [in]             Vertices matrix: */ cloud.getVertices(),
                                                /* [in]      max neighbour distance: */ maxNeighbourDist
                                        )
                                );

                            // Estimate neighbours using FLANN
                            acq::NeighboursT const neighbours =
                                    acq::calculateCloudNeighboursFromFaces(
                                            /* [in] Faces: */ cloud.getFaces()
                                    );

                            // Estimate normals for points in cloud vertices
                            cloud.setNormals(
                                    acq::calculateCloudNormals(
                                            /* [in]               Cloud: */ cloud.getVertices(),
                                            /* [in] Lists of neighbours: */ neighbours
                                    )
                            );

                            // Update viewer
                            acq::setViewerNormals(
                                    /* [in, out] Viewer to update: */ viewer,
                                    /* [in]            Pointcloud: */ cloud.getVertices(),
                                    /* [in] Normals of Pointcloud: */ cloud.getNormals()
                            );
                        } //...button push lambda
                ); //...estimate normals from faces

                // Add a button for orienting normals using face information
                viewer.ngui->addButton(
                        /* Displayed label: */ "Orient normals (from faces)",

                        /* Lambda to call: */ [&]() {
                            // Store reference to current cloud (id 0 for now)
                            acq::DecoratedCloud &cloud = cloudManager.getCloud(0);

                            // Check, if normals already exist
                            if (!cloud.hasNormals())
                                cloud.setNormals(
                                        acq::recalcNormals(
                                                /* [in]      K-neighbours for FLANN: */ kNeighbours,
                                                /* [in]             Vertices matrix: */ cloud.getVertices(),
                                                /* [in]      max neighbour distance: */ maxNeighbourDist
                                        )
                                );

                            // Orient normals in place using established neighbourhood
                            int nFlips =
                                    acq::orientCloudNormalsFromFaces(
                                            /* [in    ] Lists of neighbours: */ cloud.getFaces(),
                                            /* [in,out]   Normals to change: */ cloud.getNormals()
                                    );
                            std::cout << "nFlips: " << nFlips << "/" << cloud.getNormals().size() << "\n";

                            // Update viewer
                            acq::setViewerNormals(
                                    /* [in, out] Viewer to update: */ viewer,
                                    /* [in]            Pointcloud: */ cloud.getVertices(),
                                    /* [in] Normals of Pointcloud: */ cloud.getNormals()
                            );
                        } //...lambda to call on buttonclick
                ); //...addButton(orientFromFaces)


                // Add new group
                viewer.ngui->addGroup("Util");

                // Add a button for flipping normals
                viewer.ngui->addButton(
                        /* Displayed label: */ "Flip normals",
                        /*  Lambda to call: */ [&](){
                            // Store reference to current cloud (id 0 for now)
                            acq::DecoratedCloud &cloud = cloudManager.getCloud(0);

                            // Flip normals
                            cloud.getNormals() *= -1.f;

                            // Update viewer
                            acq::setViewerNormals(
                                    /* [in, out] Viewer to update: */ viewer,
                                    /* [in]            Pointcloud: */ cloud.getVertices(),
                                    /* [in] Normals of Pointcloud: */ cloud.getNormals()
                            );
                        } //...lambda to call on buttonclick
                );

                // Add a button for setting estimated normals for shading
                viewer.ngui->addButton(
                        /* Displayed label: */ "Set shading normals",
                        /*  Lambda to call: */ [&](){

                            // Store reference to current cloud (id 0 for now)
                            acq::DecoratedCloud &cloud = cloudManager.getCloud(0);

                            // Set normals to be used by viewer
                            viewer.data.set_normals(cloud.getNormals());

                        } //...lambda to call on buttonclick
                );

                // ------------------------
                // Dummy libIGL/nanoGUI API demo stuff:
                // ------------------------

                // Add new group
                viewer.ngui->addGroup("Dummy GUI demo");

                // Expose variable directly ...
                viewer.ngui->addVariable("float", floatVariable);

                // ... or using a custom callback
                viewer.ngui->addVariable<bool>(
                        "bool",
                        [&](bool val) {
                            boolVariable = val; // set
                        },
                        [&]() {
                            return boolVariable; // get
                        }
                );

                // Expose an enumaration type
                viewer.ngui->addVariable<Orientation>("Direction",dir)->setItems(
                        {"Up","Down","Left","Right"}
                );

                // Add a button
                viewer.ngui->addButton("Print Hello",[]() {
                    std::cout << "Hello\n";
                });

                // Generate menu
                viewer.screen->performLayout();

                return false;
            }; //...viewer menu


    // Start viewer
    viewer.launch();

    return 0;
} //...main()



