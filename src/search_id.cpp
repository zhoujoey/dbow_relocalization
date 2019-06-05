#include "DBoW3/DBoW3.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <ctime>

using namespace cv;
using namespace std;

/***************************************************
 * show how to load database to check images
 * ************************************************/
int main( int argc, char** argv )
{
    string img_fold = "/home/willis/VSLAM/DBOW/voc3_demo/Freiburg2Pioneer/img/";
    const int Num_images = 379;
    //load database
    DBoW3::Database db;
    db.load("database.db");
    cout<<"load data base"<<endl;
    DBoW3::QueryResults ret;
    //load images
    vector<Mat> images; 
    for ( int i=0; i<Num_images; i++ )
    {
        string path = img_fold + to_string(i+1) + ".png";
        images.push_back( imread(path) );
    }
    //get orb features
    Ptr< Feature2D > detector = ORB::create();
    time_t start_time = time(0);
    int k = 0;
    for ( Mat& image:images )
    {
        vector<KeyPoint> keypoints; 
        Mat descriptor;
        detector->detectAndCompute( image, Mat(), keypoints, descriptor );
        db.query( descriptor, ret, 1);
        cout<<"searching for image  "<< k << " returns "<<ret<<endl<<endl;
        k++;
    }
    time_t end_time = time(0);
    double ev_time = difftime(end_time, start_time);
    cout<< "average image search time is " << ev_time / Num_images * 1000 << "ms" <<endl;
}