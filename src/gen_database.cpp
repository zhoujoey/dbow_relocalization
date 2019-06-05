#include "DBoW3/DBoW3.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <iostream>
#include <vector>
#include <string>

using namespace cv;
using namespace std;

/***************************************************
 * show how to save images to dict database
 * in database , every image has it's vocabulary id. 
 * so when count new image id and check in database, can search the database id 
 * ************************************************/


int main( int argc, char** argv )
{
    string img_fold = "/home/willis/VSLAM/DBOW/voc3_demo/Freiburg2Pioneer/img/";
    const int Num_images = 379;
    cout<<"reading database"<<endl;
    // read the images and  Vocabulary
    DBoW3::Vocabulary vocab("voc.yml.gz");
    //DBoW3::Vocabulary vocab("ORBvoc.txt");
    // DBoW3::Vocabulary vocab("./vocab_larger.yml.gz");  // use large vocab if you want: 
    if ( vocab.empty() )
    {
        cerr<<"Vocabulary does not exist."<<endl;
        return 1;
    }
    cout<<"reading images... "<<endl;
    vector<Mat> images; 
    for ( int i=0; i<Num_images; i++ )
    {
        string path = img_fold + to_string(i+1) + ".png";
        images.push_back( imread(path) );
    }
    
    // detect ORB features
    cout<<"detecting ORB features ... "<<endl;
    Ptr< Feature2D > detector = ORB::create();
    vector<Mat> descriptors;
    for ( Mat& image:images )
    {
        vector<KeyPoint> keypoints; 
        Mat descriptor;
        detector->detectAndCompute( image, Mat(), keypoints, descriptor );
        descriptors.push_back( descriptor );
    }
    
    //draw descriptors to database
    cout<<"comparing images with database "<<endl;
    DBoW3::Database db( vocab, false, 0);
    for ( int i=0; i<descriptors.size(); i++ )
        db.add(descriptors[i]);
    cout<<"database info: "<<db<<endl;
    
    //save database
    db.save("database.db");
    cout<<"save data base"<<endl;
}