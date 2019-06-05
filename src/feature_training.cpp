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
 * train image vocabulary from images
 * the best accuracy way is train vocabulary by user train and test images
 * the most use way is train vocabulary by large scale dataset images ,like image net 
 * ************************************************/

int main( int argc, char** argv )
{
    string img_fold = "/home/willis/VSLAM/DBOW/voc3_demo/Freiburg2Pioneer/img/";
    const int Num_images = 379;
    // read the image 
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
    
    // create vocabulary 
    cout<<"creating vocabulary ... "<<endl;
    const int k = 10;
    const int L = 5;
    const DBoW3::WeightingType weight = DBoW3::TF_IDF;
    const DBoW3::ScoringType score = DBoW3::L1_NORM;
    DBoW3::Vocabulary vocab(k, L, weight, score);
    vocab.create( descriptors );
    cout<<"vocabulary info: "<<vocab<<endl;
    vocab.save( "voc.yml.gz" );
    cout<<"done"<<endl;
    
    return 0;
}