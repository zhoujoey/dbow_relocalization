#include "DBoW2/include/DBoW2/FORB.h"
#include "DBoW2/include/DBoW2/TemplatedVocabulary.h"
#include "DBoW2/include/DBoW2/TemplatedDatabase.h"
#include "DBoW2/include/DBoW2/QueryResults.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <cstdlib>
#include <ctime>
#include <cmath>

using namespace cv;
using namespace std;

typedef DBoW2::TemplatedVocabulary<DBoW2::FORB::TDescriptor, DBoW2::FORB>
ORBVocabulary;
typedef DBoW2::TemplatedDatabase<DBoW2::FORB::TDescriptor, DBoW2::FORB> 
ORBDatabase;

struct ImageGroundTruth
{
  public:
    string name;
    double x;
    double y;
    double z;
    double qx;
    double qy;
    double qz;
    double qw;
};

void ChangeStructureORB( const cv::Mat &descriptor, vector<cv::Mat> &out) {
    for (int i = 0; i < descriptor.rows; i++) {
        out.push_back(descriptor.row(i));
    }
}

void DetectFeatures(vector<vector<Mat > > &features, vector<Mat> images, int Num_images)
{
    //cout<<"detecting ORB features ... "<<endl;
    Ptr< Feature2D > detector = ORB::create();
    vector<Mat> descriptors;
    
    features.clear();
    features.reserve(Num_images);
    for ( Mat& image:images ){
        //cv::cvtColor(image, g_image, CV_RGB2GRAY);
        vector<KeyPoint> keypoints; 
        Mat descriptor;
        detector->detectAndCompute( image, Mat(), keypoints, descriptor );
        features.push_back(vector<cv::Mat>());
        ChangeStructureORB(descriptor, features.back());
    }
}

Mat DetectFeature(Mat image)
{
    //cout<<"detecting ORB features ... "<<endl;
    Ptr< Feature2D > detector = ORB::create();
    Mat descriptor;
    vector<KeyPoint> keypoints;
    detector->detectAndCompute( image, Mat(), keypoints, descriptor );
    return descriptor;
    //features.push_back(descriptor);
}

double Str2Double(string str)
{
    double d_str = atof(str.c_str());
    return d_str;
}

void ReadData(string fold, string file_name, vector<ImageGroundTruth> & gt_images)
{
    ifstream file(fold + file_name);
    string line;
    while (getline(file, line)){
        stringstream ss(line);
        string str;
        vector<string> str_list;
        while(getline(ss, str, ' ')){
            str_list.push_back(str);
        }
        if(str_list.size() == 8 ){
            ImageGroundTruth gt_image;
            gt_image.name = str_list[0];
            gt_image.x = Str2Double(str_list[1]);
            gt_image.y = Str2Double(str_list[2]);
            gt_image.z = Str2Double(str_list[3]);
            gt_image.qx = Str2Double(str_list[4]);
            gt_image.qy = Str2Double(str_list[5]);
            gt_image.qz = Str2Double(str_list[6]);
            gt_image.qw = Str2Double(str_list[7]);
            gt_images.push_back(gt_image);
        }   
    }
}
