#include<DBOW.h>

using namespace cv;
using namespace std;

/***************************************************
 * show how to load database to check images
 * ************************************************/

void diff_gt(const ImageGroundTruth im1, const ImageGroundTruth im2, double &pos, double &ang)
{
    pos = sqrt(pow((im1.x - im2.x), 2) +\
                     pow((im1.y - im2.y), 2) + \
                     pow((im1.z - im2.z), 2));
    
    ang = sqrt(pow((im1.qx - im2.qx), 2) +\
                     pow((im1.qy - im2.qy), 2) +\
                     pow((im1.qz - im2.qz), 2) +\
                     pow((im1.qw - im2.qw), 2));
}

int main( int argc, char** argv )
{
    string datafold = "/home/willis/VSLAM/Freiburg2Pioneer/";
    string trainset = "dataset_train.txt";
    time_t start_time, end_time;
    double ev_time;
    DBoW2::QueryResults ret;
    //load database
    start_time = time(0);
    cout<<"loaded data base"<<endl;
    ORBDatabase db;
    db.load("database.db");
    cout<<"loaded data base"<<endl;
    end_time = time(0);
    ev_time = difftime(end_time, start_time);
    cout<< "loading database costs " << ev_time << "s" <<endl;
    //load test images
    cout<<"reading test images... "<<endl;
    vector<ImageGroundTruth> gt_lists;
    read_data(datafold, "dataset_test.txt", gt_lists);
    vector<vector<Mat > > features;
    for ( int i=0; i<gt_lists.size(); i++ ){
        string path = gt_lists[i].name;
        Mat t_image;
        vector<Mat > feature;
        t_image = imread(path);
        detectFeature(feature, t_image);
        features.push_back(feature);
    }
    //load train images
    cout<<"reading train images... "<<endl;
    vector<ImageGroundTruth> gt_train;
    read_data(datafold, "dataset_train.txt", gt_train);
    
    start_time = time(0);
    double sum_pos, sum_ang;
    for(int i = 0; i < gt_lists.size(); i++){
        db.query(features[i], ret, 4);
        double pos , ang ;
        cout<<ret<<endl;
        diff_gt(gt_lists[i], gt_train[ret[0].Id], pos, ang);
        sum_pos += pos;
        sum_ang += ang;    
    }
    cout << "diff pose is " <<sum_pos/gt_lists.size() <<"| angle is " << sum_ang/gt_lists.size() << endl;
    end_time = time(0);
    ev_time = difftime(end_time, start_time);
    cout<< "average image search time is " << ev_time / gt_lists.size() << "s" <<endl;
    
}