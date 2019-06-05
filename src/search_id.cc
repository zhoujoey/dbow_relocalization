#include<DBOW.h>

using namespace cv;
using namespace std;

/***************************************************
 * show how to load database to check images
 * ************************************************/

void DiffGroundTruth(const ImageGroundTruth im1, const ImageGroundTruth im2, double &pos, double &ang)
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
    //-----------------------
    string datafold = "/home/willis/dataset/slam_data/KingsCollege/";
    string trainset = "dataset_train.txt";
    string testset = "dataset_test.txt";
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
    ReadData(datafold, testset, gt_lists);
    //detect features
    vector<vector<Mat >> features;
    
    for (int i = 0 ; i < gt_lists.size(); i++){
        Mat t_image;
        Mat descriptor;
        vector<Mat > feature;
        t_image.push_back(imread(datafold + gt_lists[i].name));
        descriptor = DetectFeature(t_image);
        ChangeStructureORB(descriptor, feature);
        features.push_back(feature);
    }
    cout<<"feature size is "<<features.size()<<endl;
    //load train images
    cout<<"reading train images... "<<endl;
    vector<ImageGroundTruth> gt_train;
    ReadData(datafold, trainset, gt_train);
    
    start_time = time(0);
    //vector<double> poses;
    //vector<double> angles;
    double sum_pos, sum_ang;
    double max_pos = 0;
    double max_ang = 0;
    double min_pos = 1000000;
    double min_ang = 1000000;
    for(int i = 0; i < gt_lists.size(); i++){
        db.query(features[i], ret, 4);
        double pos , ang ;
        if (ret.size()!=0){
            DiffGroundTruth(gt_lists[i], gt_train[ret[0].Id], pos, ang);
            sum_pos += pos;
            sum_ang += ang;
            if (max_pos < pos) max_pos = pos;
            if (max_ang < ang) max_ang = ang;
            if (min_pos > pos) min_pos = pos;
            if (min_ang > ang) min_ang = ang;
        }  
    }
    cout<<"max pos is "<<max_pos<<"  min_pos is "<<min_pos<<endl;
    cout<<"max ang is "<<max_ang<<"  min_ang is "<<min_ang<<endl;
    cout << "diff pose is " <<sum_pos/gt_lists.size() <<"| angle is " << sum_ang/gt_lists.size() << endl;
    end_time = time(0);
    ev_time = difftime(end_time, start_time);
    cout<< "average image search time is " << ev_time / (gt_lists.size()*1.0) << "s" <<endl;
    
}
