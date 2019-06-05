#include<DBOW.h>
#include<iostream>
#include<fstream>
using namespace cv;
using namespace std;

//detect image ORB features and train images to vocabulary.

int main( int argc, char** argv )
{
    bool is_self_train = true;
    string datafold = "/home/willis/dataset/slam_data/KingsCollege/";
    string trainset = "dataset_train.txt";
    time_t start_time, end_time;
    double ev_time;
    //import image data from dataset.txt
    vector<ImageGroundTruth> gt_lists;
    read_data(datafold, trainset, gt_lists);
    //load features
    start_time = time(0);
    vector<vector<Mat >> features;
    for (int i = 0 ; i < gt_lists.size(); i++){
        Mat t_image;
        Mat descriptor;
        vector<Mat > feature;
        t_image = imread(datafold + gt_lists[i].name);
        descriptor = detectFeature(t_image);
        changeStructureORB(descriptor, feature);
        features.push_back(feature);
    }
    end_time = time(0);
    ev_time = difftime(end_time, start_time);
    cout<< "extract " <<features.size() << "features. "<< "costs time:" << ev_time << "s" <<endl;
    //voc param
    //---------
    ORBVocabulary vocab;
    
    ifstream ifile("ORBvoc.txt");
    if(ifile){
        vocab.loadFromTextFile("ORBvoc.txt");
    }
    else cout<<"no file"<<endl;
    if ( vocab.empty() ){
        cerr<<"Vocabulary does not exist. use train image to creat vocabulary "<<endl;
        //generate vocabulary
        const int k = 10;
        const int L = 6;
        const DBoW2::WeightingType weight = DBoW2::TF_IDF;
        const DBoW2::ScoringType score = DBoW2::L1_NORM;
        ORBVocabulary m_vocab(k, L, weight, score);
        m_vocab.create( features );
        m_vocab.saveToTextFile("voc.txt" );
        vocab = m_vocab;
    }
    cout<<"begining save database"<<endl;
    start_time = time(0);
    ORBDatabase db(vocab, false, 0);
    for (int i = 0 ; i<gt_lists.size(); i++){
        db.add(features[i]);
    }
    db.save("database.db");
    cout<<db<<endl;
    end_time = time(0);
    ev_time = difftime(end_time, start_time);
    cout<< "save database "<< "costs time:" << ev_time << "s" <<endl;
    return 0;
}
