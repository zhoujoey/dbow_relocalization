#include<DBOW.h>
#include<iostream>
#include<fstream>
using namespace cv;
using namespace std;

//detect image ORB features and train images to vocabulary.

int main( int argc, char** argv )
{
    bool is_self_train = true;
    string datafold = "/home/willis/VSLAM/Freiburg2Pioneer/";
    string trainset = "dataset_train.txt";
    //import image data from dataset.txt
    vector<ImageGroundTruth> gt_lists;
    read_data(datafold, trainset, gt_lists);
    //load features
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
    cout<<"feature size is "<<features.size()<<endl;
    //voc param
    //---------
    ORBVocabulary vocab;
    
    ifstream ifile("voc.txt");
    if(ifile){
        vocab.loadFromTextFile("voc.txt");
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
    
    ORBDatabase db(vocab, false, 0);
    for (int i = 0 ; i<gt_lists.size(); i++){
        db.add(features[i]);
    }
    db.save("database.db");
    cout<<db<<endl;
    cout<<"save data base"<<endl;
    return 0;
}