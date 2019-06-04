#include<DBOW.h>

using namespace cv;
using namespace std;


int main( int argc, char** argv )
{
    const int Num_images = 379;
    // read the image 
    //read images
    cout<<"reading images... "<<endl;
    vector<Mat> images;
    vector<ImageGroundTruth> gt_lists;
    read_data("dataset_train.txt", gt_lists);
    for ( int i=0; i<gt_lists.size(); i++ ){
        string path = gt_lists[i].name;
        images.push_back( imread(path) );
    }
    /*
    cout<<"reading images... "<<endl;
    vector<Mat> images; 
    for ( int i=0; i<Num_images; i++ )
    {
        string path = "./img/"+to_string(i+1)+".png";
        images.push_back( imread(path) );
    }*/
    // detect ORB features
    vector<vector<Mat > > features;
    detectFeatures(features, images, images.size());

    // create vocabulary 
    cout<<"creating vocabulary ... "<<endl;
    const int k = 10;
    const int L = 5;
    const DBoW2::WeightingType weight = DBoW2::TF_IDF;
    const DBoW2::ScoringType score = DBoW2::L1_NORM;
    ORBVocabulary vocab(k, L, weight, score);
    vocab.create( features );
    cout<<"vocabulary info: "<<vocab<<endl;
    vocab.saveToTextFile("voc.txt" );
    cout<<"done"<<endl;
    
    return 0;
}
