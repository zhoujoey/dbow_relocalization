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
    ifstream in(datafold + trainset);
    if(!in){
        cerr<<"cannot open train file"<<endl;
        return 0;
    }
    string str;
    vector<string> image_path;
    while(getline(in, str, ' ')){
        if(str.size()>0) image_path.push_back(datafold + str);
    }
    in.close();
    //load features
    vector<vector<Mat > > features;
    for (int i = 0 ; i < image_path.size(); i++){
        Mat t_image;
        vector<Mat > feature;
        t_image = imread(image_path[i]);
        detectFeature(feature, t_image);
        features.push_back(feature);
    }
    cout<<features.size()<<endl;
    //voc param
    //---------
    ORBVocabulary vocab;
    vocab.loadFromTextFile("ORBvoc.txt");
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
    for (int i = 0 ; i<image_path.size(); i++){
        db.add(features[i]);
    }
    db.save("database.db");
    cout<<db<<endl;
    cout<<"save data base"<<endl;
    return 0;
}