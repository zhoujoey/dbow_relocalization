#include<DBOW.h>

using namespace cv;
using namespace std;

/***************************************************
 * show how to save images to dict database
 * ************************************************/

int main( int argc, char** argv )
{
    // read Vocabulary
    cout<<"reading database"<<endl;
    ORBVocabulary vocab;
    vocab.loadFromTextFile("voc.txt");
    if ( vocab.empty() ){
        cerr<<"Vocabulary does not exist."<<endl;
        return 1;
    }

    //read images
    cout<<"reading images... "<<endl;
    vector<Mat> images;
    vector<ImageGroundTruth> gt_lists;
    read_data("dataset_train.txt", gt_lists);
    for ( int i=0; i<gt_lists.size(); i++ ){
        string path = gt_lists[i].name;
        images.push_back( imread(path) );
    }
    
    // detect ORB features
    vector<vector<Mat > > features;
    detectFeatures(features, images, gt_lists.size());
    
    //draw descriptors to database
    cout<<"comparing images with database "<<endl;
    ORBDatabase db(vocab, false, 0);
    for ( int i=0; i<features.size(); i++ )
        db.add(features[i]);
    cout<<"database info: "<<db<<endl;
    
    //save database
    db.save("database.db");
    cout<<"save data base"<<endl;
    
}