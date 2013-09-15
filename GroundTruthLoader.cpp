#include <iostream>
#include <fstream>
#include <boost/filesystem.hpp>
#include <libcohog.hpp>

#include "GroundTruthLoader.hpp"

std::map<std::string, std::vector<libcohog::TruthRect> > libcohog::load_daimler_ground_truth(const char* filename, int min_h, bool only_confident)
{
    std::ifstream is(filename);

    std::map<std::string, std::vector<libcohog::TruthRect> > result;

    //シーケンス先頭まで読み飛ばす
    while(!is.eof() && is.get() != ':')
        ;

    //ヘッダの読み込み
    std::string name;
    std::string path;
    int cnt;
    is >> name >> path >> cnt;

    //データの読み込み
    for(int i = 0; i < cnt; i++)
    {
        //データのあるところ(';'文字で始まる)まで読み込み
        while(!is.eof() && is.get() != ';')
            ;
        
        std::string img_name;
        int w, h;
        int ignore, n;
        is >> img_name >> w >> h;
        is >> ignore >> n;

        std::vector<libcohog::TruthRect> rects;

        for(int j = 0; j < n; ++j)
        {
            libcohog::TruthRect rect;

            //#の行を1行読み込む(読み捨て)
            //※この時点でカーソルは前の行の末にあるかもしれない
            char c;
            int categ;
            is >> c >> categ;

            int obj_id, unique_id;
            float confidence;
            is >> obj_id >> unique_id >> confidence;

            is >> rect.rect.x >> rect.rect.y >> rect.rect.width >> rect.rect.height;
            is >> ignore;

            rect.rect.width  -= rect.rect.x;
            rect.rect.height -= rect.rect.y;

            if(rect.rect.height < min_h || (only_confident && confidence < 1.0))
                rect.confident = false;
            else
                rect.confident = true;

            if(categ == 0)
                rects.push_back(rect);
        }
        result[img_name] = rects;
    }
    return result;
}

