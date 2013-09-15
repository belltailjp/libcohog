#include <iostream>
#include <fstream>
#include <libcohog.hpp>
#include <tinyxml2.h>

#include "GroundTruthLoader.hpp"

static std::string filepath_to_filename(const std::string& filepath)
{
    int p;
    if((p = filepath.find_last_of("/")) != std::string::npos || (p = filepath.find_last_of("\\")) != std::string::npos)
        return filepath.substr(p + 1);
    
    return filepath;
}

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

            if(rect.rect.height < min_h || (only_confident && confidence < 1.0) || categ != 0)
                rect.confident = false;
            else
                rect.confident = true;
            rects.push_back(rect);
        }
        result[img_name] = rects;
    }
    return result;
}

std::map<std::string, std::vector<libcohog::TruthRect> > libcohog::load_rectan_ground_truth(const char *filename, int min_h, const std::set<int>& category)
{
    using namespace tinyxml2;

    std::map<std::string, std::vector<libcohog::TruthRect> > result;

    XMLDocument xml;
    xml.LoadFile(filename);

    XMLElement* images = xml.FirstChildElement("images");
    const std::string path = images->Attribute("path");

    //画像ごとに回す
    for(XMLElement* img = images->FirstChildElement("image");
        img;
        img = img->NextSiblingElement("image"))
    {
        std::vector<libcohog::TruthRect> rects;

        //属性読み取り
        const std::string src  = img->Attribute("src");
        const std::string img_name = filepath_to_filename(src); //boost::filesystem::path(src).filename().string();

        //子要素(rect)があれば
        for(XMLElement* rct = img->FirstChildElement("rect");
            rct;
            rct = rct->NextSiblingElement("rect"))
        {
            int categ;
            libcohog::TruthRect rect;
            rect.rect.x      = std::atoi(rct->Attribute("x"));
            rect.rect.y      = std::atoi(rct->Attribute("y"));
            rect.rect.width  = std::atoi(rct->Attribute("w"));
            rect.rect.height = std::atoi(rct->Attribute("h"));
            categ            = std::atoi(rct->Attribute("category"));

            if(rect.rect.height <= min_h || category.find(categ) == category.end())
                rect.confident = false;

            rects.push_back(rect);
        }

        result[img_name] = rects;
    }

    return result;
}

