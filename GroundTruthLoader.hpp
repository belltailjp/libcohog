#pragma once
#include <vector>
#include <map>
#include <libcohog.hpp>

namespace libcohog
{

std::string filepath_to_filename(const std::string& filepath);

//ファイル名(ディレクトリ名を全く含まない)と矩形列のマップ
std::map<std::string, std::vector<TruthRect> > load_daimler_ground_truth(const char* filename, int min_h, bool only_confident);
std::map<std::string, std::vector<TruthRect> > load_rectan_ground_truth(const char *filename, int min_h, const std::set<int>& category);

}

