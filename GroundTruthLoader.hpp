#pragma once
#include <vector>
#include <map>
#include <libcohog.hpp>

namespace libcohog
{

//ファイル名(ディレクトリ名を全く含まない)と矩形列のマップ
std::map<std::string, std::vector<TruthRect> > load_daimler_ground_truth(const char* filename, int min_h, bool only_confident);

}

