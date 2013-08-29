#include <libcohog/Trainer.hpp>

namespace libcohog
{

model* train_liblinear(const std::vector<std::vector<float> >& positive_features, const std::vector<std::vector<float> >& negative_features)
{
    const int dim = positive_features[0].size();

    std::vector<std::vector<float> > features;
    features.insert(features.end(), positive_features.begin(), positive_features.end());
    features.insert(features.end(), negative_features.begin(), negative_features.end());

    std::vector<double> responses(features.size());
    std::fill(responses.begin(), responses.begin() + positive_features.size(), 1);
    std::fill(responses.begin() + positive_features.size(), responses.end(),  -1);

    //liblinear問題構築
    problem prob;
    prob.n = dim;               //次元数
    prob.y = responses.data();  //ラベル配列
    prob.l = responses.size();  //学習データ数
    prob.bias = 0;

    std::vector<std::vector<feature_node> > features_liblinear;
    std::vector<feature_node*> features_ptr;
    for(int i = 0; i < responses.size(); ++i)
    {
        const std::vector<float>& feature = features[i];

        std::vector<feature_node> feature_nodes(dim + 1);
        for(int k = 0; k < dim; ++k)
            feature_nodes[k] = feature_node{k + 1, feature[k]};
        feature_nodes[dim] = feature_node{-1, 0};

        features_liblinear.push_back(feature_nodes);
        features_ptr.push_back(features_liblinear[features_liblinear.size() - 1].data());
    }
    prob.x = features_ptr.data();

    {
        for(int i = 0; i < responses.size(); ++i)
        {
            for(int k = 0; k < dim; ++k)
            {
                feature_node node = features_ptr[i][k];
                if(node.index != k + 1)
                    std::cout << "データ" << i << " indexが(" << k << " != " << node.index << ")" << std::endl;
                if(node.value != features[i][k])
                    std::cout << "valueが(" << features[i][k] << " != " << node.value << ")" << std::endl;
            }
        }
    }

    //liblinear学習
    parameter par = default_liblinear_parameter();
    model *m = train(&prob, &par);

    //識別テスト
    int cnt = 0;
    for(int i = 0; i < features.size(); ++i)
        if(predict(m, features_ptr[i]) != prob.y[i])
            ++cnt;
    std::cerr << "misses:" << cnt << " (" << (100 * cnt / responses.size()) << "%)" << std::endl;

    return m;
}

}

