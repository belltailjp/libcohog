#include "libcohog.hpp"
#include "gtest/gtest.h"

#include <boost/assign.hpp>
using namespace boost::assign;

TEST(verifier_test, normalize_rectangle)
{
    const cv::Rect in(100, 100, 10, 20);
    const cv::Rect out = libcohog::normalize_rectangle(in, 2, 0.8);

    //中心が等しい
    EXPECT_EQ(in.x + in.width  / 2, out.x + out.width  / 2);
    EXPECT_EQ(in.y + in.height / 2, out.y + out.height / 2);

    //アスペクト比が2になってる
    EXPECT_EQ(out.width * 2, out.height);

    //高さが1.25倍になっている(1画素の誤差は容認)
    EXPECT_TRUE(std::abs((int)(in.height * 1.25) - out.height) <= 1);
}


TEST(verifier_test, test_equivalent)
{
    const cv::Rect r(200, 200, 50, 100);

    //接していて交差してない矩形
    EXPECT_FALSE(libcohog::is_equivalent(r, cv::Rect(r.x + r.width, r.y - r.height, r.width, r.height), 0.5)); //右上
    EXPECT_FALSE(libcohog::is_equivalent(r, cv::Rect(r.x + r.width, r.y,            r.width, r.height), 0.5)); //右
    EXPECT_FALSE(libcohog::is_equivalent(r, cv::Rect(r.x + r.width, r.y + r.height, r.width, r.height), 0.5)); //右下
    EXPECT_FALSE(libcohog::is_equivalent(r, cv::Rect(r.x,           r.y + r.height, r.width, r.height), 0.5)); //下
    EXPECT_FALSE(libcohog::is_equivalent(r, cv::Rect(r.x - r.width, r.y + r.height, r.width, r.height), 0.5)); //左下
    EXPECT_FALSE(libcohog::is_equivalent(r, cv::Rect(r.x - r.width, r.y,            r.width, r.height), 0.5)); //左
    EXPECT_FALSE(libcohog::is_equivalent(r, cv::Rect(r.x - r.width, r.y - r.height, r.width, r.height), 0.5)); //左上
    EXPECT_FALSE(libcohog::is_equivalent(r, cv::Rect(r.x,           r.y - r.height, r.width, r.height), 0.5)); //上

    //完全に離れてる
    EXPECT_FALSE(libcohog::is_equivalent(r, cv::Rect(r.x + r.width + 1, r.y - r.height - 1, r.width, r.height), 0.5)); //右上
    EXPECT_FALSE(libcohog::is_equivalent(r, cv::Rect(r.x + r.width + 1, r.y,                r.width, r.height), 0.5)); //右
    EXPECT_FALSE(libcohog::is_equivalent(r, cv::Rect(r.x + r.width + 1, r.y + r.height + 1, r.width, r.height), 0.5)); //右下
    EXPECT_FALSE(libcohog::is_equivalent(r, cv::Rect(r.x,               r.y + r.height + 1, r.width, r.height), 0.5)); //下
    EXPECT_FALSE(libcohog::is_equivalent(r, cv::Rect(r.x - r.width - 1, r.y + r.height + 1, r.width, r.height), 0.5)); //左下
    EXPECT_FALSE(libcohog::is_equivalent(r, cv::Rect(r.x - r.width - 1, r.y,                r.width, r.height), 0.5)); //左
    EXPECT_FALSE(libcohog::is_equivalent(r, cv::Rect(r.x - r.width - 1, r.y - r.height - 1, r.width, r.height), 0.5)); //左上
    EXPECT_FALSE(libcohog::is_equivalent(r, cv::Rect(r.x,               r.y - r.height - 1, r.width, r.height), 0.5)); //上

    //上下左右にちょうど1/3だけずれてのでTRUE
    EXPECT_TRUE (libcohog::is_equivalent(r, cv::Rect(r.x + r.width / 3, r.y,                r.width, r.height), 0.5)); //右
    EXPECT_TRUE (libcohog::is_equivalent(r, cv::Rect(r.x,               r.y + r.height / 3, r.width, r.height), 0.5)); //下
    EXPECT_TRUE (libcohog::is_equivalent(r, cv::Rect(r.x - r.width / 3, r.y,                r.width, r.height), 0.5)); //左
    EXPECT_TRUE (libcohog::is_equivalent(r, cv::Rect(r.x,               r.y - r.height / 3, r.width, r.height), 0.5)); //上

    //上下左右に1/3より1画素ずれてのでFALSE
    EXPECT_FALSE(libcohog::is_equivalent(r, cv::Rect(r.x + r.width / 3 + 1, r.y,                    r.width, r.height), 0.5)); //右
    EXPECT_FALSE(libcohog::is_equivalent(r, cv::Rect(r.x,                   r.y + r.height / 3 + 1, r.width, r.height), 0.5)); //下
    EXPECT_FALSE(libcohog::is_equivalent(r, cv::Rect(r.x - r.width / 3 - 1, r.y,                    r.width, r.height), 0.5)); //左
    EXPECT_FALSE(libcohog::is_equivalent(r, cv::Rect(r.x,                   r.y - r.height / 3 - 1, r.width, r.height), 0.5)); //上

    //左上が一致して面積が半分前後・二倍前後
    EXPECT_FALSE(libcohog::is_equivalent(r, cv::Rect(r.x, r.y, r.width * 0.7,     r.height * 0.7    ), 0.5));
    EXPECT_TRUE (libcohog::is_equivalent(r, cv::Rect(r.x, r.y, r.width * 0.7 + 1, r.height * 0.7 + 1), 0.5));
    EXPECT_TRUE (libcohog::is_equivalent(r, cv::Rect(r.x, r.y, r.width * 1.4,     r.height * 1.4    ), 0.5));
    EXPECT_FALSE(libcohog::is_equivalent(r, cv::Rect(r.x, r.y, r.width * 1.4 + 1, r.height * 1.4 + 1), 0.5));
}

TEST(verifier_test, simple_verify_test)
{
    std::vector<libcohog::Window> windows;
    windows += libcohog::Window{100, 100, 100, 200, 1.0};

    std::vector<libcohog::TruthRect> truth;
    truth += libcohog::TruthRect{cv::Rect(100, 100, 100, 200), true};

    const libcohog::VerificationResult result = libcohog::verify(windows, truth, 0, 0.5, 0, 0); //eps=0なのでグルーピングはしない
    
    EXPECT_EQ(result.windows_grouped.size(), 1);
    EXPECT_EQ(result.TP_flags.size(), 1);
    EXPECT_EQ(result.TP_grouped_flags.size(), 1);
    EXPECT_TRUE(result.TP_flags[0]);
    EXPECT_TRUE(result.TP_grouped_flags[0]);
    EXPECT_TRUE(result.found_flags[0]);

    const libcohog::EvaluationResult eval = result.to_eval();
    EXPECT_EQ(eval.nTP, 1);
    EXPECT_EQ(eval.nFP, 0);
    EXPECT_EQ(eval.nFN, 0);
}

TEST(verifier_test, found_2_tp_windows)
{
    std::vector<libcohog::Window> windows;
    windows += libcohog::Window{100, 100, 100, 200, 1.0};
    windows += libcohog::Window{100, 110, 100, 200, 1.0};   //少しずれた位置にあるTP

    std::vector<libcohog::TruthRect> truth;
    truth += libcohog::TruthRect{cv::Rect(100, 100, 100, 200), true};

    const libcohog::VerificationResult result = libcohog::verify(windows, truth, 0, 0.5, 0, 0); //eps=0なのでグルーピングはしない
    
    EXPECT_EQ(result.windows_grouped.size(), 2);
    EXPECT_EQ(result.TP_flags.size(), 2);
    EXPECT_EQ(result.TP_grouped_flags.size(), 2);

    //ウィンドウごとでは両方TP
    EXPECT_TRUE(result.TP_flags[0]);
    EXPECT_TRUE(result.TP_flags[1]);

    //グループ化された結果では片方だけTP
    EXPECT_TRUE (result.TP_grouped_flags[0]);
    EXPECT_FALSE(result.TP_grouped_flags[1]);

    EXPECT_TRUE(result.found_flags[0]);

    const libcohog::EvaluationResult eval = result.to_eval();
    EXPECT_EQ(eval.nTP, 1);
    EXPECT_EQ(eval.nFP, 1);
    EXPECT_EQ(eval.nFN, 0);
}

TEST(verifier_test, notconfident_fn)
{
    std::vector<libcohog::Window> windows;  //空っぽ

    std::vector<libcohog::TruthRect> truth;
    truth += libcohog::TruthRect{cv::Rect(100, 100, 100, 200), false};

    const libcohog::EvaluationResult eval = libcohog::evaluate(windows, truth, 0, 0.5, 0, 0); //eps=0なのでグルーピングはしない
    EXPECT_EQ(eval.nFN, 0);
}


