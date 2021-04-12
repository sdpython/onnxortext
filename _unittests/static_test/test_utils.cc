#include "gtest/gtest.h"
#include "utils/string_utils.h"

TEST(utils, make_string) {
	std::string res = MakeString("a", "b", 0);
	EXPECT_EQ(res, "ab0");
}

TEST(utils, split_string) {
	auto result = SplitString("a b c d e f", " ");
	EXPECT_EQ(result.size(), 6);

	// contain a space
	result = SplitString("ab cd ef  gh", " ");
	EXPECT_EQ(result.size(), 5);

	// contain a space
	result = SplitString("ab cd ef  gh", " ", true);
	EXPECT_EQ(result.size(), 4);

	result = SplitString("abcd", " ");
	EXPECT_EQ(result.size(), 1);
	EXPECT_EQ(result[0], "abcd");

	result = SplitString("eabc\nasbd", "\n");
	EXPECT_EQ(result.size(), 2);
	EXPECT_EQ(result[0], "eabc");

	result = SplitString("a\nb\n", "\n");
	EXPECT_EQ(result.size(), 2);
	EXPECT_EQ(result[1], "b");

	// two seps
	result = SplitString("ea,bc\nas,bd", ",\n");
	EXPECT_EQ(result.size(), 4);
	EXPECT_EQ(result[1], "bc");
}

TEST(utils, utf8) {
	std::vector<std::string> srcs = { "abc", "ABCé", "中文" };
	std::vector<std::string> lowered = { "abc", "abcé", "中文" };
	for (size_t i = 0; i < srcs.size(); ++i) {
		std::string lower;
		lower = srcs[i];
		std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
		EXPECT_EQ(lowered[i], lower);
	}
}