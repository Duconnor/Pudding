#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>
#include <pudding/clustering.h>

TEST_CASE ("Test kmeans", "[kmeans]") {
    int res = pudding::kmeans();

    REQUIRE( res == 1 );
}