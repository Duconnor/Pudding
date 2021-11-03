#ifndef PUDDING_HELPER_H
#define PUDDING_HELPER_H

int twoDimIndexToOneDim(int i, int j, int m, int n) {
    return i * n + j;
}

#endif