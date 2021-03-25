float map(float val, float currMin, float currMax, float mappedMin, float mappedMax){
    float diff1 = currMax - currMin;
    float diff2 = mappedMax - mappedMin;
    float ratio  = (val - currMin) / diff1;
    float mappedVal = ratio * diff2 + mappedMax;
    return mappedVal;
}