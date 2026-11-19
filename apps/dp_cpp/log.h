#ifndef __LOG__
#define __LOG__

#include <cstdio>
// #define LOGD(fmt, ...)  printf("DEBUG %s:%d " fmt "\n", __FILE__, __LINE__, ##__VA_ARGS__)
// #define LOGI(fmt, ...)  printf("INFO %s:%d " fmt "\n", __FILE__, __LINE__, ##__VA_ARGS__)
// #define LOGW(fmt, ...)  printf("WARN %s:%d " fmt "\n", __FILE__, __LINE__, ##__VA_ARGS__)
// #define LOGE(fmt, ...)  printf("ERROR %s:%d " fmt "\n", __FILE__, __LINE__, ##__VA_ARGS__)

#define LOGD(fmt, ...)  printf("DEBUG " fmt "\n", ##__VA_ARGS__)
#define LOGI(fmt, ...)  printf("INFO  " fmt "\n", ##__VA_ARGS__)
#define LOGW(fmt, ...)  printf("WARN  " fmt "\n", ##__VA_ARGS__)
#define LOGE(fmt, ...)  printf("ERROR " fmt "\n", ##__VA_ARGS__)

#endif // __LOG__