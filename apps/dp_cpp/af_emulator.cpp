// af_emulator.cpp
//
// Command-line entry point for the dual-projector autofocus emulator. Parses
// the options, loads or generates the test pattern and hands it to
// AfEmulator (see autofocus_emulator.h).
//
// Usage:
//   af_emulator [options]
//     --pattern <path>   test pattern image (default: generated grid pattern)
//     --camera <index>   capture device (default: 1; negative = emulated)
//     --no-preview       don't open the camera preview window
//     -h, --help         show this help

#include "autofocus_emulator.h"
#include "autofocus_utils.h"
#include "log.h"

#include <cstdio>
#include <string>

namespace {

const cv::Size kPatternSize(1920, 1080);

enum class ParseResult { Ok, ExitSuccess, Error };

void printUsage(const char* program) {
    std::printf(
        "Usage: %s [options]\n"
        "  --pattern <path>  test pattern image (default: generated grid pattern)\n"
        "  --camera <index>  capture device (default: 1; negative = emulated)\n"
        "  --no-preview      do not open the camera preview window\n"
        "  -h, --help        show this help\n",
        program);
}

ParseResult parseOptions(int argc, char* argv[], Options& options) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--pattern" && i + 1 < argc) {
            options.patternPath = argv[++i];
        } else if (arg == "--camera" && i + 1 < argc) {
            options.cameraIndex = std::stoi(argv[++i]);
        } else if (arg == "--no-preview") {
            options.preview = false;
        } else if (arg == "-h" || arg == "--help") {
            printUsage(argv[0]);
            return ParseResult::ExitSuccess;
        } else {
            LOGE("unknown argument: %s", arg.c_str());
            printUsage(argv[0]);
            return ParseResult::Error;
        }
    }
    return ParseResult::Ok;
}

} // namespace

int main(int argc, char* argv[]) {
    Options options;
    const ParseResult parsed = parseOptions(argc, argv, options);
    if (parsed == ParseResult::ExitSuccess)
        return 0;
    if (parsed == ParseResult::Error)
        return 1;

    cv::Mat pattern;
    if (options.patternPath.empty()) {
        LOGI("no pattern image given, using the generated grid pattern");
        pattern = AfUtils::gridPattern(kPatternSize.width, kPatternSize.height,
                                       40, 30, 8);
    } else {
        pattern = AfUtils::imagePattern(options.patternPath, kPatternSize.width,
                                        kPatternSize.height);
    }
    if (pattern.empty()) {
        LOGE("failed to load pattern: %s", options.patternPath.c_str());
        return 1;
    }

    const cv::Rect roi = {160, 120, 320, 240};
    AfEmulator emulator(options);
    if (!emulator.init(pattern, roi))
        return 1;
    emulator.run();
    return 0;
}