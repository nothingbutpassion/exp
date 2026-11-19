#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>
#include <string>

#ifndef M_PI
#define M_PI       3.14159265358979323846   // pi
#endif

// Generate a projector auto-focus calibration pattern (1920x1080)
// Combines Siemens star, checkerboard elements, and resolution targets.
void generatePattern(const std::string& output) {
    const int W = 1920;
    const int H = 1080;

    // Create gray background image (128 = gray)
    cv::Mat img(H, W, CV_8UC1, cv::Scalar(128));

    // ── 1. Checkerboard corners ──────────────────────────────────────
    int check_size = 40;
    std::vector<cv::Rect> corners = {
        cv::Rect(0, 0, W / 3, H / 3),
        cv::Rect(W - W / 3, 0, W, H / 3),
        cv::Rect(0, H - H / 3, W / 3, H),
        cv::Rect(W - W / 3, H - H / 3, W, H)
    };

    for (const auto& corner : corners) {
        for (int y = corner.y; y < corner.y + corner.height; y += check_size) {
            for (int x = corner.x; x < corner.x + corner.width; x += check_size) {
                if (((x / check_size) + (y / check_size)) % 2 == 0) {
                    cv::rectangle(img, cv::Point(x, y), cv::Point(x + check_size - 1, y + check_size - 1),
                                 cv::Scalar(0), cv::FILLED);
                }
            }
        }
    }

    // ── 2. Siemens star (center) ─────────────────────────────────────
    int cx = W / 2;
    int cy = H / 2;
    int outer_r = 380;
    int inner_r = 8;
    int n_sectors = 72; // 36 white + 36 black

    for (int i = 0; i < n_sectors; i++) {
        double angle_start = 2 * M_PI * i / n_sectors;
        double angle_end = 2 * M_PI * (i + 1) / n_sectors;
        int fill = (i % 2 == 0) ? 255 : 0;

        // Create polygon points for sector (outer to inner)
        std::vector<cv::Point> points;

        // Add center point
        points.push_back(cv::Point(cx, cy));

        // Add outer points around the sector
        int steps = 15;
        for (int j = 0; j <= steps; j++) {
            double a = angle_start + (angle_end - angle_start) * j / steps;
            points.push_back(cv::Point(int(cx + outer_r * cos(a)), int(cy + outer_r * sin(a))));
        }

        // Add inner points (reverse order to create proper polygon)
        for (int j = steps; j >= 0; j--) {
            double a = angle_start + (angle_end - angle_start) * j / steps;
            points.push_back(cv::Point(int(cx + inner_r * cos(a)), int(cy + inner_r * sin(a))));
        }

        // Fill the sector with the correct color
        std::vector<std::vector<cv::Point>> polygon = {points};
        cv::fillPoly(img, polygon, cv::Scalar(fill));
    }

    // white ring around star
    cv::circle(img, cv::Point(cx, cy), outer_r, cv::Scalar(255), 2);
    // black ring
    cv::circle(img, cv::Point(cx, cy), outer_r + 10, cv::Scalar(0), 2);

    // ── 3. Vertical / horizontal resolution bars ─────────────────────
    int bar_h = 15;
    int bar_gap = 4;
    std::vector<int> freqs = {1, 2, 4, 8, 16, 32};

    // horizontal bars (vertical resolution)
    int x0 = W / 2 - 350;
    int y0 = H - 120;
    for (size_t fi = 0; fi < freqs.size(); fi++) {
        int f = freqs[fi];
        for (int col = 0; col < 300; col += f * 2) {
            cv::rectangle(img, cv::Point(x0 + col, y0 + static_cast<int>(fi) * (bar_h + bar_gap)),
                         cv::Point(x0 + col + f - 1, y0 + static_cast<int>(fi) * (bar_h + bar_gap) + bar_h - 1),
                         cv::Scalar(0), cv::FILLED);
        }
    }

    // vertical bars (horizontal resolution)
    x0 = W / 2 + 50;
    for (size_t fi = 0; fi < freqs.size(); fi++) {
        int f = freqs[fi];
        for (int row = 0; row < 300; row += f * 2) {
            cv::rectangle(img, cv::Point(x0 + static_cast<int>(fi) * (bar_h + bar_gap), y0 + row),
                         cv::Point(x0 + static_cast<int>(fi) * (bar_h + bar_gap) + bar_h - 1, y0 + row + f - 1),
                         cv::Scalar(0), cv::FILLED);
        }
    }

    // ── 4. Crosshair ─────────────────────────────────────────────────
    cv::line(img, cv::Point(cx - 20, cy), cv::Point(cx + 20, cy), cv::Scalar(255), 2);
    cv::line(img, cv::Point(cx, cy - 20), cv::Point(cx, cy + 20), cv::Scalar(255), 2);
    cv::line(img, cv::Point(cx - 10, cy), cv::Point(cx + 10, cy), cv::Scalar(0), 1);
    cv::line(img, cv::Point(cx, cy - 10), cv::Point(cx, cy + 10), cv::Scalar(0), 1);

    // ── 5. Resolution circles ────────────────────────────────────────
    std::vector<int> circles = {40, 80, 120, 160, 200, 240, 280, 320, 360};
    for (int r : circles) {
        cv::circle(img, cv::Point(cx, cy), r, cv::Scalar(64), 1);
    }

    // ── 6. Edge targets (high-contrast rectangles around star) ───────
    std::vector<cv::Point> placements = {
        cv::Point(W / 4 - 80, H / 4 - 60),
        cv::Point(W * 3 / 4 - 80, H / 4 - 60),
        cv::Point(W / 4 - 80, H * 3 / 4 - 60),
        cv::Point(W * 3 / 4 - 80, H * 3 / 4 - 60)
    };

    for (const auto& point : placements) {
        // Outer black rectangle with white outline
        cv::rectangle(img, point, cv::Point(point.x + 160, point.y + 120),
                     cv::Scalar(0), cv::FILLED);
        // White outline around the black rectangle
        cv::rectangle(img, point, cv::Point(point.x + 160, point.y + 120),
                     cv::Scalar(255), 2);  // White outline with thickness 2

        // inner white square
        cv::rectangle(img, cv::Point(point.x + 40, point.y + 30),
                     cv::Point(point.x + 120, point.y + 90), cv::Scalar(255), cv::FILLED);
        // diagonal lines with consistent thickness
        cv::line(img, cv::Point(point.x, point.y), cv::Point(point.x + 160, point.y + 120), cv::Scalar(255), 2);
        cv::line(img, cv::Point(point.x + 160, point.y), cv::Point(point.x, point.y + 120), cv::Scalar(255), 2);
        cv::line(img, cv::Point(point.x + 40, point.y + 30), cv::Point(point.x + 120, point.y + 90), cv::Scalar(0), 2);
        cv::line(img, cv::Point(point.x + 120, point.y + 30), cv::Point(point.x + 40, point.y + 90), cv::Scalar(0), 2);
    }

    // ── 7. Text label ─────────────────────────────────────────────────
    // Add text labels similar to the Python version with matching colors
    int fontFace = cv::FONT_HERSHEY_SIMPLEX;
    int thickness = 2;

    // Top label - main title (black like Python)
    cv::putText(img, "PROJECTOR AUTO-FOCUS TARGET", cv::Point(W/2 - 180, 40), fontFace, 1.0, cv::Scalar(0, 0, 0), thickness, cv::LINE_AA);

    // Middle label - description (dark gray like Python - fill=64)
    cv::putText(img, "Siemens Star + Checkerboard + Resolution Bars", cv::Point(W/2 - 200, 80), fontFace, 0.6, cv::Scalar(64, 64, 64), thickness, cv::LINE_AA);

    // Bottom label - resolution info (light gray like Python - fill=192)
    cv::putText(img, "1920 x 1080", cv::Point(W/2 - 42, H - 20), fontFace, 0.6, cv::Scalar(192, 192, 192), thickness, cv::LINE_AA);

    // Save the image
    cv::imwrite(output, img);
    std::cout << "Saved: " << output << " (" << img.cols << "x" << img.rows << ")" << std::endl;
}

int main(int argc, char* argv[]) {
    std::string output = "autofocus_target.png";
    // Check if user specified an output path
    if (argc > 1) {
        output = argv[1];
    }
    generatePattern(output);
    return 0;
}