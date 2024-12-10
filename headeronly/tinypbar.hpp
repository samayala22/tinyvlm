#include <chrono>
#include <algorithm> // std::fill_n (todo: remove?)
#include <cmath>
#include <cstring>
#include <cstdio>

namespace tiny {
    
class pbar {
private:
    int start_, end_, step_; // idx range
    bool disable; // disable pbar
    double rate; // update freq in Hz
    std::chrono::steady_clock::time_point start_time;
    int i; // current idx value
    int n; // current iteration
    int skip; // Skip displaying pbar
    int total; // total iterations
    const char* desc_; // pbar description

    // must have 9 bytes of space
    void HMS(double t, char* out) {
        const int hours = static_cast<int>(t) / 3600;
        const int minutes = (static_cast<int>(t) % 3600) / 60;
        const int seconds = static_cast<int>(t) % 60;

        std::snprintf(out, 9, "%02d:%02d:%02d", hours, minutes, seconds);
    }

    // out must have 8 bytes of space
    // XXX.XXS\0 (X number, S char)
    void SI(double x, char* out) {
        if (x == 0) {
            std::snprintf(out, 8, "0.00");
            return;
        }
        const int g = (x==0) ? 0 : static_cast<int>(std::log10(x) / 3);
        const double scaled = x / std::pow(1000, g);
        std::snprintf(out, 8, "%.2f%c", scaled, " kMGTPEZY"[g]);
    }

public:
    pbar(int start, int end, int step = 1, const char* desc = "", 
         bool disable = false, double rate = 60.0)
        : start_(start), end_(end), step_(step), disable(disable), rate(rate), 
          i(start-step), n(0), skip(1ull), desc_(desc) {
        start_time = std::chrono::steady_clock::now();
        total = (end_ - start_ + step_ - 1) / step_;
        update(0);
    }

    ~pbar() {
        update(0, true);
    }

    void update(int increment = 0, bool close = false) {
        n += increment;
        i += step_;

        if (disable || (!close && static_cast<int>(i - start_) % skip != 0)) return;

        auto now = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(now - start_time).count(); // cast to seconds
        elapsed += 1e-6; // add 1 microsecond to avoid rounding errors
        double progress = static_cast<double>(n) / total;

        if (n / elapsed > rate && n != 0) {
            skip = static_cast<int>(n / elapsed / rate);
        }

        constexpr int bar_length = 50;

        char prog_bar[bar_length+1] = {0};  // Assuming max bar length is 100
        if (total > 0) {
            const int filled_length = static_cast<int>(bar_length * progress);
            std::fill_n(prog_bar, filled_length, '#');
            std::fill_n(prog_bar + filled_length, bar_length - filled_length, ' ');
            prog_bar[bar_length] = '\0';
        }

        char hms_time_elapsed[9], hms_time_remaining[9], si_speed[8];
        HMS(elapsed, hms_time_elapsed);
        double time_remaining = (progress > 0) ? elapsed / progress - elapsed : 0.0;
        HMS(time_remaining, hms_time_remaining);
        SI(static_cast<double>(n) / elapsed, si_speed);

        std::printf("\r%s%3.0f%% [%s] %d/%d [%s<%s, %s it/s]",
                 desc_, progress * 100.0, prog_bar, n, total,
                 hms_time_elapsed, hms_time_remaining, si_speed);

        if (close) std::printf("\n");
    }

    class iterator {
    private:
        pbar& bar;
        int current;
    public:
        iterator(pbar& t, int start) : bar(t), current(start) {}
        [[nodiscard]] iterator& operator++() {
            current += bar.step_;
            bar.update(1);
            return *this;
        }
        [[nodiscard]] bool operator!=(const iterator& other) const {
            return current < other.current;
        }
        [[nodiscard]] int operator*() const { return current; }
    };

    iterator begin() { return iterator(*this, start_); }
    iterator end() { return iterator(*this, end_); }
};

}  // namespace tiny

// #include <thread>

// using namespace tiny;

// int main() {
//     volatile float x = 3.f;
//     for (auto i : pbar<int>(0, 30, 1)) {
//         x = x*2+std::log(x*x*x*std::exp(1.f/x)+2.0);
//     }
//     std::printf("%f \n", x);
//     return 0;
// }