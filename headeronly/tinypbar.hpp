#include <chrono>
#include <algorithm> // std::fill_n (todo: remove?)
#include <cmath>
#include <cstring>
#include <cstdio>

// #ifdef _WIN32
// #include <windows.h>
// #else
// #include <sys/ioctl.h>
// #include <unistd.h>
// #endif

namespace tiny {

inline int get_terminal_width() {
// #ifdef _WIN32
//     CONSOLE_SCREEN_BUFFER_INFO csbi;
//     if (GetConsoleScreenBufferInfo(GetStdHandle(STD_OUTPUT_HANDLE), &csbi)) {
//         return csbi.srWindow.Right - csbi.srWindow.Left + 1;
//     }
// #else
//     struct winsize w;
//     if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &w) == 0) {
//         return w.ws_col;
//     }
// #endif
    // Return a default value if unable to get the terminal size
    return 80;
}

template<typename T>
class pbar {
private:
    T start_, end_, step_; // idx range
    bool disable; // disable pbar
    double rate; // update freq in Hz
    std::chrono::steady_clock::time_point start_time;
    T i; // current idx value
    T n; // current iteration
    long long skip; // Skip displaying pbar
    size_t total; // total iterations
    const char* desc_; // pbar description

    // must have 9 bytes of space
    void HMS(double t, char* out) {
        const int hours = static_cast<int>(t) / 3600;
        const int minutes = (static_cast<int>(t) % 3600) / 60;
        const int seconds = static_cast<int>(t) % 60;
        
        out[0] = '0' + hours / 10;
        out[1] = '0' + hours % 10;
        out[2] = ':';
        out[3] = '0' + minutes / 10;
        out[4] = '0' + minutes % 10;
        out[5] = ':';
        out[6] = '0' + seconds / 10;
        out[7] = '0' + seconds % 10;
        out[8] = '\0'; // null terminate
    }

    // out must have 8 bytes of space
    // XXX.XXS\0 (X number, S char)
    void SI(double x, char* out) {
        if (x == 0) {
            std::strcpy(out, "0.00");
            return;
        }
        const int g = static_cast<int>(std::log10(x) / 3);
        const double scaled = x / std::pow(1000, g);
        snprintf(out, 8, "%.2f%c", scaled, " kMGTPEZY"[g]);
    }

public:
    pbar(T start, T end, T step = 1, const char* desc = "", 
         bool disable = false, double rate = 100.0)
        : start_(start), end_(end), step_(step), disable(disable), rate(rate), 
          i(start-step), n(0), skip(1ull), desc_(desc) {
        start_time = std::chrono::steady_clock::now();
        total = static_cast<size_t>((end_ - start_ + step_ - 1) / step_);
        update(0);
    }

    ~pbar() {
        update(0, true);
    }

    void update(T increment = 0, bool close = false) {
        n += increment;
        i += step_;

        if (disable || (!close && static_cast<int>(i - start_) % skip != 0)) return;

        auto now = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(now - start_time).count();
        double progress = static_cast<double>(n) / total;

        if (n / elapsed > rate && n != start_) {
            skip = static_cast<long long>(n / elapsed / rate);
        }

        const int terminal_width = get_terminal_width();
        const int bar_length = terminal_width - 50;

        char prog_bar[101] = {0};  // Assuming max bar length is 100
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

        std::printf("\r%s%3.0f%% [%s] %zu/%zu [%s<%s, %s it/s]",
                 desc_, progress * 100.0, prog_bar, static_cast<size_t>(n), total,
                 hms_time_elapsed, hms_time_remaining, si_speed);

        if (close) std::printf("\n");
    }

    class iterator {
    private:
        pbar& bar;
        T current;
    public:
        iterator(pbar& t, T start) : bar(t), current(start) {}
        iterator& operator++() {
            current += bar.step_;
            bar.update(1);
            return *this;
        }
        bool operator!=(const iterator& other) const {
            return current < other.current;
        }
        T operator*() const { return current; }
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