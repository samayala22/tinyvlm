#pragma once

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace tiny {

    template<typename T>
    class Interpolator {
        public:
            Interpolator(const std::vector<T>& x, const std::vector<T>& y) : m_x(x), m_y(y) {};
            virtual ~Interpolator() = default;
            virtual const T operator()(const T x) const = 0;
        protected:
            const std::vector<T>& m_x;
            const std::vector<T>& m_y;
    };

    template<typename T>
    class AkimaInterpolator : public Interpolator<T> {
        // https://en.wikipedia.org/wiki/Akima_spline
        public:
            AkimaInterpolator(const std::vector<T>& x, const std::vector<T>& y) : Interpolator<T>(x, y) {
                // todo: add verification for size x == size of y
                size_t n = x.size();
                m_m.resize(n - 1);
                m_s.resize(n);
                // Compute linear slopes
                for (size_t i = 0; i < n - 1; ++i) {
                    T dx = x[i + 1] - x[i];
                    T dy = y[i + 1] - y[i];
                    m_m[i] = dy / dx;
                }
                // Compute spline coefficients
                m_s[0] = m_m[0];
                m_s[1] = 0.5f * (m_m[0] + m_m[1]);
                for (size_t i = 2; i < n-2; ++i) {
                    T d1 = std::abs(m_m[i+1] - m_m[i]);
                    T d2 = std::abs(m_m[i-1] - m_m[i-2]);
                    if (d1 + d2 > 0.0f) {
                        m_s[i] = (d1 * m_m[i-1] + d2 * m_m[i]) / (d1 + d2);
                    } else {
                        m_s[i] = 0.5f * (m_m[i] + m_m[i-1]);
                    }
                }
                m_s[n-2] = 0.5f * (m_m[n-3] + m_m[n-2]);
                m_s[n-1] = m_m[n-2];
            }
            ~AkimaInterpolator() = default;
            const T operator()(const T x) const override {
                auto it = std::lower_bound(this->m_x.begin(), this->m_x.end(), x);
                // If x is outside of the range of m_x, fall back to linear interpolation using last slope
                if (it == this->m_x.begin()) {
                    return this->m_y[0] - m_m[0] * (this->m_x[0] - x);
                } else if (it == this->m_x.end()) {
                    return this->m_y.back() + m_m.back() * (x - this->m_x.back());
                }
                
                size_t idx = it - this->m_x.begin() - 1;
                T dx = this->m_x[idx+1] - this->m_x[idx];
                T dxx = x - this->m_x[idx];
                T a = this->m_y[idx];
                T b = m_s[idx];
                T c = (3*m_m[idx] - 2*m_s[idx] - m_s[idx+1]) / dx;
                T d = (m_s[idx] + m_s[idx+1] - 2*m_m[idx]) / (dx * dx);
                return a + b*dxx + c*dxx*dxx + d*dxx*dxx*dxx;
            };

        private:
            const std::vector<T> m_m; // segment slopes (per segment)
            const std::vector<T> m_s; // spline coefficients (per point)
    };

    template<typename T>
    class LinearInterpolator : public Interpolator<T> {
        public:
            LinearInterpolator(const std::vector<T>& x, const std::vector<T>& y) : Interpolator<T>(x, y) {
                // todo: add verification for size x == size of y
                size_t n = x.size();
                m_m.resize(n - 1);
                // Compute linear slopes
                for (size_t i = 0; i < n - 1; ++i) {
                    T dx = x[i + 1] - x[i];
                    T dy = y[i + 1] - y[i];
                    m_m[i] = dy / dx;
                }
            }
            ~LinearInterpolator() = default;
            const T operator()(const T x) const override {
                auto it = std::lower_bound(this->m_x.begin(), this->m_x.end(), x);
                // If x is outside of the range of m_x, fall back to linear interpolation using last slope
                if (it == this->m_x.begin()) {
                    return this->m_y[0] - m_m[0] * (this->m_x[0] - x);
                } else if (it == this->m_x.end()) {
                    return this->m_y.back() + m_m.back() * (x - this->m_x.back());
                }
                                
                size_t idx = it - this->m_x.begin() - 1;
                return this->m_y[idx] + m_m[idx] * (x - this->m_x[idx]);
            };

        private:
            const std::vector<T> m_m; // segment slopes (per segment)
    };
} // namespace tiny