#pragma once

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace tiny {

    template<typename T>
    class Interpolator {
        public:
            Interpolator() = default;
            virtual ~Interpolator() = default;
            virtual const T operator()(T x) const = 0;
            virtual void set_data(const std::vector<T>& x, const std::vector<T>& y) = 0;
        protected:
            std::vector<T> m_x;
            std::vector<T> m_y;
    };

    template<typename T>
    class AkimaInterpolator : public Interpolator<T> {
        // https://en.wikipedia.org/wiki/Akima_spline
        public:
            AkimaInterpolator() = default;
            ~AkimaInterpolator() = default;
            const T operator()(T x) const override {
                auto it = std::lower_bound(this->m_x.begin(), this->m_x.end(), x);
                if (it == this->m_x.end() || it == this->m_x.begin()) throw std::runtime_error("x out of range");
                
                size_t idx = it - this->m_x.begin() - 1;
                T dx = this->m_x[idx+1] - this->m_x[idx];
                T dxx = x - this->m_x[idx];
                T a = this->m_y[idx];
                T b = m_s[idx];
                T c = (3*m_m[idx] - 2*m_s[idx] - m_s[idx+1]) / dx;
                T d = (m_s[idx] + m_s[idx+1] - 2*m_m[idx]) / (dx * dx);
                return a + b*dxx + c*dxx*dxx + d*dxx*dxx*dxx;
            };

            void set_data(const std::vector<T>& x, const std::vector<T>& y) override {
                // todo: add verification for size x == size of y
                size_t n = x.size();
                this->m_x = x; // deep copy
                this->m_y = y; // deep copy
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

        private:
            std::vector<T> m_m; // segment slopes (per segment)
            std::vector<T> m_s; // spline coefficients (per point)
    };
} // namespace tiny