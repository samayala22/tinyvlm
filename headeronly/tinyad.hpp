#pragma once

#include <cmath>

namespace fwd {
template<typename T>
class Number {
public:

    Number() = default;
    Number(const Number<T> &rhs) = default;
    Number(Number<T> &&rhs) = default;
    ~Number() = default;

    Number(const T value)
        : value_(value)
    { }

    Number(const T value, const T derivative)
        : value_(value), derivative_(derivative)
    { }

    T val() const
    {
        return value_;
    }

    T grad() const
    {
        return derivative_;
    }

    Number<T> &operator=(const Number<T> &rhs) & = default;
    Number<T> &operator=(Number<T> &&rhs) && = default;

    Number<T> &operator=(const T rhs) &
    {
        value_ = rhs;
        derivative_ = 0;

        return *this;
    }

    Number<T> &operator+=(const Number<T> &rhs)
    {
        value_ += rhs.value_;
        derivative_ += rhs.derivative_;

        return *this;
    }

    Number<T> &operator*=(const Number<T> &rhs)
    {
        derivative_ = rhs.value_ * derivative_  + value_ * rhs.derivative_;
        value_ *= rhs.value_;

        return *this;
    }

    Number<T> &operator-=(const Number<T> &rhs)
    {
        *this += -rhs;
        return *this;
    }

    Number<T> &operator/=(const Number<T> &rhs)
    {
        derivative_ = (derivative_ * rhs.value_ - rhs.derivative_ * value_) / (rhs.value_ * rhs.value_);
        value_ /= rhs.value_;

        return *this;
    }

    Number<T> operator-() const
    {
        return Number<T>(-value_, -derivative_);
    }

    explicit operator T() const
    {
        return val();
    }

private:
    T value_{0};
    T derivative_{0};
};

template<typename T>
inline Number<T> operator+(const Number<T> &lhs, const Number<T> &rhs)
{
    auto result = lhs;
    result += rhs;
    return result;
}

template<typename T>
inline Number<T> operator-(const Number<T> &lhs, const Number<T> &rhs)
{
    auto result = lhs;
    result -= rhs;
    return result;
}

template<typename T>
inline Number<T> operator/(const Number<T> &lhs, const Number<T> &rhs)
{
    auto result = lhs;
    result /= rhs;
    return result;
}

template<typename T>
inline Number<T> operator*(const Number<T> &lhs, const Number<T> &rhs)
{
    auto result = lhs;
    result *= rhs;
    return result;
}

template<typename T>
inline bool operator==(const Number<T> &lhs, const Number<T> &rhs)
{
    return lhs.val() == rhs.val();
}

template<typename T>
inline bool operator!=(const Number<T> &lhs, const Number<T> &rhs)
{
    return lhs.val() != rhs.val();
}

template<typename T>
inline bool operator<(const Number<T> &lhs, const Number<T> &rhs)
{
    return lhs.val() < rhs.val();
}

template<typename T>
inline bool operator<=(const Number<T> &lhs, const Number<T> &rhs)
{
    return lhs.val() <= rhs.val();
}

template<typename T>
inline bool operator>(const Number<T> &lhs, const Number<T> &rhs)
{
    return lhs.val() > rhs.val();
}

template<typename T>
inline bool operator>=(const Number<T> &lhs, const Number<T> &rhs)
{
    return lhs.val() >= rhs.val();
}

template<typename T>
inline T &operator+=(T &lhs, const Number<T> &rhs)
{
    lhs += rhs.val();
    return lhs;
}

template<typename T>
inline Number<T> operator+(const T lhs, const Number<T> &rhs)
{
    return Number<T>(lhs) + rhs;
}

template<typename T>
inline T &operator-=(T &lhs, const Number<T> &rhs)
{
    lhs -= rhs.val();
    return lhs;
}

template<typename T>
inline Number<T> operator-(const T lhs, const Number<T> &rhs)
{
    return Number<T>(lhs) - rhs;
}

template<typename T>
inline T &operator*=(T &lhs, const Number<T> &rhs)
{
    lhs *= rhs.val();
    return lhs;
}

template<typename T>
inline Number<T> operator*(const T lhs, const Number<T> &rhs)
{
    return Number<T>(lhs) * rhs;
}

template<typename T>
inline T &operator/=(T &lhs, const Number<T> &rhs)
{
    lhs /= rhs.val();
    return lhs;
}

template<typename T>
inline Number<T> operator/(const T lhs, const Number<T> &rhs)
{
    return Number<T>(lhs) / rhs;
}

template<typename T>
inline Number<T> sin(const Number<T> &val)
{
    T value = std::sin(val.val());
    T derivative = val.grad() * std::cos(val.val());
    return Number<T>(value, derivative);
}

template<typename T>
inline Number<T> asin(const Number<T> &val)
{
    T value = std::asin(val.val());
    T derivative = val.grad() * 1 / std::sqrt(1 - val.val() * val.val());
    return Number<T>(value, derivative);
}

template<typename T>
inline Number<T> cos(const Number<T> &val)
{
    T value = std::cos(val.val());
    T derivative = val.grad() * -std::sin(val.val());
    return Number<T>(value, derivative);
}

template<typename T>
inline Number<T> acos(const Number<T> &val)
{
    T value = std::acos(val.val());
    T derivative = val.grad() * -1 / std::sqrt(1 - val.val() * val.val());
    return Number<T>(value, derivative);
}

template<typename T>
inline Number<T> tan(const Number<T> &val)
{
    T value = std::tan(val.val());
    T c = std::cos(val.val());
    T derivative = val.grad() * 1 / (c * c);
    return Number<T>(value, derivative);
}

template<typename T>
inline Number<T> atan(const Number<T> &val)
{
    T value = std::atan(val.val());
    T derivative = val.grad() * 1 / (1 + val.val() * val.val());

    return Number<T>(value, derivative);
}

template<typename T>
inline Number<T> atan2(const Number<T> &y, const Number<T> &x)
{
    T value = std::atan2(y.val(), x.val());
    T denom = x.val() * x.val() + y.val() * y.val();
    T derivative = x.grad() * y.val() / denom +
        y.grad() * x.val() / denom;

    return Number<T>(value, derivative);
}

template<typename T>
inline Number<T> exp(const Number<T> &val)
{
    T value = std::exp(val.val());
    T derivative = val.grad() * std::exp(val.val());
    return Number<T>(value, derivative);
}

template<typename T>
inline Number<T> pow(const Number<T> &val, const T exponent)
{
    T value = std::pow(val.val(), exponent);
    T derivative = val.grad() * exponent * std::pow(val.val(), exponent - 1);
    return Number<T>(value, derivative);
}

template<typename T>
inline Number<T> pow(const Number<T> &val, const int exponent)
{
    T value = std::pow(val.val(), exponent);
    T derivative = val.grad() * exponent * std::pow(val.val(), exponent - 1);
    return Number<T>(value, derivative);
}

template<typename T>
inline Number<T> sqrt(const Number<T> &val)
{
    T value = std::sqrt(val.val());
    T derivative = val.grad() / (2 * value);
    return Number<T>(value, derivative);
}

template<typename T>
inline Number<T> conj(const Number<T> &val)
{
    return val;
}

template<typename T>
inline Number<T> real(const Number<T> &val)
{
    return val;
}

template<typename T>
inline Number<T> imag(const Number<T> &)
{
    return Number<T>(0, 0);
}

template<typename T>
inline Number<T> abs(const Number<T> &val)
{
    return Number<T>(std::abs(val.val()), std::abs(val.grad()));
}

template<typename T>
inline Number<T> abs2(const Number<T> &val)
{
    return val * val;
}

template<typename T>
inline Number<T> log(const Number<T> &val)
{
    T value = std::log(val.val());
    T derivative = val.grad() * 1 / val.val();
    return Number<T>(value, derivative);
}

template<typename T>
inline Number<T> log2(const Number<T> &val)
{
    T value = std::log2(val.val());
    T derivative = val.grad() * 1 / (val.val() * static_cast<T>(0.6931471805599453));
    return Number<T>(value, derivative);
}

template<typename T>
inline bool isfinite(const Number<T> &val)
{
    return std::isfinite(val.val());
}

typedef Number<double> Double;
typedef Number<float> Float;
} // namespace fwd