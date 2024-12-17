#include "vlm_memory.hpp"
#include "tinyvtu.hpp"

namespace vlm {

template<typename T>
class TensorViewAccessor : public tiny::VtuDataAccessor<T> {
    TensorView<T, 2, Location::Host> m_view;
public:
    TensorViewAccessor(const TensorView<T, 2, Location::Host>& view) : m_view(view) {}
    ~TensorViewAccessor() = default;
    T& operator[](std::int64_t i) const override {
        return m_view(i/m_view.shape(1), i%m_view.shape(1));
    }
    std::int64_t size() const override { return m_view.size(); }
    std::int32_t components() const override { return m_view.shape(1); }
};

} // namespace vlm