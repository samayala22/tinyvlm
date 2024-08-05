#pragma once

#include <cstdint>
#include <cassert>
#include <type_traits>
#include <vector>

namespace vlm {

enum class MemoryLocation {
    Device, Host, HostDevice
};

enum class MemoryTransfer {
    HostToDevice, DeviceToHost, HostToHost, DeviceToDevice
};

class Memory {
    public:
        Memory(bool unified) : _unified(unified) {}
        ~Memory() = default;
        virtual void* alloc(MemoryLocation location, std::size_t size) const = 0;
        virtual void free(MemoryLocation location, void* ptr) const = 0;
        virtual void copy(MemoryTransfer transfer, void* dst, const void* src, std::size_t size) const = 0;
        virtual void fill_f32(MemoryLocation location, float* ptr, float value, std::size_t size) const = 0;
        bool is_unified() const { return _unified; }
    private:
        const bool _unified; // host and device memory are the same
};

template<typename T>
struct LayoutTraits {
    static_assert(std::is_same_v<decltype(std::declval<T>().size()), std::size_t>,
                  "Layout must have a size() method returning std::size_t");
};

template<typename T, class Layout>
class View {
    static_assert(std::is_same_v<decltype(LayoutTraits<Layout>{}), LayoutTraits<Layout>>, "Incorrect Layout");

    public:
        T* ptr = nullptr; // required
        Layout layout;

        View() = default;
        View(T* ptr, const Layout& layout) : ptr(ptr), layout(layout) {}
        ~View() = default;

        std::size_t size() const { return layout.size(); }
        std::size_t size_bytes() const { return layout.size() * sizeof(T); }
        inline T& operator[](std::size_t index) {
            assert(ptr != nullptr && "Pointer is null");
            assert(index < layout.size() && "Index out of bounds");
            return ptr[index];
        }
        inline const T& operator[](std::size_t index) const {
            assert(ptr != nullptr && "Pointer is null");
            assert(index < layout.size() && "Index out of bounds");
            return ptr[index];
        }
};

template<typename T, MemoryLocation Location, class Layout>
class Buffer {
    using is_host = std::bool_constant<Location == MemoryLocation::Host || Location == MemoryLocation::HostDevice>;
    using is_device = std::bool_constant<Location == MemoryLocation::Device || Location == MemoryLocation::HostDevice>;

public:
    Buffer() = delete;
    explicit Buffer(const Memory& memory, const Layout& layout) : memory(memory) { alloc(layout); }
    explicit Buffer(const Memory& memory) : memory(memory) {}
    ~Buffer() { dealloc(); }

    View<T, Layout>& h_view() {
        static_assert(is_host::value);
        return _host;
    }
    View<T, Layout>& d_view() {
        static_assert(is_device::value);
        return _device;
    }

    void to_device() {
        static_assert(is_host::value && is_device::value);
        if (!memory.is_unified())
            memory.copy(MemoryTransfer::HostToDevice, _device.ptr, _host.ptr, size_bytes());
    }

    void to_host() {
        static_assert(is_host::value && is_device::value);
        if (!memory.is_unified()) 
            memory.copy(MemoryTransfer::DeviceToHost, _host(), _device(), size_bytes());
    }

    std::size_t size() const { return _host.size(); }
    std::size_t size_bytes() const { return _host.size() * sizeof(T); }

    void alloc(const Layout& layout) {
        _host.layout = layout;
        _device.layout = layout;

        if (memory.is_unified()) {
            _host.ptr = static_cast<T*>(memory.alloc(MemoryLocation::Host, size_bytes()));
            _device.ptr = _host.ptr;
        } else {
            if constexpr (is_host::value) _host.ptr = static_cast<T*>(memory.alloc(MemoryLocation::Host, size_bytes()));
            if constexpr (is_device::value) _device.ptr = static_cast<T*>(memory.alloc(MemoryLocation::Device, size_bytes()));
        }
    }

    void dealloc() {
        if (memory.is_unified()) {
            memory.free(MemoryLocation::Host, _host.ptr);
        } else {
            if constexpr (is_host::value) memory.free(MemoryLocation::Host, _host.ptr);
            if constexpr (is_device::value) memory.free(MemoryLocation::Device, _device.ptr);
        }
    }

    const Memory& memory;

private:
    View<T, Layout> _host;
    View<T, Layout> _device;

    Buffer(Buffer&&) = delete;
    Buffer(const Buffer&) = delete;
    Buffer& operator=(Buffer&&) = delete;
    Buffer& operator=(const Buffer&) = delete;
};

struct SurfaceDims { // ns major
    uint64_t nc;
    uint64_t ns;
    uint64_t offset;
    uint64_t size() const {return nc * ns;}
};

class SingleSurface {
    public:

        SingleSurface() = default;
        SingleSurface(const SurfaceDims& surface, uint64_t stride, uint32_t dim) { construct(surface, stride,dim); }
        ~SingleSurface() = default;

        void construct(const SurfaceDims& surface, uint64_t stride, uint32_t dim) {
            _surface = surface;
            _stride = stride;
            _dim = dim;
        }

        std::size_t size() const {return dim() * stride(); } // required
        const SurfaceDims& surface() const {return _surface; }
        uint64_t stride() const {return _stride; }
        uint64_t dim() const {return _dim; }

    private:
        SurfaceDims _surface;
        uint64_t _stride = 0;
        uint32_t _dim = 1;
};

class MultiSurface {
    public:

        MultiSurface() = default;
        MultiSurface(const std::vector<SurfaceDims>& surfaces, uint32_t dim) { construct(surfaces, dim); }
        ~MultiSurface() = default;

        void construct(const std::vector<SurfaceDims>& surfaces, uint32_t dim) {
            _surfaces = surfaces;
            _stride = surfaces.back().offset + surfaces.back().size();
            _dim = dim;
        }

        uint64_t operator()(uint32_t wing_id, uint32_t dim) { // returns start of the specific wing buffer
            assert(wing_id < _surfaces.size());
            assert(dim < _dim);
            return _surfaces[wing_id].offset + dim * _stride;
        }

        std::size_t size() const {return dims() * stride(); } // required
        const std::vector<SurfaceDims>& surfaces() const {return _surfaces; }
        const SurfaceDims& surface(uint32_t wing_id) const {return _surfaces[wing_id]; }
        uint64_t stride() const {return _stride; }
        uint32_t dims() const {return _dim; }

        uint64_t nc(uint32_t wing_id) const {return _surfaces[wing_id].nc; }
        uint64_t ns(uint32_t wing_id) const {return _surfaces[wing_id].ns; }
        uint64_t offset(uint32_t wing_id) const {return _surfaces[wing_id].offset; }
        
        template<typename T>
        View<T, SingleSurface> subview(T* ptr, uint32_t wing_id, uint64_t i, uint64_t m, uint64_t j, uint64_t n) const {
            assert(wing_id < _surfaces.size());
            assert(i + m <= nc(wing_id));
            assert(j + n <= ns(wing_id));

            return {
                ptr + offset(wing_id) + i * ns(wing_id) + j,
                SingleSurface{SurfaceDims{m,n,0}, stride(), dims()}
            };
        }

    private:
        std::vector<SurfaceDims> _surfaces;
        uint64_t _stride = 0;
        uint32_t _dim = 1;
};

enum class MatrixLayout {
    RowMajor, ColMajor
};

template<MatrixLayout Layout>
class Matrix {
    public:
        Matrix() = default;
        Matrix(uint64_t m, uint64_t n, uint64_t stride) { construct(m, n, stride); }
        void construct(uint64_t m, uint64_t n, uint64_t stride) {
            _m = m;
            _n = n;
            _stride = stride;
        }

        std::size_t size() const {return _m * _n; }
        uint64_t m() const {return _m; }
        uint64_t n() const {return _n; }
        uint64_t stride() const {return _stride; }
        constexpr MatrixLayout layout() const {return Layout; }

    private:
        uint64_t _m = 0;
        uint64_t _n = 0;
        uint64_t _stride = 0; // leading dimension
};

} // namespace vlm

// #include <algorithm>
// #include <cstring>
// #include <memory>

// using namespace vlm;

// class MemoryCPU final : public Memory {
//     public:
//         MemoryCPU() : Memory(true) {}
//         ~MemoryCPU() = default;
//         void* alloc(MemoryLocation location, std::size_t size) const override {return std::malloc(size);}
//         void free(MemoryLocation location, void* ptr) const override {std::free(ptr);}
//         void copy(MemoryTransfer transfer, void* dst, const void* src, std::size_t size) const override {std::memcpy(dst, src, size);}
//         void fill_f32(MemoryLocation location, float* ptr, float value, std::size_t size) const override {std::fill(ptr, ptr + size, value);}
// };

// int main() {
//     std::unique_ptr<Memory> memory = std::make_unique<MemoryCPU>(); // memory manager
//     Buffer<float, MemoryLocation::Host, MultiSurface> buffer{*memory}; // owning host/device buffer with a view type

//     MultiSurface layout{{{5,5,0}},3}; // dimensions of the view following the custom layout

//     buffer.alloc(layout); // allocate and initialize the views in the buffer.

//     assert(buffer.size() == layout.size()); // linear size of the allocated buffer must match the size of the view
// }