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
        virtual ~Memory() = default;
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

        std::size_t size() const { return layout.size(); }
        std::size_t size_bytes() const { return layout.size() * sizeof(T); }
        T& operator[](std::size_t index) {
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

    View<T, Layout>& h_ptr() {
        static_assert(is_host::value);
        return _host;
    }
    View<T, Layout>& d_ptr() {
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

        if constexpr (is_host::value) {
            _host.ptr = static_cast<T*>(memory.alloc(MemoryLocation::Host, size_bytes()));
        }
        if constexpr (is_device::value) {
            _device.ptr = (!memory.is_unified()) 
                ? static_cast<T*>(memory.alloc(MemoryLocation::Device, size_bytes()))
                : _host.ptr;
        }
    }

    void dealloc() {
        if constexpr (is_host::value) {
            memory.free(MemoryLocation::Host, _host.ptr);
        }
        if constexpr (is_device::value) {
            if (!memory.is_unified()) memory.free(MemoryLocation::Device, _device());
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

class MultiSurface {
    public:
        struct Surface { // ns major
            uint64_t ns;
            uint64_t nc;
            uint64_t offset;
        };

        MultiSurface() = default;
        MultiSurface(const std::vector<Surface>& surfaces, uint32_t dim) : 
            _surfaces(surfaces),
            _stride(surfaces.back().offset + surfaces.back().ns * surfaces.back().nc),
            _dim(dim) {}
        ~MultiSurface() = default;

        uint64_t operator()(uint32_t wing_id, uint32_t dim) { // returns start of the specific wing buffer
            assert(wing_id < _surfaces.size());
            assert(dim < _dim);
            return _surfaces[wing_id].offset + dim * _stride;
        }

        uint64_t size() const {return _dim * _stride; } // required
        uint64_t stride() const {return _stride; }
        uint64_t dim() const {return _dim; }

    private:
        std::vector<Surface> _surfaces;
        uint64_t _stride = 0; // stride between each Dim
        uint32_t _dim = 1;
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