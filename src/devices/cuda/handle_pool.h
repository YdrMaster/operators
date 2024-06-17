#include <atomic>
#include <cublas_v2.h>
#include <mutex>
#include <optional>
#include <vector>

template<class T>
class Pool {
public:
    Pool() : _head(nullptr) {}

    Pool(const Pool &) = delete;

    Pool(Pool &&pool) {
        _head.store(pool._head.load());
    }

    ~Pool() {
        while (this->pop()) {}
    }

    void push(T &&val) const {
        Node<T> *new_node = new Node<T>(val);
        new_node->next = _head.load();
        while (!_head.compare_exchange_weak(new_node->next, new_node));
    }

    std::optional<T> pop() const {
        Node<T> *top = _head.load();
        Node<T> *new_head = nullptr;
        do {
            if (!top) {
                return std::nullopt;
            }
            new_head = top->next;
        } while (!_head.compare_exchange_weak(top, new_head));
        return {std::move(top->data)};
    }

private:
    template<class U>
    struct Node {
        U data;
        Node<U> *next;
        Node(const U &data) : data(data), next(nullptr) {}
    };

    mutable std::atomic<Node<T> *> _head;
};


const Pool<cublasHandle_t> &get_cublas_pool() {
    int device_id;
    cudaGetDevice(&device_id);
    static std::once_flag flag;
    static std::vector<Pool<cublasHandle_t>> cublas_pool;
    std::call_once(flag, [&]() {
        int device_count;
        cudaGetDeviceCount(&device_count);
        for (int i = 0; i < device_count; i++) {
            auto pool = Pool<cublasHandle_t>();
            cublasHandle_t handle;
            cublasCreate(&handle);
            pool.push(std::move(handle));
            cublas_pool.emplace_back(std::move(pool));
        } });
    return cublas_pool[device_id];
}

template<typename T>
void use_cublas(cudaStream_t stream, T const &f) {
    auto &pool = get_cublas_pool();
    auto handle = pool.pop();
    if (!handle) {
        cublasCreate(&(*handle));
    }
    cublasSetStream(*handle, (cudaStream_t) stream);
    f(*handle);
    pool.push(std::move(*handle));
}
