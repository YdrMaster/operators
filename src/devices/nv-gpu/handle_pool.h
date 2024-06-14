#include <atomic>
#include <cublas_v2.h>
#include <optional>

template<class T>
class Pool {
public:
    Pool() : _head(nullptr) {}

    template<class U>
    struct Node {
        U data;
        Node<U> *next;
        Node(const U &data) : data(data), next(nullptr) {}
    };

    void push(T &&val) {
        Node<T> *new_node = new Node<T>(val);
        new_node->next = _head.load();
        while (!_head.compare_exchange_weak(new_node->next, new_node));
    }

    std::optional<T> pop() {
        Node<T> *top = _head.load();
        Node<T> *new_head = nullptr;
        do {
            if (top == nullptr) {
                return std::nullopt;
            }
            new_head = top->next;
        } while (!_head.compare_exchange_weak(top, new_head));
        return {std::move(top->data)};
    }

private:
    std::atomic<Node<T> *> _head;
};

static auto cublas_pool = Pool<cublasHandle_t>();

template<typename T>
void use_cublas(cudaStream_t stream, T const &f) {
    auto handle = cublas_pool.pop();
    if (!handle) {
        cublasCreate(&(*handle));
    }
    cublasSetStream(*handle, (cudaStream_t) stream);
    f(*handle);
    cublas_pool.push(std::move(*handle));
}
