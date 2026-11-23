#pragma once

#include <forward_list>
#include <limits>
#include <map>
#include <new>
#include <queue>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace cosyvoice
{
    void* memory_allocate(std::size_t bytes, std::size_t alignment);
    void  memory_deallocate(void* ptr, std::size_t bytes, std::size_t alignment) noexcept;

    template <class T>
    struct allocator
    {
        using value_type = T;
        using is_always_equal = std::true_type;

        allocator() noexcept = default;

        template <class U>
        allocator(const allocator<U>&) noexcept {}

        template <class U>
        struct rebind
        {
            using other = allocator<U>;
        };

        [[nodiscard]] T* allocate(std::size_t n)
        {
            if (n > (std::numeric_limits<std::size_t>::max)() / sizeof(T))
                throw std::bad_array_new_length();
            return static_cast<T*>(memory_allocate(n * sizeof(T), alignof(T)));
        }

        void deallocate(T* p, std::size_t n) noexcept
        {
            memory_deallocate(p, n * sizeof(T), alignof(T));
        }

        template <class U>
        bool operator==(const allocator<U>&) const noexcept { return true; }

        template <class U>
        bool operator!=(const allocator<U>&) const noexcept { return false; }
    };

    using string  = std::basic_string<char, std::char_traits<char>, allocator<char>>;
    using wstring = std::basic_string<wchar_t, std::char_traits<wchar_t>, allocator<wchar_t>>;

    template <class T>
    using vector = std::vector<T, allocator<T>>;

    template <class Key, class T, class Compare = std::less<Key>>
    using map = std::map<Key, T, Compare, allocator<std::pair<const Key, T>>>;

    template <class Key, class T, class Hash = std::hash<Key>, class KeyEqual = std::equal_to<Key>>
    using unordered_map = std::unordered_map<Key, T, Hash, KeyEqual, allocator<std::pair<const Key, T>>>;

    template <class T>
    using forward_list = std::forward_list<T, allocator<T>>;

    template <class Key, class Compare = std::less<Key>>
    using set = std::set<Key, Compare, allocator<Key>>;

    template <class Key, class Hash = std::hash<Key>, class KeyEqual = std::equal_to<Key>>
    using unordered_set = std::unordered_set<Key, Hash, KeyEqual, allocator<Key>>;

    template <class T, class Container = vector<T>, class Compare = std::less<typename Container::value_type>>
    using priority_queue = std::priority_queue<T, Container, Compare>;
}

using cosyvoice::string;
using cosyvoice::wstring;
using cosyvoice::vector;
using cosyvoice::map;
using cosyvoice::unordered_map;
using cosyvoice::forward_list;
using cosyvoice::set;
using cosyvoice::unordered_set;
using cosyvoice::priority_queue;

static_assert(sizeof(string) == sizeof(std::string), "pooled string must not add per-object overhead");
static_assert(sizeof(wstring) == sizeof(std::wstring), "pooled wstring must not add per-object overhead");
static_assert(sizeof(vector<int>) == sizeof(std::vector<int>), "pooled vector must not add per-object overhead");
