#include "cosyvoice.h"
#include "cosyvoice-frontend.h"
#include "tool_common.h"

#include <atomic>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

namespace
{
    constexpr const char* kSample =
        "The 1st, 2nd, 3rd, 4th, 5th, 8th, 9th, 12th, and 21st of May. "
        "Add 5 kg, 250 mg, 3 km, 12 cm, 2.5 L, and 30 deg C. "
        "The train leaves at 10:30. Please bring 2 bottles of water and 3 apples. "
        "We will meet at the station when the weather is nice. ";
    constexpr const char* kLocale = "en";

    std::string normalize(const char* text)
    {
        cosyvoice_text_ptr output = cosyvoice_text_create();
        cosyvoice_frontend_util_text_normalize(
            text, static_cast<uint32_t>(std::strlen(text)), kLocale, output);
        std::string result(output->text ? output->text : "", output->length);
        cosyvoice_text_free(output);
        return result;
    }

    bool init_pool(uint64_t bytes)
    {
        cosyvoice_runtime_params_t params{};
        params.memory_pool_bytes = bytes;
        return cosyvoice_init_runtime(&params, sizeof(params));
    }

    int run_late()
    {
        normalize(kSample);

        if (!init_pool(16 * 1024 * 1024))
        {
            std::printf("PASS: late memory pool initialization rejected\n");
            return 0;
        }

        std::printf("FAIL: late memory pool initialization was accepted\n");
        return 1;
    }

    int run_early()
    {
        cosyvoice_runtime_params_t undersized{};
        undersized.memory_pool_bytes = 16 * 1024 * 1024;
        if (cosyvoice_init_runtime(&undersized, 4))
        {
            std::printf("FAIL: undersized runtime params struct was accepted\n");
            return 1;
        }

        if (!init_pool(16 * 1024 * 1024))
        {
            std::printf("FAIL: initial memory pool initialization failed\n");
            return 1;
        }

        if (init_pool(16 * 1024 * 1024))
        {
            std::printf("FAIL: repeated memory pool initialization was accepted\n");
            return 1;
        }

        const std::string expected = normalize(kSample);
        if (expected.empty())
        {
            std::printf("FAIL: normalization produced empty output\n");
            return 1;
        }

        constexpr int kThreads = 4;
        constexpr int kIterations = 8;
        std::atomic<bool> failed{false};
        std::vector<std::thread> threads;
        threads.reserve(kThreads);
        for (int i = 0; i < kThreads; ++i)
        {
            threads.emplace_back([&] {
                cosyvoice_text_ptr output = cosyvoice_text_create();
                for (int j = 0; j < kIterations; ++j)
                {
                    cosyvoice_frontend_util_text_normalize(
                        kSample, static_cast<uint32_t>(std::strlen(kSample)), kLocale, output);
                    if (output->length != expected.size()
                        || std::memcmp(output->text, expected.data(), expected.size()) != 0)
                    {
                        failed.store(true);
                        break;
                    }
                }
                cosyvoice_text_free(output);
            });
        }
        for (auto& thread : threads)
            thread.join();

        if (failed.load())
        {
            std::printf("FAIL: concurrent normalization differed\n");
            return 1;
        }

        std::printf("PASS: memory pool initialized once and concurrent normalization matched\n");
        return 0;
    }
}

int tool_entry(int argc, char** argv)
{
    if (argc > 1 && std::strcmp(argv[1], "late") == 0)
        return run_late();
    return run_early();
}
