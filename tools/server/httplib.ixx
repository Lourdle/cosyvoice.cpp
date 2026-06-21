module;
#include "httplib.h"

export module httplib;

export using Request = httplib::Request;
export using Response = httplib::Response;
export using ThreadPool = httplib::ThreadPool;
export using HandlerResponse = httplib::Server::HandlerResponse;

// NOTE: Do NOT export `using Server = httplib::Server;` directly!
// Tested on MSVC: this triggers unresolved externals (e.g. `__tlregdtor`) when using C++20 modules,
// which seems related to thread_local and module symbol export.
// The underlying cause is unclear and may be an MSVC bug (other compilers not tested).
// Using a wrapper avoids this problem.
export class Server
{
    httplib::Server server;

public:
    Server() : new_task_queue(server.new_task_queue) {}

    decltype(httplib::Server::new_task_queue)& new_task_queue;
    
    template<class Handler>
    void Get(const std::string& pattern, Handler handler)
    {
        server.Get(pattern, std::move(handler));
    }

    template<class Handler>
    void Post(const std::string& pattern, Handler handler)
    {
        server.Post(pattern, std::move(handler));
    }

    template<class Handler>
    void Delete(const std::string& pattern, Handler handler)
    {
        server.Delete(pattern, std::move(handler));
    }

    template<class Handler>
    void Put(const std::string& pattern, Handler handler)
    {
        server.Put(pattern, std::move(handler));
    }

    template<class Handler>
    void set_pre_routing_handler(Handler handler)
    {
        server.set_pre_routing_handler(std::move(handler));
    }

    template<class ExceptionHandler>
    void set_exception_handler(ExceptionHandler handler)
    {
        server.set_exception_handler(std::move(handler));
    }

    template <class ErrorHandlerFunc>
    void set_error_handler(ErrorHandlerFunc&& handler)
    {
        server.set_error_handler(std::move(handler));
    }

    bool listen(const char* host, int port)
    {
        return server.listen(host, port);
    }
};
