#pragma once
#include <ggml.h>
#include <gguf.h>

#include <cstring>
#include <type_traits>
#include <string>
#include <format>

inline
std::string combine_prefix(const std::string& prefix, const char* name)
{
	if (prefix.empty())
		return name;
	else
		return std::format("{}.{}", prefix, name);
}

struct gguf_metadata_loader
{
	gguf_context* gguf_ctx;

	gguf_metadata_loader(gguf_context* ctx) : gguf_ctx(ctx) {}

	operator bool() const { return gguf_ctx; }

	operator gguf_context* () const { return gguf_ctx; }

	template<typename T>
	static constexpr T(*gguf_get_val_provider())(const gguf_context*, int64_t)
	{
		if constexpr (std::is_same_v<T, int32_t>)
			return gguf_get_val_i32;
		else if constexpr (std::is_same_v<T, uint32_t>)
			return gguf_get_val_u32;
		else if constexpr (std::is_same_v<T, int64_t>)
			return gguf_get_val_i64;
		else if constexpr (std::is_same_v<T, float>)
			return gguf_get_val_f32;
		else static_assert("unsupported type");
	}

	template<typename T>
	bool get_metadata(const std::string& prefix, const char* name, T& value) const
	{
		return get_metadata(combine_prefix(prefix, name).c_str(), value);
	}

	template<typename T>
	bool get_metadata(const char* key_name, T& value) const
	{
		int64_t id;
		if (find_metadata_key(key_name, id))
		{
			value = gguf_get_val_provider<T>()(gguf_ctx, id);
			return true;
		}
		return false;
	}

	std::string_view get_string(const char* key_name) const
	{
		int64_t id;
		GGML_ASSERT(find_metadata_key(key_name, id));
		return std::string_view(gguf_get_val_str(gguf_ctx, id));
	}

	bool find_metadata_key(const char* key_name, int64_t& key_id) const
	{
		const auto n_kv = gguf_get_n_kv(gguf_ctx);
		for (int64_t i = 0; i != n_kv; ++i)
			if (strcmp(key_name, gguf_get_key(gguf_ctx, i)) == 0)
			{
				key_id = i;
				return true;
			}
		return false;
	}
};

struct gguf_loader : gguf_metadata_loader
{
	ggml_context* gguf_ggml_ctx;

	gguf_loader(const char* filename) :
		gguf_metadata_loader(gguf_init_from_file(filename, gguf_init_params{
			.no_alloc = false, .ctx = &gguf_ggml_ctx })) {}

	~gguf_loader() { close(); }

	void close()
	{
		if (gguf_ctx)
		{
			gguf_free(gguf_ctx);
			ggml_free(gguf_ggml_ctx);
			gguf_ctx = nullptr;
			gguf_ggml_ctx = nullptr;
		}
	}

	operator ggml_context* () const { return gguf_ggml_ctx; }

	ggml_tensor* get_gguf_tensor(const std::string& prefix, const char* name, bool optional = false) const
	{
		auto tensor = ggml_get_tensor(gguf_ggml_ctx, combine_prefix(prefix, name).c_str());
		GGML_ASSERT(optional || tensor);
		return tensor;
	}
};
