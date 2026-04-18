#include "cosyvoice-internal.h"
#include "cosyvoice-frontend.h"

#ifndef COSYVOICE_NO_ICU

    #include <string_view>
    #include <string>
    #include <unordered_map>
    #include <functional>

    #include <unicode/unistr.h>
    #include <unicode/normalizer2.h>
    #include <unicode/regex.h>
    #include <unicode/locid.h>
    #include <unicode/uscript.h>
    #include <unicode/numfmt.h>
    #include <unicode/rbnf.h>
    #include <unicode/measfmt.h>
    #include <unicode/measure.h>

static bool is_valid_locale(const char* locale)
{
    if (!locale || !*locale) return false;
    UErrorCode status = U_ZERO_ERROR;
    char language[32];
    uloc_getLanguage(locale, language, sizeof(language), &status);
    return U_SUCCESS(status) && language[0] != 0;
}

static icu::Locale detect_locale(const icu::UnicodeString& text)
{
    std::unordered_map<UScriptCode, int> counts;
    for (int32_t i = 0; i < text.length();) {
        UChar32 c = text.char32At(i);
        i += U16_LENGTH(c);
        UErrorCode status = U_ZERO_ERROR;
        UScriptCode script = uscript_getScript(c, &status);
        if (U_FAILURE(status)) continue;
        if (script == USCRIPT_COMMON || script == USCRIPT_INHERITED) continue;
        counts[script]++;
    }

    UScriptCode bestScript = USCRIPT_LATIN;
    int bestCount = -1;
    for (const auto& kv : counts) {
        if (kv.second > bestCount) {
            bestCount = kv.second;
            bestScript = kv.first;
        }
    }

    switch (bestScript)
    {
    case USCRIPT_HAN: return icu::Locale("zh");
    case USCRIPT_HIRAGANA:
    case USCRIPT_KATAKANA: return icu::Locale("ja");
    case USCRIPT_HANGUL: return icu::Locale("ko");
    case USCRIPT_CYRILLIC: return icu::Locale("ru");
    case USCRIPT_ARABIC: return icu::Locale("ar");
    case USCRIPT_DEVANAGARI: return icu::Locale("hi");
    case USCRIPT_THAI: return icu::Locale("th");
    case USCRIPT_HEBREW: return icu::Locale("he");
    case USCRIPT_GREEK: return icu::Locale("el");
    default: return icu::Locale("en");
    }
}

static icu::UnicodeString replace_with_matcher(
    const icu::UnicodeString& input,
    const icu::UnicodeString& pattern,
    UErrorCode& status,
    const std::function<icu::UnicodeString(icu::RegexMatcher&)>& replacer)
{
    icu::RegexMatcher matcher(pattern, input, 0, status);
    if (U_FAILURE(status)) return input;

    icu::UnicodeString out;
    int32_t last = 0;
    while (matcher.find()) {
        out.append(input, last, matcher.start(status) - last);
        out.append(replacer(matcher));
        last = matcher.end(status);
    }
    out.append(input, last, input.length() - last);
    return out;
}

static void normalize_fullwidth_digits(icu::UnicodeString& text)
{
    for (int32_t i = 0; i < text.length(); )
    {
        UChar32 c = text.char32At(i);
        if (c >= 0xFF10 && c <= 0xFF19)
            text.replace(i, U16_LENGTH(c), icu::UnicodeString(static_cast<UChar32>(c - 0xFF10 + '0')));
        i += U16_LENGTH(c);
    }
}
#endif

static bool strip(const char*& orig_text, uint32_t& text_len)
{
    std::string_view text(orig_text, text_len);

    size_t start = text.find_first_not_of(" \t\n\r");
    size_t end = text.find_last_not_of(" \t\n\r");
    if (start == std::string::npos || end == std::string::npos
        || (start == 0 && end == text.length() - 1))
        return false;
    orig_text = text.data() + start;
    text_len = static_cast<uint32_t>(end - start + 1);
    return true;
}

struct cosyvoice_text_impl : cosyvoice_text
{
    std::string text_storage;
};

cosyvoice_text_ptr cosyvoice_text_create()
{
    return new cosyvoice_text_impl{};
}

void cosyvoice_text_free(cosyvoice_text_ptr text)
{
    delete reinterpret_cast<cosyvoice_text_impl*>(text);
}

void cosyvoice_frontend_util_text_normalize(const char* text, uint32_t text_len, const char* locale, cosyvoice_text_ptr normalized_text)
{
    auto p = static_cast<cosyvoice_text_impl*>(normalized_text);
    if (!cosyvoice_frontend_util_text_normalize(p->text_storage, text, text_len, locale))
        p->text_storage.assign(text, text_len);

    p->length = static_cast<uint32_t>(p->text_storage.size());
    p->text = p->text_storage.c_str();
}

bool cosyvoice_frontend_util_text_normalize(std::string& text, const char* orig_text, uint32_t text_len, const char* locale)
{
#ifdef COSYVOICE_NO_ICU
    if (strip(orig_text, text_len))
    {
        text.assign(orig_text, text_len);
        return true;
    }
    return false;
#else
    strip(orig_text, text_len);

    UErrorCode status = U_ZERO_ERROR;
    const icu::Normalizer2* nfc = icu::Normalizer2::getNFCInstance(status);
    if (U_FAILURE(status)) return false;

    icu::UnicodeString utext = icu::UnicodeString::fromUTF8(icu::StringPiece(orig_text, text_len));
    icu::UnicodeString norm;
    nfc->normalize(utext, norm, status);
    if (U_FAILURE(status)) return false;

    // Keep fullwidth punctuation unchanged while still normalizing fullwidth digits.
    normalize_fullwidth_digits(norm);

    icu::Locale loc = is_valid_locale(locale) ? icu::Locale(locale) : detect_locale(norm);

    auto finalize = [&]() {
        text.clear();
        norm.toUTF8String(text);
        return true;
    };
    icu::RuleBasedNumberFormat spellout(icu::URBNF_SPELLOUT, loc, status);
    if (U_FAILURE(status)) return finalize();

    std::unique_ptr<icu::NumberFormat> numberParser(icu::NumberFormat::createInstance(loc, status));
    if (U_FAILURE(status) || !numberParser) return finalize();

    // Common unit suffixes such as kg, mg, km, m, cm, mm, L, ml, °C, and °F.
    static const icu::UnicodeString unitPattern = icu::UnicodeString::fromUTF8(
        reinterpret_cast<const char*>(u8R"(([+-]?\d+(?:[.,]\d+)?)(\s*)(kg|mg|g|km|cm|mm|L|ml|m|°C|°F))"));
    norm = replace_with_matcher(norm, unitPattern, status,
        [&](icu::RegexMatcher& m) -> icu::UnicodeString {
            icu::UnicodeString num = m.group(1, status);
            icu::UnicodeString unit = m.group(3, status);
            icu::Formattable parsed;
            icu::ParsePosition pos(0);
            numberParser->parse(num, parsed, pos);

            std::unique_ptr<icu::MeasureUnit> mu;
            if (unit == u"kg") mu.reset(icu::MeasureUnit::createKilogram(status));
            else if (unit == u"g") mu.reset(icu::MeasureUnit::createGram(status));
            else if (unit == u"mg") mu.reset(icu::MeasureUnit::createMilligram(status));
            else if (unit == u"km") mu.reset(icu::MeasureUnit::createKilometer(status));
            else if (unit == u"m") mu.reset(icu::MeasureUnit::createMeter(status));
            else if (unit == u"cm") mu.reset(icu::MeasureUnit::createCentimeter(status));
            else if (unit == u"mm") mu.reset(icu::MeasureUnit::createMillimeter(status));
            else if (unit == u"L") mu.reset(icu::MeasureUnit::createLiter(status));
            else if (unit == u"ml") mu.reset(icu::MeasureUnit::createMilliliter(status));
            else if (unit == u"°C") mu.reset(icu::MeasureUnit::createCelsius(status));
            else if (unit == u"°F") mu.reset(icu::MeasureUnit::createFahrenheit(status));

            if (!mu || U_FAILURE(status)) return m.group(0, status);

            icu::Measure measure(parsed, mu.release(), status);
            icu::MeasureFormat mf(loc, UMEASFMT_WIDTH_WIDE, status);
            icu::UnicodeString out;
            icu::FieldPosition fp;
            mf.formatMeasures(&measure, 1, out, fp, status);
            return out;
        });

    static const icu::UnicodeString ordinalPattern = icu::UnicodeString::fromUTF8(reinterpret_cast<const char*>(u8R"(\b(\d+)(st|nd|rd|th)\b)"));
    norm = replace_with_matcher(norm, ordinalPattern, status,
        [&](icu::RegexMatcher& m) -> icu::UnicodeString
        {
            icu::UnicodeString num = m.group(1, status);
            icu::Formattable parsed;
            icu::ParsePosition pos(0);
            numberParser->parse(num, parsed, pos);
            num.remove();
            spellout.format(parsed, num, status);
            auto view = std::u16string_view(num.getBuffer(), num.length());
            static const std::unordered_map<std::u16string_view, std::u16string_view> specialOrdinals =
            {
                {u"one", u"first"},
                {u"two", u"second"},
                {u"three", u"third"},
                {u"five", u"fifth"},
                {u"eight", u"eighth"},
                {u"nine", u"ninth"},
                {u"twelve", u"twelfth"}
            };

            auto it = specialOrdinals.find(view);
            if (it != specialOrdinals.end())
            {
                num = it->second.data();
                return num;
            }

            std::u16string_view base(num.getBuffer(), num.length());
            size_t hyphenPos = base.rfind(u'-');
            if (hyphenPos != std::u16string_view::npos)
            {
                auto suffix = base.substr(hyphenPos + 1);

                it = specialOrdinals.find(suffix);
                if (it != specialOrdinals.end())
                {
                    num.removeBetween(static_cast<int32_t>(hyphenPos) + 1);
                    num.append(it->second.data(), 0, static_cast<int32_t>(it->second.length()));
                    return num;
                }

                if (suffix.length() > 0 && suffix.back() == L'e')
                    num.removeBetween(num.length() - 1);
                num.append(u"th", 0, 2);
                return num;
            }

            if (base.length() > 0 && base.back() == L'e')
                num.removeBetween(num.length() - 1);
            else if (base.length() > 0 && base.back() == L'y')
                num.removeBetween(num.length() - 1);

            num.append(u"th", 0, 2);
            return num;
        });

    static const icu::UnicodeString numberPattern = icu::UnicodeString::fromUTF8(reinterpret_cast<const char*>(u8R"(\d+(?:[.,]\d+)?)"));
    norm = replace_with_matcher(norm, numberPattern, status,
        [&](icu::RegexMatcher& m) -> icu::UnicodeString
        {
            icu::UnicodeString num = m.group(0, status);
            icu::Formattable parsed;
            icu::ParsePosition pos(0);
            numberParser->parse(num, parsed, pos);
            icu::UnicodeString out;
            spellout.format(parsed, out, status);
            return out;
        });

    return finalize();
#endif
}
