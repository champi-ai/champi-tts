"""
Async text utilities for Kokoro TTS.
Handles normalization, phonemization, and tokenization with non-blocking operations.
"""

import asyncio
import re

from loguru import logger

from champi_tts.providers.kokoro.events import TTSSignalManager

try:
    import inflect
    import phonemizer
except ImportError:
    logger.error("Missing dependencies. Install with: pip install phonemizer inflect")
    raise


class AsyncTextNormalizer:
    """Async text normalization for TTS processing"""

    def __init__(self):
        self.inflect_engine = inflect.engine()

        # Symbol replacements
        self.symbol_replacements = {
            "~": " ",
            "@": " at ",
            "#": " number ",
            "$": " dollar ",
            "%": " percent ",
            "^": " ",
            "&": " and ",
            "*": " ",
            "_": " ",
            "|": " ",
            "\\": " ",
            "/": " slash ",
            "=": " equals ",
            "+": " plus ",
        }

        # Money units
        self.money_units = {
            "$": ("dollar", "cent"),
            "£": ("pound", "pence"),
            "€": ("euro", "cent"),
        }

        # Precompiled patterns
        self._compile_patterns()

    def _compile_patterns(self):
        """Compile regex patterns for performance"""
        self.email_pattern = re.compile(
            r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-z]{2,}\b", re.IGNORECASE
        )

        self.url_pattern = re.compile(
            r"(https?://|www\.|)+(localhost|[a-zA-Z0-9.-]+\.(com|org|net|edu|gov|io|co)"
            r")(:[0-9]+)?([/?][^\s]*)?",
            re.IGNORECASE,
        )

        self.time_pattern = re.compile(
            r"([0-9]{1,2} ?: ?[0-9]{2}( ?: ?[0-9]{2})?)( ?(pm|am)\b)?", re.IGNORECASE
        )

        self.money_pattern = re.compile(
            r"(-?)([$£€])(\d+(?:\.\d+)?)((?: hundred| thousand| (?:[bm]|tr|quadr)illion|k|m|b|t)*)\b",
            re.IGNORECASE,
        )

        self.number_pattern = re.compile(
            r"(-?)(\d+(?:\.\d+)?)((?: hundred| thousand| (?:[bm]|tr|quadr)illion|k|m|b)*)\b",
            re.IGNORECASE,
        )

    async def handle_email(self, match) -> str:
        """Convert email to speakable format"""
        email = match.group(0)
        parts = email.split("@")
        if len(parts) == 2:
            user, domain = parts
            domain = domain.replace(".", " dot ")
            return f"{user} at {domain}"
        return email

    async def handle_url(self, match) -> str:
        """Convert URL to speakable format"""
        url = match.group(0).strip()

        # Handle protocol
        url = re.sub(
            r"^https?://",
            lambda a: "https " if "https" in a.group() else "http ",
            url,
            flags=re.IGNORECASE,
        )
        url = re.sub(r"^www\.", "www ", url, flags=re.IGNORECASE)

        # Replace dots and common patterns
        url = url.replace(".", " dot ")
        url = url.replace("/", " slash ")
        url = url.replace("_", " underscore ")
        url = url.replace("-", " dash ")

        return url.strip()

    async def handle_time(self, match) -> str:
        """Convert time to speakable format"""
        time_str = match.group(0)
        clean_time = re.sub(r"[^\d:apm ]", "", time_str.lower())

        try:
            if ":" in clean_time:
                parts = clean_time.split(":")
                if len(parts) >= 2:
                    hours = int(parts[0])
                    minutes = int(parts[1])

                    # Convert to 12-hour format if needed
                    if hours == 0:
                        hour_word = "twelve"
                        period = "a m" if "am" in clean_time else "midnight"
                    elif hours <= 12:
                        hour_word = self.inflect_engine.number_to_words(hours)
                        period = "a m" if "am" in clean_time or hours < 12 else "p m"
                    else:
                        hour_word = self.inflect_engine.number_to_words(hours - 12)
                        period = "p m"

                    if minutes == 0:
                        return f"{hour_word} o'clock {period}".replace(
                            "a m", "a m"
                        ).replace("p m", "p m")
                    else:
                        minute_word = self.inflect_engine.number_to_words(minutes)
                        return f"{hour_word} {minute_word} {period}"

        except (ValueError, AttributeError):
            pass

        return time_str

    async def handle_money(self, match) -> str:
        """Convert money amounts to speakable format"""
        sign = match.group(1)
        currency = match.group(2)
        amount = match.group(3)
        suffix = match.group(4) or ""

        try:
            number = float(amount)

            # Handle currency
            currency_name, _cent_name = self.money_units.get(
                currency, ("dollar", "cent")
            )

            # Handle large numbers with suffixes
            if "k" in suffix.lower():
                number *= 1000
            elif "m" in suffix.lower():
                number *= 1000000
            elif "b" in suffix.lower():
                number *= 1000000000
            elif "t" in suffix.lower():
                number *= 1000000000000

            # Convert to words
            if number == int(number):
                number = int(number)

            number_words = self.inflect_engine.number_to_words(number)

            # Handle singular/plural
            if number == 1:
                result = f"{number_words} {currency_name}"
            else:
                result = f"{number_words} {currency_name}s"

            if sign == "-":
                result = f"negative {result}"

            return result

        except (ValueError, AttributeError):
            return match.group(0)

    async def handle_numbers(self, match) -> str:
        """Convert numbers to words"""
        sign = match.group(1)
        number = match.group(2)
        suffix = match.group(3) or ""

        try:
            num = float(number)

            # Handle suffixes
            if "thousand" in suffix:
                num *= 1000
            elif "million" in suffix:
                num *= 1000000
            elif "billion" in suffix:
                num *= 1000000000
            elif "trillion" in suffix:
                num *= 1000000000000
            elif "k" in suffix.lower():
                num *= 1000
            elif "m" in suffix.lower():
                num *= 1000000
            elif "b" in suffix.lower():
                num *= 1000000000

            # Convert to integer if it's a whole number
            if num == int(num):
                num = int(num)

            result = self.inflect_engine.number_to_words(num)

            if sign == "-":
                result = f"negative {result}"

            return result

        except (ValueError, AttributeError):
            return match.group(0)

    async def replace_symbols(self, text: str) -> str:
        """Replace symbols with speakable equivalents"""
        for symbol, replacement in self.symbol_replacements.items():
            text = text.replace(symbol, replacement)
        return text

    def _normalize_sync(self, text: str) -> str:
        """Synchronous normalization helper"""
        # Clean up text
        normalized = text.strip()

        # Handle special patterns synchronously
        def handle_email_sync(match):
            email = match.group(0)
            parts = email.split("@")
            if len(parts) == 2:
                user, domain = parts
                domain = domain.replace(".", " dot ")
                return f"{user} at {domain}"
            return email

        def handle_url_sync(match):
            url = match.group(0).strip()
            url = re.sub(
                r"^https?://",
                lambda a: "https " if "https" in a.group() else "http ",
                url,
                flags=re.IGNORECASE,
            )
            url = re.sub(r"^www\.", "www ", url, flags=re.IGNORECASE)
            url = url.replace(".", " dot ")
            url = url.replace("/", " slash ")
            url = url.replace("_", " underscore ")
            url = url.replace("-", " dash ")
            return url.strip()

        def handle_time_sync(match):
            time_str = match.group(0)
            clean_time = re.sub(r"[^\d:apm ]", "", time_str.lower())
            try:
                if ":" in clean_time:
                    parts = clean_time.split(":")
                    if len(parts) >= 2:
                        hours = int(parts[0])
                        minutes = int(parts[1])

                        if hours == 0:
                            hour_word = "twelve"
                            period = "a m" if "am" in clean_time else "midnight"
                        elif hours <= 12:
                            hour_word = self.inflect_engine.number_to_words(hours)
                            period = (
                                "a m" if "am" in clean_time or hours < 12 else "p m"
                            )
                        else:
                            hour_word = self.inflect_engine.number_to_words(hours - 12)
                            period = "p m"

                        if minutes == 0:
                            return f"{hour_word} o'clock {period}".replace(
                                "a m", "a m"
                            ).replace("p m", "p m")
                        else:
                            minute_word = self.inflect_engine.number_to_words(minutes)
                            return f"{hour_word} {minute_word} {period}"
            except (ValueError, AttributeError):
                pass
            return time_str

        def handle_money_sync(match):
            sign = match.group(1)
            currency = match.group(2)
            amount = match.group(3)
            suffix = match.group(4) or ""
            try:
                number = float(amount)
                currency_name, _cent_name = self.money_units.get(
                    currency, ("dollar", "cent")
                )

                if "k" in suffix.lower():
                    number *= 1000
                elif "m" in suffix.lower():
                    number *= 1000000
                elif "b" in suffix.lower():
                    number *= 1000000000
                elif "t" in suffix.lower():
                    number *= 1000000000000

                if number == int(number):
                    number = int(number)

                number_words = self.inflect_engine.number_to_words(number)

                if number == 1:
                    result = f"{number_words} {currency_name}"
                else:
                    result = f"{number_words} {currency_name}s"

                if sign == "-":
                    result = f"negative {result}"

                return result
            except (ValueError, AttributeError):
                return match.group(0)

        def handle_numbers_sync(match):
            sign = match.group(1)
            number = match.group(2)
            suffix = match.group(3) or ""
            try:
                num = float(number)

                if "thousand" in suffix:
                    num *= 1000
                elif "million" in suffix:
                    num *= 1000000
                elif "billion" in suffix:
                    num *= 1000000000
                elif "trillion" in suffix:
                    num *= 1000000000000
                elif "k" in suffix.lower():
                    num *= 1000
                elif "m" in suffix.lower():
                    num *= 1000000
                elif "b" in suffix.lower():
                    num *= 1000000000

                if num == int(num):
                    num = int(num)

                result = self.inflect_engine.number_to_words(num)

                if sign == "-":
                    result = f"negative {result}"

                return result
            except (ValueError, AttributeError):
                return match.group(0)

        # Apply all pattern replacements
        normalized = self.email_pattern.sub(handle_email_sync, normalized)
        normalized = self.url_pattern.sub(handle_url_sync, normalized)
        normalized = self.time_pattern.sub(handle_time_sync, normalized)
        normalized = self.money_pattern.sub(handle_money_sync, normalized)
        normalized = self.number_pattern.sub(handle_numbers_sync, normalized)

        # Replace symbols
        for symbol, replacement in self.symbol_replacements.items():
            normalized = normalized.replace(symbol, replacement)

        # Clean up whitespace
        normalized = re.sub(r"\s+", " ", normalized).strip()

        return normalized

    async def normalize(self, text: str) -> str:
        """Normalize text for TTS with async processing"""
        if not text or not text.strip():
            return ""

        # Run heavy processing in thread pool
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._normalize_sync, text)


class AsyncPhonemizerBackend:
    """Async wrapper for phonemizer backend"""

    def __init__(self, language: str = "en-us"):
        self.language = language
        self._backend = None

    async def _ensure_backend(self):
        """Lazy initialize backend"""
        if self._backend is None:
            loop = asyncio.get_event_loop()

            # Spanish-specific configuration for better accentuation
            if self.language == "es":
                self._backend = await loop.run_in_executor(
                    None,
                    lambda: phonemizer.backend.EspeakBackend(
                        language="es",
                        preserve_punctuation=True,
                        with_stress=True,
                        tie=True,  # Use tie marks for proper Spanish liaison
                        language_switch="keep-flags",  # Preserve language markers
                        punctuation_marks=";:,.!?¿¡-\"'",  # Spanish punctuation
                    ),
                )
            else:
                # Default configuration for other languages
                self._backend = await loop.run_in_executor(
                    None,
                    lambda: phonemizer.backend.EspeakBackend(
                        language=self.language,
                        preserve_punctuation=True,
                        with_stress=True,
                    ),
                )

    async def phonemize(self, text: str) -> str:
        """Convert text to phonemes asynchronously"""
        await self._ensure_backend()

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, lambda: self._backend.phonemize([text])[0]
        )


class AsyncTextProcessor:
    """Complete async text processing pipeline for Kokoro TTS"""

    def __init__(self):
        self.normalizer = AsyncTextNormalizer()
        self.phonemizers: dict[str, AsyncPhonemizerBackend] = {}
        self.signals = (
            TTSSignalManager()
        )  # Global signal manager for text processing events

    async def _get_phonemizer(self, lang_code: str) -> AsyncPhonemizerBackend:
        """Get or create phonemizer for language code"""
        if lang_code not in self.phonemizers:
            # Map Kokoro language codes to phonemizer languages
            lang_map = {
                "a": "en-us",
                "b": "en-gb",
                "e": "es",  # Spanish
                "f": "fr",  # French
                "h": "hi",  # Hindi
                "i": "it",  # Italian
                "j": "ja",  # Japanese
                "p": "pt-br",  # Portuguese (Brazil)
                "z": "zh-cn",  # Chinese (Mandarin)
                "en": "en-us",
                "en-us": "en-us",
                "en-gb": "en-gb",
            }

            language = lang_map.get(lang_code, "en-us")
            self.phonemizers[lang_code] = AsyncPhonemizerBackend(language)

        return self.phonemizers[lang_code]

    async def normalize(self, text: str) -> str:
        """Normalize text for TTS"""
        return await self.normalizer.normalize(text)

    async def phonemize(self, text: str, lang_code: str = "a") -> str:
        """Convert text to phonemes"""
        phonemizer = await self._get_phonemizer(lang_code)
        return await phonemizer.phonemize(text)

    async def process_full_pipeline(
        self, text: str, lang_code: str = "a", normalize: bool = True
    ) -> tuple[str, str]:
        """Complete processing pipeline"""
        # Step 1: Normalization
        normalized = await self.normalize(text) if normalize else text.strip()

        # Step 2: Phonemization
        phonemes = await self.phonemize(normalized, lang_code)

        return normalized, phonemes

    async def split_text(self, text: str, max_length: int = 200) -> list[str]:
        """Split text into manageable chunks for streaming"""

        if len(text) <= max_length:
            return [text]

        # Run text splitting in thread pool for complex texts
        loop = asyncio.get_event_loop()

        def _split_sync():
            # Split on sentences first
            sentences = re.split(r"([.!?;:])\s*", text)
            chunks = []
            current_chunk = []
            current_length = 0

            for i in range(0, len(sentences), 2):
                sentence = sentences[i].strip()
                punct = sentences[i + 1] if i + 1 < len(sentences) else ""

                if not sentence:
                    continue

                full_sentence = sentence + punct

                if current_length + len(full_sentence) > max_length and current_chunk:
                    chunks.append(" ".join(current_chunk))
                    current_chunk = [full_sentence]
                    current_length = len(full_sentence)
                else:
                    current_chunk.append(full_sentence)
                    current_length += len(full_sentence)

            if current_chunk:
                chunks.append(" ".join(current_chunk))

            return chunks

        return await loop.run_in_executor(None, _split_sync)
