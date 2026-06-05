import re

import cv2
import numpy as np

from neural_cvat.utils import resolve_torch_device

_PLATE_LETTERS = "авекмнорстух"
_LATIN_TO_CYR = str.maketrans(
    {
        "a": "а",
        "b": "в",
        "c": "с",
        "d": "д",
        "e": "е",
        "f": "р",
        "g": "г",
        "h": "н",
        "i": "и",
        "j": "й",
        "k": "к",
        "l": "л",
        "m": "м",
        "n": "н",
        "o": "о",
        "p": "р",
        "q": "к",
        "r": "р",
        "s": "с",
        "t": "т",
        "u": "у",
        "v": "в",
        "w": "в",
        "x": "х",
        "y": "у",
        "z": "з",
    },
)

_LETTER = _PLATE_LETTERS

_PLATE_PATTERNS = (
    re.compile(rf"[{_LETTER}]\d{{3}}[{_LETTER}]{{2}}\d{{2,3}}"),
    re.compile(rf"[{_LETTER}]{{2}}\d{{5}}"),
    re.compile(rf"[{_LETTER}]\d{{5}}"),
    re.compile(rf"[{_LETTER}]{{2}}\d{{4}}"),
    re.compile(rf"[{_LETTER}]{{2}}\d{{3}}[{_LETTER}]"),
    re.compile(rf"\d{{4}}[{_LETTER}]{{2}}"),
    re.compile(r"\d{5,7}"),
)

_SKIP_TOKENS = {"", "rus", "·", "."}


def normalize_plate_text(text: str) -> str:
    text = text.lower().translate(_LATIN_TO_CYR)
    return re.sub(r"[^0-9а-яё]", "", text)


def _collect_plate_matches(normalized: str) -> list[tuple[str, int, int, int]]:
    matches = []
    for idx, pattern in enumerate(_PLATE_PATTERNS):
        for match in pattern.finditer(normalized):
            matches.append((match.group(), match.start(), match.end(), idx))
    return matches


def _pick_suffix_match(normalized: str, suffix: list[tuple[str, int, int, int]]) -> str:
    l5d = [item for item in suffix if item[3] == 2]
    digits = [item for item in suffix if item[3] == 6]
    if l5d and digits:
        l5d_match, l5d_start, _, _ = max(l5d, key=lambda item: len(item[0]))
        digit_match = max(digits, key=lambda item: len(item[0]))[0]
        if l5d_start == 0:
            return l5d_match
        prefix = normalized[:l5d_start]
        has_latin = any("a" <= ch <= "z" for ch in prefix)
        if not has_latin and len(prefix) >= 2:
            return digit_match
        return l5d_match if len(l5d_match) >= len(digit_match) else digit_match
    return max(suffix, key=lambda item: len(item[0]))[0]


def extract_plate_number(text: str) -> str:
    normalized = normalize_plate_text(text)
    if not normalized:
        return ""

    for pattern in _PLATE_PATTERNS:
        if pattern.fullmatch(normalized):
            return normalized

    matches = _collect_plate_matches(normalized)
    if not matches:
        return ""

    filtered = [
        item for item in matches if not (item[3] == 0 and len(normalized) > len(item[0]) + 5)
    ]
    if filtered:
        matches = filtered

    suffix = [item for item in matches if item[2] == len(normalized)]
    if suffix:
        return _pick_suffix_match(normalized, suffix)

    return max(matches, key=lambda item: len(item[0]))[0]


def extract_ru_plate(text: str) -> str:
    return extract_plate_number(text)


def clean_ocr_token(token: str) -> str:
    cleaned = token.strip("·. \t").lower()
    if cleaned in _SKIP_TOKENS:
        return ""
    cleaned = cleaned.translate(_LATIN_TO_CYR)
    digit_chars = sum(ch.isdigit() or ch in "oо" for ch in cleaned)
    if cleaned and digit_chars / len(cleaned) >= 0.5:
        cleaned = cleaned.replace("о", "0").replace("o", "0")
    return cleaned


def join_ocr_tokens(tokens: list[str]) -> str:
    parts = [clean_ocr_token(t) for t in tokens if clean_ocr_token(t)]
    if not parts:
        return ""

    merged = [parts[0]]
    for part in parts[1:]:
        prev = merged[-1]
        if prev.isdigit() and part and part[0].isdigit() and prev[-1] == part[0]:
            merged[-1] = prev + part[1:]
        elif prev.isdigit() and len(prev) < 3 and part and part[0].isdigit():
            merged[-1] = prev + part
        else:
            merged.append(part)
    return "".join(merged)


def pad_plate_crop(crop: np.ndarray, ratio: float = 0.15) -> np.ndarray:
    if crop is None or crop.size == 0 or ratio <= 0:
        return crop
    h, w = crop.shape[:2]
    pad_x = max(2, int(w * ratio))
    pad_y = max(2, int(h * ratio))
    return cv2.copyMakeBorder(
        crop,
        pad_y,
        pad_y,
        pad_x,
        pad_x,
        cv2.BORDER_REPLICATE,
    )


def upscale_plate_crop(crop: np.ndarray, min_height: int = 128) -> np.ndarray:
    if crop is None or crop.size == 0:
        return crop
    h, w = crop.shape[:2]
    scale = max(1.0, min_height / max(h, 1))
    if scale <= 1.0:
        return crop
    return cv2.resize(
        crop,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_CUBIC,
    )


def _to_bgr(gray: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def generate_preprocessing_variants(
    crop: np.ndarray,
    min_height: int = 128,
    pad_ratio: float = 0.15,
) -> list[np.ndarray]:
    if crop is None or crop.size == 0:
        return []

    variants: list[np.ndarray] = []
    seen: set[tuple[int, int, int]] = set()

    def add(image: np.ndarray | None) -> None:
        if image is None or image.size == 0:
            return
        key = (image.shape[0], image.shape[1], int(image.mean()))
        if key in seen:
            return
        seen.add(key)
        variants.append(image)

    padded = pad_plate_crop(crop, pad_ratio)
    upscaled = upscale_plate_crop(padded, min_height)
    add(crop)
    add(padded)
    add(upscaled)

    gray = cv2.cvtColor(upscaled, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    add(_to_bgr(enhanced))

    blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
    for source in (enhanced, blurred):
        for block in (31, 41):
            for c_value in (8, 12):
                binary = cv2.adaptiveThreshold(
                    source,
                    255,
                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY,
                    block,
                    c_value,
                )
                add(_to_bgr(binary))
                add(_to_bgr(cv2.bitwise_not(binary)))
        _, otsu = cv2.threshold(source, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        add(_to_bgr(otsu))
        add(_to_bgr(cv2.bitwise_not(otsu)))

    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    add(cv2.filter2D(upscaled, -1, kernel))
    return variants


def preprocess_plate_crop(crop: np.ndarray, min_height: int = 128) -> np.ndarray:
    if crop is None or crop.size == 0:
        return crop
    crop = pad_plate_crop(crop)
    crop = upscale_plate_crop(crop, min_height)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)


def _pattern_index(plate: str) -> int:
    for idx, pattern in enumerate(_PLATE_PATTERNS):
        if pattern.fullmatch(plate):
            return idx
    return -1


def _plate_metric(plate: str, text: str, ocr_score: float) -> float:
    normalized = normalize_plate_text(text)
    if not plate:
        return -1.0

    pattern_idx = _pattern_index(plate)
    coverage = len(plate) / max(len(normalized), 1)
    if pattern_idx == 0 and coverage < 0.65:
        return -1.0

    suffix_bonus = 1.4 if normalized.endswith(plate) else 1.0
    type_bonus = 1.2 if pattern_idx > 0 else 1.0
    full_text_bonus = (
        1.5 if any(pattern.fullmatch(normalized) for pattern in _PLATE_PATTERNS) else 1.0
    )
    return (
        ocr_score * len(plate) * max(coverage, 0.25) * suffix_bonus * type_bonus * full_text_bonus
    )


def assemble_ocr_lines(results: list) -> tuple[str, float]:
    if not results:
        return "", 0.0
    ordered = sorted(results, key=lambda item: min(point[0] for point in item[0]))
    tokens = [item[1] for item in ordered]
    text = join_ocr_tokens(tokens)
    score = sum(item[2] for item in ordered) / len(ordered)
    return text, score


class PlateOCR:
    def __init__(self, ocr_config: dict | None = None) -> None:
        config = ocr_config or {}
        self.backend = config.get("backend", "rapidocr")
        self.preprocess = config.get("preprocess", True)
        self.extract_ru_plate = config.get("extract_ru_plate", True)
        self.min_height = int(config.get("min_height", 128))
        self.pad_ratio = float(config.get("pad_ratio", 0.15))
        self._reader = self._create_reader(config)

    def _create_reader(self, config: dict):
        if self.backend == "easyocr":
            import easyocr

            langs = config.get("languages") or ["ru", "en"]
            return easyocr.Reader(
                langs,
                gpu=resolve_torch_device().startswith("cuda"),
                verbose=False,
            )

        from rapidocr_onnxruntime import RapidOCR

        return RapidOCR()

    def _read_rapidocr(self, image: np.ndarray) -> tuple[str, float]:
        results, _ = self._reader(image)
        return assemble_ocr_lines(results or [])

    def _read_easyocr(self, image: np.ndarray) -> tuple[str, float]:
        results = self._reader.readtext(image)
        if not results:
            return "", 0.0
        ordered = sorted(results, key=lambda item: item[0][0][0])
        tokens = [item[1] for item in ordered]
        text = join_ocr_tokens(tokens)
        score = sum(item[2] for item in ordered) / len(ordered)
        return text, score

    def _read_variant(self, image: np.ndarray) -> tuple[str, float]:
        if self.backend == "easyocr":
            return self._read_easyocr(image)
        return self._read_rapidocr(image)

    def _build_variants(self, image: np.ndarray) -> list[np.ndarray]:
        if not self.preprocess:
            return [image]
        return generate_preprocessing_variants(
            image,
            min_height=self.min_height,
            pad_ratio=self.pad_ratio,
        )

    def _select_plate(
        self,
        image: np.ndarray,
        votes: dict[str, float],
        auto_votes: dict[str, float],
        fallback: str,
    ) -> str:
        prefer_auto = image.shape[0] < 55
        primary, secondary = (auto_votes, votes) if prefer_auto else (votes, auto_votes)
        if primary:
            return max(primary, key=primary.get)
        if secondary:
            return max(secondary, key=secondary.get)
        return fallback

    def read(self, image: np.ndarray) -> str:
        if image is None or image.size == 0:
            return ""

        votes: dict[str, float] = {}
        auto_votes: dict[str, float] = {}
        best_plate = ""
        best_metric = -1.0
        for variant in self._build_variants(image):
            text, score = self._read_variant(variant)
            if self.extract_ru_plate:
                plate = extract_plate_number(text)
            else:
                plate = normalize_plate_text(text)
            metric = _plate_metric(plate, text, score)
            if metric < 0:
                continue
            pattern_idx = _pattern_index(plate)
            bucket = auto_votes if pattern_idx == 0 else votes
            bucket[plate] = bucket.get(plate, 0.0) + metric
            if metric > best_metric:
                best_plate, best_metric = plate, metric

        return self._select_plate(image, votes, auto_votes, best_plate)
