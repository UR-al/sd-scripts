"""Image loading and color-space utilities for sd_scripts.

Provides :func:`to_srgb`, an ICC-profile-aware replacement for bare
``Image.convert("RGB")`` calls.  When an image carries an embedded ICC
profile the conversion honours it; otherwise a sensible default
(``"sRGB"``) is assumed.
"""

import io
import logging

from PIL import Image, ImageCms

logger = logging.getLogger(__name__)

# Cache the sRGB profile at module level so it is created exactly once.
_SRGB_PROFILE = ImageCms.createProfile("sRGB")
_SRGB_CMS = ImageCms.ImageCmsProfile(_SRGB_PROFILE)

# Progressively less strict (flags, rendering-intent) pairs.  Some ICC
# profiles (V4, wide-gamut, CMYK, etc.) reject BLACKPOINTCOMPENSATION
# or only define certain rendering intents, so we try several combos
# before giving up.
_TRANSFORM_ATTEMPTS = [
    (ImageCms.Flags.BLACKPOINTCOMPENSATION, ImageCms.Intent.PERCEPTUAL),
    (0, ImageCms.Intent.PERCEPTUAL),
    (ImageCms.Flags.BLACKPOINTCOMPENSATION, ImageCms.Intent.RELATIVE_COLORIMETRIC),
    (0, ImageCms.Intent.RELATIVE_COLORIMETRIC),
]

# Maps ICC colour-space signatures (bytes 16-19 of the ICC header) to
# the PIL image modes that are *compatible* with that colour-space.
# littlecms cannot build a transform when the image mode and the ICC
# colour-space are fundamentally incompatible (e.g. greyscale pixel
# data with an RGB profile).
_CS_TO_MODES = {
    "RGB ": ("RGB", "RGBA"),
    "GRAY": ("L", "LA"),
    "CMYK": ("CMYK",),
}


def _extract_icc_color_space(src_profile, icc_bytes: bytes | None = None) -> str:
    """Return the ICC colour-space signature, or ``""`` if unknown.

    Tries the ``ImageCms`` API first, then falls back to reading the
    4-byte colour-space field directly from the raw ICC header (bytes
    16-19 per ICC.1:2022 §7.2.6).
    """
    try:
        cs = ImageCms.getColorSpace(src_profile)
        if cs:
            return cs.strip()
    except Exception:
        pass

    # Fallback: raw ICC header from bytes or from profile object.
    raw = icc_bytes
    if raw is None and src_profile is not None:
        try:
            raw = src_profile.tobytes()
        except Exception:
            pass
    if raw is not None and len(raw) >= 20:
        try:
            return raw[16:20].decode("ascii", errors="replace")
        except Exception:
            pass

    return ""


def _mode_matches_profile(im_mode: str, color_space: str) -> bool:
    """Return ``True`` if *im_mode* is compatible with *color_space*.

    If *color_space* is unrecognised the function returns ``True``
    (benefit of the doubt — we'll let littlecms try).
    """
    expected = _CS_TO_MODES.get(color_space)
    if expected is None:
        return True  # unknown colour-space → don't block the attempt
    return im_mode in expected


def _log_profile_failure(im: Image.Image, src_profile=None) -> None:
    """Log diagnostic information about a failed ICC conversion.

    Extracts as much detail as possible from both the PIL image object
    (format, filename, mode, size) and the ICC source profile (name,
    class, colour-space, byte-size) so that the root cause — typically a
    mode/profile mismatch such as a greyscale image carrying an sRGB
    profile — can be identified without needing to reproduce the exact
    image.
    """
    # --- image metadata ---
    img_format = getattr(im, "format", None) or "<unknown>"
    img_filename = getattr(im, "filename", None) or "<unknown>"

    # --- ICC profile details (each call independent so one failure
    #     doesn't suppress the others) ---
    profile_name = profile_class = "<unavailable>"
    color_space = ""
    icc_size = 0
    if src_profile is not None:
        try:
            profile_name = ImageCms.getProfileName(src_profile).strip()
        except Exception:
            pass
        try:
            profile_class = ImageCms.getProfileClass(src_profile).strip()
        except Exception:
            pass
        try:
            icc_size = len(src_profile.tobytes())
        except Exception:
            pass

    icc_bytes = im.info.get("icc_profile")
    color_space = _extract_icc_color_space(src_profile, icc_bytes) or "<unavailable>"
    if icc_bytes:
        icc_size = icc_size or len(icc_bytes)

    reason = ""
    if color_space != "<unavailable>" and not _mode_matches_profile(im.mode, color_space):
        reason = (
            f" — likely cause: image mode={im.mode} does not match "
            f"ICC profile colour-space={color_space!r}; "
            f"littlecms cannot build a transform across these types"
        )

    logger.warning(
        "ICC-aware colour conversion failed for image mode=%s, size=%s, "
        "format=%s, filename=%s; "
        "ICC profile: name=%r, class=%r, colour-space=%r, bytes=%d%s — "
        "falling back to plain .convert('RGB')",
        im.mode,
        im.size,
        img_format,
        img_filename,
        profile_name,
        profile_class,
        color_space,
        icc_size,
        reason,
    )


def to_srgb(im: Image.Image, assume: str = "sRGB") -> Image.Image:
    """Convert *im* to sRGB, respecting any embedded ICC profile.

    Parameters
    ----------
    im : PIL.Image.Image
        Source image in any mode (RGB, RGBA, CMYK, L, …).
    assume : str
        Colour-space name to use when the image carries **no** embedded
        ICC profile.  Must be a string accepted by
        ``ImageCms.createProfile`` (e.g. ``"sRGB"``, ``"AdobeRGB"``).
        Default is ``"sRGB"`` — meaning no conversion is needed for
        typical web images.

    Returns
    -------
    PIL.Image.Image
        An ``"RGB"`` image in the sRGB colour space.

    Notes
    -----
    * CMYK images are converted to RGB through the ICC pipeline when a
      CMYK profile is present (or the *assume* fallback).  PIL's
      ``profileToProfile`` handles the CMYK → RGB channel reduction
      natively when ``outputMode="RGB"``.
    * If the image mode and the embedded ICC profile colour-space are
      incompatible (e.g. a greyscale image carrying an sRGB profile),
      the ICC pipeline is skipped entirely and a plain
      ``im.convert("RGB")`` is returned without logging a warning.
    * The function tries several (flags, rendering-intent) combinations
      before falling back, so that images carrying unusual ICC profiles
      (V4, wide-gamut, etc.) still get correct colour conversion when
      possible.
    * On any failure a plain ``im.convert("RGB")`` is returned so
      callers never need to handle exceptions.
    """
    # Fast path: already RGB and sRGB — nothing to do.
    if im.mode == "RGB" and not im.info.get("icc_profile"):
        return im

    icc_bytes = im.info.get("icc_profile")

    # --- proactive mismatch check ---
    # If the image carries an ICC profile whose colour-space is
    # incompatible with the image mode (e.g. greyscale pixels + sRGB
    # profile), littlecms will *never* be able to build a transform.
    # Detect this and convert silently rather than trying all four
    # fallback combos and emitting a noisy warning.
    if icc_bytes:
        try:
            src = ImageCms.ImageCmsProfile(io.BytesIO(icc_bytes))
        except Exception:
            # Profile bytes are corrupt — plain convert is the only option.
            return im.convert("RGB")

        cs = _extract_icc_color_space(src, icc_bytes)
        if cs and not _mode_matches_profile(im.mode, cs):
            logger.debug(
                "Image mode=%s is incompatible with ICC colour-space=%r; "
                "skipping ICC pipeline and converting directly to RGB",
                im.mode,
                cs,
            )
            return im.convert("RGB")
    else:
        src = ImageCms.createProfile(assume)

    # --- convert via the ICC pipeline ---
    try:
        for flags, intent in _TRANSFORM_ATTEMPTS:
            try:
                result = ImageCms.profileToProfile(
                    im,
                    src,
                    _SRGB_PROFILE,
                    outputMode="RGB",
                    renderingIntent=intent,
                    flags=flags,
                )
                # Tag the result so downstream consumers know the colour space.
                result.info["icc_profile"] = _SRGB_CMS.tobytes()
                return result
            except Exception:
                continue

        # All ICC transform attempts failed — fall back.
        _log_profile_failure(im, src)
        return im.convert("RGB")

    except Exception:
        _log_profile_failure(im, src)
        return im.convert("RGB")


_ALPHA_MODES = ("RGBA", "LA", "PA", "RGBa", "La")


def flatten_alpha(im: Image.Image, background=(255, 255, 255)) -> Image.Image:
    """Composite an image with transparency onto a solid background (white by default).

    ``to_srgb`` and ``convert("RGB")`` drop the alpha channel, so transparent areas would train as whatever colour the
    file stores under them (often black). Covers alpha modes and palette/colour-key transparency (PNG tRNS). Images
    without transparency, fully opaque RGBA included, are returned unchanged (the same object). The ICC profile is kept
    for :func:`to_srgb`.
    """
    if im.mode not in _ALPHA_MODES and "transparency" not in im.info:
        return im
    rgba = im.convert("RGBA")
    if rgba.getchannel("A").getextrema()[0] == 255:
        return im
    canvas = Image.new("RGBA", rgba.size, tuple(background) + (255,))
    out = Image.alpha_composite(canvas, rgba).convert("RGB")
    icc = im.info.get("icc_profile")
    if icc:
        out.info["icc_profile"] = icc
    return out
