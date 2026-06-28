import logging
from typing import Optional
from urllib.parse import urlparse
import re

from lxml import html as lxml_html
from lxml.html import HtmlElement

logger = logging.getLogger(__name__)

# ── DOM-level content distillation ───────────────────────────────────────────
#
# Runs BEFORE trafilatura, not instead of it. Prunes dead subtrees (nav, ads,
# cookie banners, social widgets) structurally, then hands the cleaned HTML to
# trafilatura for its already-tuned text extraction.
#
# Weights below (tag boosts/penalties, link-density cutoff) are starting
# defaults, not derived values — there's no way to know the "right" numbers
# without running against real traffic. They're plain module constants so
# they're a one-line change once ReputationStore/coherence data shows where
# they're wrong. Don't treat these as final.

POSITIVE_TAGS = {"article", "main", "section"}
NEGATIVE_TAGS = {"nav", "aside", "footer", "form", "header"}
POSITIVE_ROLES = {"main", "article"}

NOISE_TOKENS = [
    "sidebar", "comment", "widget", "promo", "share", "related",
    "breadcrumb", "newsletter", "subscribe", "cookie", "consent",
    "gdpr", "paywall", "modal", "popup", "banner", "advert", "ad-",
    "social", "nav-", "menu", "masthead", "skip-link",
    "pagination", "tag-list", "author-box",
]

STRUCTURAL_STRIP_TAGS = {"script", "style", "noscript", "template", "svg", "iframe"}

DEFAULT_NOISE_SELECTORS = [
    ".cookie-banner", "#onetrust-consent-sdk", "[class*='cookie']",
    "[class*='consent']", "[id*='gdpr']", ".newsletter-signup",
    ".social-share", ".paywall-prompt", "[class*='modal']",
    "[class*='popup']", ".related-articles", ".comments", "#comments",
]

MIN_TEXT_LENGTH = 25
LINK_DENSITY_DISQUALIFY = 0.5


def _get_domain(url: str) -> str:
    try:
        return urlparse(url).netloc.replace("www.", "")
    except Exception:
        return ""


def _class_id_string(el: HtmlElement) -> str:
    try:
        return f"{el.get('class', '')} {el.get('id', '')}".lower()
    except Exception:
        return ""


def _has_noise_token(el: HtmlElement, tokens: list[str]) -> bool:
    s = _class_id_string(el)
    return any(tok in s for tok in tokens)


def _text_length(el: HtmlElement) -> int:
    try:
        return len(" ".join(el.itertext()).strip())
    except Exception:
        return 0


def _link_text_length(el: HtmlElement) -> int:
    try:
        return sum(len(" ".join(a.itertext()).strip()) for a in el.iter("a"))
    except Exception:
        return 0


def _link_density(el: HtmlElement) -> float:
    total = _text_length(el)
    if total == 0:
        return 1.0
    return _link_text_length(el) / total


def _tag_weight(el: HtmlElement, tokens: list[str]) -> float:
    tag = el.tag if isinstance(el.tag, str) else ""
    role = (el.get("role") or "").lower()
    weight = 1.0
    if tag in POSITIVE_TAGS or role in POSITIVE_ROLES:
        weight += 1.5
    if tag in NEGATIVE_TAGS:
        weight -= 1.5
    if _has_noise_token(el, tokens):
        weight -= 2.0
    return weight


def _score_node(el: HtmlElement, tokens: list[str]) -> float:
    text_len = _text_length(el)
    if text_len < MIN_TEXT_LENGTH:
        return 0.0
    density = _link_density(el)
    if density > LINK_DENSITY_DISQUALIFY:
        return 0.0
    weight = _tag_weight(el, tokens)
    return max(0.0, text_len * weight * (1.0 - density))


def _is_disqualified(el: HtmlElement) -> bool:
    """
    True if a node is too short or too link-dense to be real content,
    independent of tag weighting. Used to decide whether a NEGATIVE_TAGS
    element (nav/aside/footer/form/header) gets removed.

    Deliberately separate from _score_node's weighted score: that score
    clips negative-weighted-but-substantial nodes to 0.0, which would make
    a real <article><header> with a title+byline indistinguishable from an
    actual <nav> full of links. Disqualification only fires on length/link
    density, so a content-bearing header with no links survives while a
    link-only nav does not.
    """
    text_len = _text_length(el)
    if text_len < MIN_TEXT_LENGTH:
        return True
    if _link_density(el) > LINK_DENSITY_DISQUALIFY:
        return True
    return False


class DOMDistiller:
    """
    Structural pruning pass that runs before trafilatura. Never raises on
    malformed HTML — falls back to returning the input unchanged so a
    distillation bug never becomes the reason a page fails to extract at all.
    """

    def __init__(
        self,
        domain_filters=None,    # DomainFilterStore | None, injected by orchestrator
        noise_tokens: Optional[list[str]] = None,
        min_text_length: int = MIN_TEXT_LENGTH,
        link_density_disqualify: float = LINK_DENSITY_DISQUALIFY,
    ):
        self.domain_filters = domain_filters
        self.noise_tokens = noise_tokens or NOISE_TOKENS
        self.min_text_length = min_text_length
        self.link_density_disqualify = link_density_disqualify

    def distill(self, html: str, url: str = "") -> dict:
        """
        Prune a raw HTML string.
        Returns {"html": pruned_html, "removed": int, "candidate_score": float}.
        candidate_score is diagnostic only — doesn't gate anything itself;
        coherence scoring downstream already does that job.
        """
        if not html or not html.strip():
            return {"html": html, "removed": 0, "candidate_score": 0.0}

        try:
            tree = lxml_html.fromstring(html)
        except Exception:
            logger.debug("DOMDistiller: parse failed, passing html through unchanged")
            return {"html": html, "removed": 0, "candidate_score": 0.0}

        removed = 0

        # ── Structural strip — dead regardless of text ───────────────────
        for el in list(tree.iter()):
            if isinstance(el.tag, str) and el.tag in STRUCTURAL_STRIP_TAGS:
                try:
                    el.drop_tree()
                    removed += 1
                except Exception:
                    continue
            elif el.get("aria-hidden") == "true":
                try:
                    el.drop_tree()
                    removed += 1
                except Exception:
                    continue

        # ── Noise selectors — global defaults + per-domain overrides ────
        selectors = list(DEFAULT_NOISE_SELECTORS)
        if self.domain_filters and url:
            domain = _get_domain(url)
            try:
                extra = self.domain_filters.get_selectors(domain)
                if extra:
                    selectors = selectors + extra
            except Exception:
                logger.debug("DOMDistiller: domain_filters lookup failed for %s", domain)

        for selector in selectors:
            try:
                for el in tree.cssselect(selector):
                    el.drop_tree()
                    removed += 1
            except Exception:
                continue

        # ── Noise-token sweep ──────────────────────────────────────────
        for el in list(tree.iter()):
            if not isinstance(el.tag, str):
                continue
            if _has_noise_token(el, self.noise_tokens):
                try:
                    el.drop_tree()
                    removed += 1
                except Exception:
                    continue

        # ── Negative-tag pruning ──────────────────────────────────────────
        # nav/aside/footer/form/header are NOT blanket-stripped — only
        # removed when _is_disqualified() says they're too short or too
        # link-dense to be real content.
        for el in list(tree.iter()):
            if not isinstance(el.tag, str) or el.tag not in NEGATIVE_TAGS:
                continue
            if _is_disqualified(el):
                try:
                    el.drop_tree()
                    removed += 1
                except Exception:
                    continue

        # ── Score remaining candidates (diagnostic, not a gate) ──────────
        best_score = 0.0
        for el in tree.iter():
            if not isinstance(el.tag, str):
                continue
            score = _score_node(el, self.noise_tokens)
            if score > best_score:
                best_score = score

        try:
            pruned_html = lxml_html.tostring(tree, encoding="unicode")
        except Exception:
            logger.debug("DOMDistiller: serialization failed, passing html through unchanged")
            return {"html": html, "removed": 0, "candidate_score": 0.0}

        return {"html": pruned_html, "removed": removed, "candidate_score": best_score}


# ── Live-DOM extraction (Playwright path) ───────────────────────────────────
#
# Runs inside the page context via page.evaluate() so it can check
# getComputedStyle() against the live, rendered DOM — a static parser can't
# see computed visibility. Clones body, strips marked nodes from the clone,
# then un-marks the live elements — avoids matching elements between a
# detached clone and the live document by index/position.

_NOISE_TOKENS_JS_ARRAY = "[" + ",".join(f'"{t}"' for t in NOISE_TOKENS) + "]"

LIVE_DISTILL_JS = """
() => {
  const NOISE_TOKENS = """ + _NOISE_TOKENS_JS_ARRAY + """;
  const STRIP_TAGS = new Set(["script", "style", "noscript", "template", "svg", "iframe"]);

  function isHidden(el) {
    const style = window.getComputedStyle(el);
    return style.display === "none" || style.visibility === "hidden" || parseFloat(style.opacity) === 0;
  }
  function hasNoiseToken(el) {
    const raw = ((el.className && el.className.toString()) || "") + " " + (el.id || "");
    const low = raw.toLowerCase();
    return NOISE_TOKENS.some(tok => low.includes(tok));
  }

  const marked = [];
  document.body.querySelectorAll("*").forEach(el => {
    const tag = el.tagName.toLowerCase();
    if (STRIP_TAGS.has(tag) || el.getAttribute("aria-hidden") === "true"
        || hasNoiseToken(el) || isHidden(el)) {
      el.setAttribute("data-srag-remove", "1");
      marked.push(el);
    }
  });

  const clone = document.body.cloneNode(true);
  clone.querySelectorAll("[data-srag-remove]").forEach(el => el.remove());

  marked.forEach(el => el.removeAttribute("data-srag-remove"));

  return clone.innerHTML;
}
"""