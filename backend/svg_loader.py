# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Mehmet Raşit Narçiçek

"""
svg_loader.py — SVG Loader & CSS Style Resolver module.
Parses SVG XML structure, resolves CSS style blocks (using tinycss2 with regex fallback), class definitions, inline styles,
and presentation attributes according to SVG priority rules. Filters out hidden/invisible elements.
Detects advanced features (clipPath, mask, fill-rule) and records warnings for report outputs.
"""

from __future__ import annotations
import re
# Use defusedxml for safe XML parsing; fall back to stdlib with a warning.
try:
    import defusedxml.ElementTree as ET  # type: ignore[import]
except ImportError:  # pragma: no cover
    import warnings as _warnings
    _warnings.warn(
        "defusedxml not installed. Falling back to stdlib xml.etree.ElementTree. "
        "Install defusedxml>=0.7.1 for XML attack protection.",
        ImportWarning, stacklevel=2,
    )
    import xml.etree.ElementTree as ET  # type: ignore[assignment]
from typing import Dict, List, Tuple, Optional

try:
    import tinycss2
    HAS_TINYCSS2 = True
except ImportError:
    HAS_TINYCSS2 = False


def parse_css_style_block(style_content: str) -> Dict[str, Dict[str, str]]:
    """
    Parses CSS <style> block content and extracts class rules.
    Uses tinycss2 if available, otherwise falls back to regex parser.
    Example:
      .cls-1 { fill: #1d1d1b; }
      .cls-2, .cls-3 { fill: none; }
      .cls-3 { stroke: #000; stroke-width: .24px; }
    """
    rules: Dict[str, Dict[str, str]] = {}
    if not style_content:
        return rules

    if HAS_TINYCSS2:
        try:
            rulesets = tinycss2.parse_stylesheet(style_content, skip_comments=True, skip_whitespace=True)
            for rule in rulesets:
                if rule.type == 'qualified-rule':
                    # Extract selector text
                    selector_str = "".join([token.serialize() for token in rule.prelude]).strip()
                    # Extract declaration block
                    declarations = tinycss2.parse_declaration_list(rule.content, skip_comments=True, skip_whitespace=True)
                    props: Dict[str, str] = {}
                    for decl in declarations:
                        if decl.type == 'declaration':
                            prop_name = decl.name.lower()
                            prop_val = "".join([token.serialize() for token in decl.value]).strip().lower()
                            props[prop_name] = prop_val

                    for sel in selector_str.split(','):
                        sel_clean = sel.strip()
                        if sel_clean.startswith('.'):
                            class_name = sel_clean[1:]
                            if class_name not in rules:
                                rules[class_name] = {}
                            rules[class_name].update(props)
            if rules:
                return rules
        except Exception as e:
            # tinycss2 failed unexpectedly; surface it as a warning and fall back
            # to the regex parser instead of silently discarding the error.
            import warnings as _warnings
            _warnings.warn(
                f"tinycss2 failed to parse CSS <style> block ({e!r}); "
                "falling back to regex CSS parser.",
                RuntimeWarning, stacklevel=2,
            )

    # Fallback Regex CSS Parser
    clean_css = re.sub(r'/\*.*?\*/', '', style_content, flags=re.DOTALL)
    blocks = re.findall(r'([^{]+)\{([^}]+)\}', clean_css)

    for selectors, props_raw in blocks:
        props: Dict[str, str] = {}
        for line in props_raw.split(';'):
            if ':' in line:
                key, val = line.split(':', 1)
                props[key.strip().lower()] = val.strip().lower()

        for sel in selectors.split(','):
            sel_clean = sel.strip()
            if sel_clean.startswith('.'):
                class_name = sel_clean[1:]
                if class_name not in rules:
                    rules[class_name] = {}
                rules[class_name].update(props)

    return rules


def parse_style_attribute(style_str: str) -> Dict[str, str]:
    """Parses inline style="..." attribute string into a dictionary."""
    props: Dict[str, str] = {}
    if not style_str:
        return props
    for item in style_str.split(';'):
        if ':' in item:
            k, v = item.split(':', 1)
            props[k.strip().lower()] = v.strip().lower()
    return props


def parse_length(val_str: str, default: float = 0.0) -> float:
    """Parses length values like '.24px', '100', '10.5mm', '12pt' to float pixels."""
    if not val_str:
        return default
    val_str = str(val_str).strip().lower()
    if val_str.endswith('px'):
        val_str = val_str[:-2]
    elif val_str.endswith('pt'):
        try:
            return float(val_str[:-2]) * 1.33333
        except ValueError:
            return default
    elif val_str.endswith('em') or val_str.endswith('rem'):
        try:
            return float(val_str[:-2]) * 16.0
        except ValueError:
            return default
    try:
        return float(val_str)
    except ValueError:
        return default


class SVGNode:
    """Represents a resolved SVG element with effective styles and geometry attributes."""
    def __init__(self, tag: str, attribs: Dict[str, str], styles: Dict[str, str], transform_str: str):
        self.tag = tag.split('}')[-1]  # Strip XML namespace if present
        self.attribs = attribs
        self.styles = styles
        self.transform_str = transform_str

        # Resolved properties
        self.fill = styles.get('fill', 'black')
        self.stroke = styles.get('stroke', 'none')
        self.stroke_width = parse_length(styles.get('stroke-width', '1'), default=1.0)
        self.opacity = float(styles.get('opacity', '1.0'))
        self.display = styles.get('display', 'inline')
        self.visibility = styles.get('visibility', 'visible')

        # Per-channel alpha: effective alpha = opacity * channel_opacity (SVG spec)
        # fill-opacity and stroke-opacity default to 1.0 when not specified.
        _fill_opacity = float(styles.get('fill-opacity', '1.0'))
        _stroke_opacity = float(styles.get('stroke-opacity', '1.0'))
        self.effective_fill_alpha = self.opacity * _fill_opacity
        self.effective_stroke_alpha = self.opacity * _stroke_opacity

        # Flags: use per-channel alpha so fill-opacity:0 correctly marks fill invisible.
        self.has_fill = (
            self.fill not in ('none', 'transparent', '')
            and self.effective_fill_alpha > 0
        )
        self.has_stroke = (
            self.stroke not in ('none', 'transparent', '')
            and self.stroke_width > 0
            and self.effective_stroke_alpha > 0
        )
        self.is_visible = (
            self.display != 'none'
            and self.visibility != 'hidden'
            and (self.has_fill or self.has_stroke)
        )


class SVGLoader:
    """Loads and parses SVG files into structured nodes with fully resolved styles and warning detection."""
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.tree = ET.parse(filepath)
        self.root = self.tree.getroot()
        self.css_rules: Dict[str, Dict[str, str]] = {}
        self.viewbox: Optional[Tuple[float, float, float, float]] = None
        self.width: float = 0.0
        self.height: float = 0.0
        self.warnings: List[str] = []

        self._parse_metadata()
        self._collect_css_styles()

    def _parse_metadata(self):
        """Extracts viewBox, width, and height from root <svg> tag."""
        root_attribs = self.root.attrib

        # Case-insensitive attribute lookup
        attr_map = {k.lower(): v for k, v in root_attribs.items()}

        # viewBox: "minX minY width height"
        if 'viewbox' in attr_map:
            parts = [float(p) for p in re.split(r'[\s,]+', attr_map['viewbox'].strip()) if p]
            if len(parts) == 4:
                self.viewbox = (parts[0], parts[1], parts[2], parts[3])

        w_str = attr_map.get('width', '')
        h_str = attr_map.get('height', '')
        self.width = parse_length(w_str, default=0.0)
        self.height = parse_length(h_str, default=0.0)

        if not self.viewbox and self.width > 0 and self.height > 0:
            self.viewbox = (0.0, 0.0, self.width, self.height)
            self.warnings.append(f"viewBox attribute missing. Defaulted to (0, 0, {self.width}, {self.height}) from width/height.")
        elif self.viewbox and (self.width == 0 or self.height == 0):
            self.width = self.viewbox[2]
            self.height = self.viewbox[3]

    def _collect_css_styles(self):
        """Extracts all <style> tag contents from SVG."""
        for elem in self.root.iter():
            tag = elem.tag.split('}')[-1]
            if tag == 'style' and elem.text:
                parsed = parse_css_style_block(elem.text)
                for cls_name, props in parsed.items():
                    if cls_name not in self.css_rules:
                        self.css_rules[cls_name] = {}
                    self.css_rules[cls_name].update(props)

    def get_elements(self) -> List[Tuple[SVGNode, List[str]]]:
        """
        Traverses SVG tree and returns a list of (SVGNode, transform_stack).
        Applies style priority: Inline style > Presentation Attrs > CSS Class > Inherited.
        Detects clipPath, mask, and fill-rule features for warnings.
        """
        elements: List[Tuple[SVGNode, List[str]]] = []
        self._traverse_node(self.root, parent_styles={}, transform_stack=[], results=elements)
        return elements

    def _traverse_node(
        self,
        elem: ET.Element,
        parent_styles: Dict[str, str],
        transform_stack: List[str],
        results: List[Tuple[SVGNode, List[str]]]
    ):
        tag = elem.tag.split('}')[-1]
        
        # Feature detection warnings
        if tag == 'clipPath':
            self.warnings.append("clipPath element detected. Clipping geometry bounds are ignored in core v1.0.")
            return
        if tag == 'mask' or 'mask' in elem.attrib:
            self.warnings.append("mask attribute/element detected. Alpha masking is ignored in core v1.0.")
            return
        if elem.attrib.get('fill-rule') == 'evenodd':
            self.warnings.append("fill-rule='evenodd' detected. Default non-zero winding rule used in core v1.0.")

        if tag in ('defs', 'symbol'):
            return  # Skip definitions and non-rendered templates

        # Inherit parent styles
        effective_styles = parent_styles.copy()

        # 1. CSS Class rules
        class_attr = elem.attrib.get('class', '')
        if class_attr:
            for cls_name in class_attr.split():
                if cls_name in self.css_rules:
                    effective_styles.update(self.css_rules[cls_name])

        # 2. Presentation Attributes (fill, stroke, stroke-width, etc.)
        for attr_key, attr_val in elem.attrib.items():
            if attr_key in ('fill', 'stroke', 'stroke-width', 'opacity', 'display', 'visibility', 'fill-opacity', 'stroke-opacity'):
                effective_styles[attr_key.lower()] = attr_val.lower()

        # 3. Inline style attribute (highest priority)
        inline_style_str = elem.attrib.get('style', '')
        if inline_style_str:
            effective_styles.update(parse_style_attribute(inline_style_str))

        # Transform attribute handling
        node_transform = elem.attrib.get('transform', '')
        current_transform_stack = list(transform_stack)
        if node_transform:
            current_transform_stack.append(node_transform)

        # Check if rendering element
        render_tags = {'path', 'rect', 'circle', 'ellipse', 'line', 'polyline', 'polygon'}
        if tag in render_tags:
            node = SVGNode(tag, elem.attrib, effective_styles, node_transform)
            if node.is_visible:
                results.append((node, current_transform_stack))

        # Recurse into children (e.g. <g> groups)
        for child in elem:
            self._traverse_node(child, effective_styles, current_transform_stack, results)
