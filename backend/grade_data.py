"""Conservative source parsing; raw records are always retained."""
import math
import re

FEATURES = ['Geology', 'Lithology', 'Host_Rock', 'Formation', 'Age']
MISSING = {'', '-', 'nan', 'none', 'n/a', 'na', 'unknown', 'not available'}


def normalize(value):
    value = re.sub(r'\s+', ' ', str(value or '').strip()).lower()
    return 'unknown' if value in MISSING else value


def parse_grade(value):
    raw = str(value or '').strip()
    result = dict(raw=raw, kind='missing', lower_pct=None, upper_pct=None, midpoint_pct=None)
    if normalize(raw) == 'unknown':
        return result
    if re.search(r'mn\s*o\s*2', raw, re.I):
        return {**result, 'kind': 'unresolved', 'reason': 'MnO2 is not elemental Mn grade.'}
    text = raw.replace('−', '-').replace('–', '-').replace('—', '-')
    mn = re.search(r'\b(?:mn|manganese)\b', text, re.I)
    phosphorus = re.search(r'\b(?:p|phosphorus|phosphorous)\b', text, re.I)
    if phosphorus:
        clauses = re.split(r'[,;]|\band\b', text, flags=re.I)
        manganese = [c for c in clauses if re.search(r'\b(?:mn|manganese)\b', c, re.I)]
        if manganese and len(clauses) > 1:
            text = manganese[0]
        elif mn and mn.start() < phosphorus.start() and re.match(r'\s*[:=]?\s*\d', text[phosphorus.end():]):
            text = text[:phosphorus.start()]
        else:
            # Remove a separately labelled phosphorus value without consuming Mn.
            text = re.sub(r'\d+(?:\.\d+)?\s*%?\s*(?:phosphorus|phosphorous|p)\b', '', text, flags=re.I)
            text = re.sub(r'\b(?:phosphorus|phosphorous|p)\s*[:=]?\s*\d+(?:\.\d+)?\s*%?', '', text, flags=re.I)
    # No number after a lower bound may be promoted into a point target.
    lower = re.search(r'(?:\+|>=|>|≥)\s*(\d+(?:\.\d+)?)', text)
    pair = re.search(r'(\d+(?:\.\d+)?)\s*%?\s*(?:-|to)\s*(\d+(?:\.\d+)?)', text, re.I)
    nums = re.findall(r'\d+(?:\.\d+)?', text)
    if lower:
        lo, hi, kind = float(lower[1]), None, 'lower_bound'
    elif pair:
        lo, hi, kind = float(pair[1]), float(pair[2]), 'range'
    elif len(nums) == 1 and '%' in text:
        lo = hi = float(nums[0]); kind = 'exact'
    else:
        return {**result, 'kind': 'unresolved', 'reason': 'No unambiguous numeric Mn percentage.'}
    if lo < 0 or lo > 100 or (hi is not None and not lo <= hi <= 100):
        return {**result, 'kind': 'unresolved', 'reason': 'Invalid percentage bounds.'}
    flags = ['High grade: elemental/oxide basis requires verification'] if (hi or lo) > 65 else []
    return dict(raw=raw, kind=kind, lower_pct=lo, upper_pct=hi,
                midpoint_pct=(lo + hi) / 2 if hi is not None else None, quality_flags=flags)


def parse_reserve(value):
    raw = str(value or '').strip()
    if normalize(raw) == 'unknown':
        return None
    match = re.search(r'\b(?:million\s+|thousand\s+)?(?:metric\s+tonnes?|tonnes?|tons?|kt|mt|t)\b', raw, re.I)
    unit = match[0] if match else None
    return dict(original_text=raw, original_unit=unit, unit_status='as written; basis unverified' if unit else 'unspecified',
                metric_tonnes=None, provenance='Supplied CSV record; original source unverified')


def km(a, b):
    p, q = math.radians(a['latitude']), math.radians(b['latitude'])
    dl = math.radians(b['longitude'] - a['longitude'])
    v = math.sin((q-p)/2)**2 + math.cos(p)*math.cos(q)*math.sin(dl/2)**2
    return 12742 * math.asin(min(1, math.sqrt(v)))
