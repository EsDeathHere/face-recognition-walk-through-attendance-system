# -*- coding: utf-8 -*-
"""Recognition helpers: matching best frames to multi-angle gallery."""

import time
import numpy as np

from database import get_all_profiles
from utils import cosine_similarity, normalize_emb
from config import MATCH_THRESHOLD


def refresh_gallery():
    """Load gallery from DB: list of (user_id, name, embedding)."""
    return get_all_profiles()


def match_embedding(emb, gallery, threshold=None):
    """Return (user_id, name, best_score) or (None, None, best_score)."""
    if threshold is None:
        threshold = MATCH_THRESHOLD
    emb = normalize_emb(emb)
    best_score = 0.0
    best_user = None
    best_name = None
    for uid, name, gemb in gallery:
        s = cosine_similarity(emb, gemb)
        if s > best_score:
            best_score = s
            best_user = uid
            best_name = name
    if best_user is not None and best_score >= threshold:
        return best_user, best_name, best_score
    return None, None, best_score


def cooldown_ok(user_id):
    from database import can_attend
    return can_attend(user_id)

