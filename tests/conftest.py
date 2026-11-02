from __future__ import annotations

import pytest

from chatbot.domain.models import Emotion, EmotionDistribution


@pytest.fixture
def miedo_alto() -> EmotionDistribution:
    return EmotionDistribution(
        {
            Emotion.MIEDO: 0.70,
            Emotion.TRISTEZA: 0.10,
            Emotion.NEUTRAL: 0.10,
            Emotion.IRA: 0.05,
            Emotion.ALEGRIA: 0.02,
            Emotion.SORPRESA: 0.02,
            Emotion.ASCO: 0.01,
        }
    )
