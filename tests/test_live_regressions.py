"""Regressions reproduced during Telegram smoke testing."""
import pytest

from services.vision import should_use_latest_image


@pytest.mark.parametrize('message', [
    'What color and shape are in the latest image? Answer briefly.',
    'Describe my last uploaded photo.',
    'Read the most recent screenshot.',
    'WHAT IS IN MY LATEST PICTURE?',
])
def test_latest_image_wording_routes_to_vision(message):
    assert should_use_latest_image(message)


@pytest.mark.parametrize('message', [
    'What is the latest Python release?',
    'Find my last task about guitar.',
    'Search the most recent notes.',
])
def test_latest_unrelated_items_do_not_route_to_vision(message):
    assert not should_use_latest_image(message)
