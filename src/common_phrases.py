"""Common words and phrases for everyday communication."""

from __future__ import annotations

import numpy as np

from src.constants import INDEX_TIP, MIDDLE_TIP, PINKY_TIP, RING_TIP, THUMB_TIP, WRIST
from src.finger_geometry import (
    count_extended,
    get_detailed_finger_states,
    fingertips_pinch,
    index_middle_spread,
    index_middle_together,
    palm_size,
    pointing_horizontal,
)


PHRASE_LABELS: dict[str, str] = {
    "HELLO": "Hello",
    "HI": "Hi",
    "GOODBYE": "Goodbye",
    "THANK_YOU": "Thank you",
    "PLEASE": "Please",
    "SORRY": "Sorry",
    "EXCUSE_ME": "Excuse me",
    "YES": "Yes",
    "NO": "No",
    "MAYBE": "Maybe",
    "OK": "OK",
    "NICE": "Nice to meet you",
    "YOU": "You",
    "ME": "Me",
    "I": "I",
    "WE": "We",
    "THEY": "They",
    "LOVE": "I love you",
    "HAPPY": "Happy",
    "SAD": "Sad",
    "ANGRY": "Angry",
    "TIRED": "Tired",
    "HURT": "It hurts",
    "GOOD": "Good",
    "BAD": "Bad",
    "FINE": "I'm fine",
    "HELP": "Help",
    "STOP": "Stop",
    "WAIT": "Wait",
    "COME": "Come here",
    "GO": "Go",
    "LOOK": "Look",
    "LISTEN": "Listen",
    "UNDERSTAND": "I understand",
    "DONT_UNDERSTAND": "I don't understand",
    "REPEAT": "Please repeat",
    "WRITE": "Please write it",
    "WATER": "Water",
    "FOOD": "Food",
    "BATHROOM": "Bathroom",
    "MEDICINE": "Medicine",
    "DOCTOR": "Doctor",
    "EMERGENCY": "Emergency",
    "PAIN": "Pain",
    "HUNGRY": "Hungry",
    "THIRSTY": "Thirsty",
    "NOW": "Now",
    "LATER": "Later",
    "TODAY": "Today",
    "TOMORROW": "Tomorrow",
    "HERE": "Here",
    "THERE": "There",
    "NAME": "My name is",
    "QUESTION": "I have a question",
    "ANSWER": "Answer",
    "AGREE": "I agree",
    "DISAGREE": "I disagree",
    "MUTE": "Please mute",
    "UNMUTE": "Please unmute",
    "CAN_YOU_HEAR": "Can you hear me",
    "CAN_YOU_SEE": "Can you see my captions",
    "THANKS_WAIT": "Thank you for waiting",
    "ONE_MOMENT": "One moment please",
}


def classify_phrase(landmarks: list[tuple], handedness: str) -> str | None:
    """
    Detect common word/phrase gestures with exact parity to the browser rules.

    Only clean, non-colliding geometric shapes are matched:
    - FOOD: gathered fingertips toward mouth
    - OK: thumb & index pinch ring, other three extended
    - LOVE: thumb, index and pinky extended (I-love-you)
    - HELP: index and pinky extended without thumb (horns)
    - YES: thumb alone pointing upward
    - NO: thumb alone pointing downward
    - WATER: thumb, index, middle and ring extended (W with thumb)
    - YOU: index pointing horizontal toward person
    - THANK_YOU: flat hand fingers together pointing horizontal
    - HELLO: open hand fingers spread
    - PLEASE: open hand fingers together
    """
    s = get_detailed_finger_states(landmarks, handedness)
    thumb, index, middle, ring, pinky = s
    ext = count_extended(s)
    palm = palm_size(landmarks)
    curled = lambda f: f != "extended"

    # --- gathered fingertips: pinching food toward the mouth
    tips = (INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP)
    cx = sum(landmarks[t][0] for t in tips) / len(tips)
    cy = sum(landmarks[t][1] for t in tips) / len(tips)
    gathered = all(
        ((landmarks[t][0] - cx) ** 2 + (landmarks[t][1] - cy) ** 2) ** 0.5 < palm * 0.5
        for t in tips
    )
    if gathered and thumb == "curled" and ext == 0:
        return "FOOD"

    # --- OK ring: thumb and index meet, the other three stay up
    if (
        fingertips_pinch(landmarks, THUMB_TIP, INDEX_TIP)
        and middle == "extended"
        and ring == "extended"
        and pinky == "extended"
    ):
        return "OK"

    # --- I-love-you: thumb, index and pinky out together
    if (
        thumb == "extended"
        and index == "extended"
        and pinky == "extended"
        and curled(middle)
        and curled(ring)
    ):
        return "LOVE"

    # --- horns: the same two fingers without the thumb, so the pair never
    #     collides with I-love-you
    if (
        index == "extended"
        and pinky == "extended"
        and thumb != "extended"
        and curled(middle)
        and curled(ring)
    ):
        return "HELP"

    # --- thumb alone: up means yes, down means no
    if thumb == "extended" and ext == 0:
        return "YES" if landmarks[THUMB_TIP][1] < landmarks[WRIST][1] else "NO"

    # --- W with the thumb out, so it reads as water rather than the letter
    if (
        thumb == "extended"
        and index == "extended"
        and middle == "extended"
        and ring == "extended"
        and pinky != "extended"
    ):
        return "WATER"

    # --- pointing sideways, at the person you are talking to
    if (
        index == "extended"
        and ext == 1
        and curled(middle)
        and curled(ring)
        and curled(pinky)
        and pointing_horizontal(landmarks, handedness)
    ):
        return "YOU"

    # --- a flat hand moving out from the chin
    if (
        index == "extended"
        and middle == "extended"
        and ring == "extended"
        and pinky == "extended"
        and index_middle_together(landmarks)
        and pointing_horizontal(landmarks, handedness)
    ):
        return "THANK_YOU"

    # --- an open hand held up: spread waves hello, together asks please
    if all(f == "extended" for f in s):
        return "HELLO" if index_middle_spread(landmarks) else "PLEASE"

    return None
