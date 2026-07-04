"""Common words and phrases for everyday communication."""

from __future__ import annotations

import numpy as np

from src.constants import INDEX_TIP, MIDDLE_TIP, PINKY_TIP, RING_TIP, THUMB_TIP, WRIST
from src.finger_geometry import (
    all_fingertips_touch,
    count_extended,
    get_detailed_finger_states,
    fingertips_pinch,
    index_middle_spread,
    index_middle_together,
    palm_size,
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


def _dist_to_point(landmarks: list[tuple], idx: int, point: np.ndarray) -> float:
    return float(np.linalg.norm(np.array(landmarks[idx][:2]) - point))


def classify_phrase(landmarks: list[tuple], handedness: str) -> str | None:
    """Detect common word/phrase gestures."""
    s = get_detailed_finger_states(landmarks, handedness)
    thumb, index, middle, ring, pinky = s
    ext = count_extended(s)
    palm = palm_size(landmarks)

    if thumb == "extended" and index == "extended" and pinky == "extended" and middle == "curled" and ring == "curled":
        return "LOVE"

    if thumb == "extended" and ext == 0:
        if landmarks[THUMB_TIP][1] < landmarks[WRIST][1]:
            return "YES"
        return "NO"

    if fingertips_pinch(landmarks, THUMB_TIP, INDEX_TIP) and middle == "extended" and ring == "extended" and pinky == "extended":
        return "OK"

    if ext == 0 and thumb != "extended":
        return "PLEASE"

    if thumb == "extended" and index == "extended" and middle == "extended" and ring == "curled" and pinky == "curled":
        return "HELP"

    if index == "extended" and ext == 1 and thumb != "extended":
        return "YOU"

    if index == "extended" and middle == "extended" and ring == "extended" and pinky == "extended" and index_middle_together(landmarks) and thumb in ("bent", "curled"):
        return "ME"

    if thumb == "extended" and index == "extended" and middle == "extended" and ring == "extended" and pinky == "curled":
        return "WATER"

    tips = (INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP)
    center = np.mean([landmarks[t][:2] for t in tips], axis=0)
    if all(_dist_to_point(landmarks, t, center) < palm * 0.5 for t in tips) and thumb == "curled":
        return "FOOD"

    if thumb == "bent" and ext == 0:
        return "BATHROOM"

    if index == "bent" and middle == "curled" and ring == "curled" and pinky == "curled" and thumb == "bent":
        return "SORRY"

    if all(f == "extended" for f in s):
        return "HELLO"

    if index == "extended" and middle == "extended" and ring == "extended" and pinky == "extended" and index_middle_spread(landmarks):
        return "EMERGENCY"

    if thumb == "extended" and index == "curled" and middle == "curled" and ring == "curled" and pinky == "curled":
        if landmarks[THUMB_TIP][1] < landmarks[WRIST][1]:
            return "GOOD"
        return "BAD"

    bent_count = sum(1 for f in s[1:] if f == "bent")
    if bent_count >= 2 and thumb != "extended" and 2 <= ext <= 3:
        return "HUNGRY"

    if index == "bent" and middle == "curled" and ring == "curled" and pinky == "curled" and thumb == "extended":
        return "QUESTION"

    if index == "extended" and ext == 1 and thumb == "curled":
        return "ONE_MOMENT"

    if ext == 3 and pinky == "curled" and thumb == "bent":
        return "CAN_YOU_HEAR"

    if index == "bent" and middle == "bent" and ring == "bent" and pinky == "extended" and thumb == "curled":
        return "HI"

    if index == "extended" and middle == "extended" and ring == "curled" and pinky == "curled" and thumb == "extended":
        return "GOODBYE"

    if index == "extended" and middle == "extended" and ring == "extended" and pinky == "extended" and thumb != "extended":
        return "THANK_YOU"

    if index == "bent" and middle == "curled" and ring == "curled" and pinky == "curled" and thumb == "curled":
        return "UNDERSTAND"

    if index == "bent" and middle == "bent" and ring == "curled" and pinky == "curled":
        return "DONT_UNDERSTAND"

    if index == "bent" and ext == 0 and thumb == "extended":
        return "REPEAT"

    if thumb == "curled" and index == "bent" and middle == "bent":
        return "NAME"

    if all_fingertips_touch(landmarks, ratio=0.32) and thumb == "curled":
        return "PAIN"

    if thumb == "curled" and index == "bent" and middle == "bent" and ring == "bent":
        return "MEDICINE"

    if index == "extended" and ext == 1 and thumb == "bent":
        return "DOCTOR"

    if thumb == "extended" and index == "extended" and middle == "curled" and ring == "curled" and pinky == "curled":
        return "LATER"

    if thumb == "extended" and pinky == "extended" and index == "curled" and middle == "curled" and ring == "curled":
        return "NOW"

    if thumb == "extended" and pinky == "extended" and index == "curled" and middle == "curled" and ring == "curled":
        return "CAN_YOU_SEE"

    if index == "extended" and middle == "curled" and ring == "curled" and pinky == "curled" and thumb == "curled":
        return "WAIT"

    if index == "extended" and middle == "extended" and ring == "curled" and pinky == "curled" and thumb == "curled":
        return "COME"

    if index == "curled" and middle == "extended" and ring == "curled" and pinky == "curled":
        return "GO"

    if index == "extended" and middle == "extended" and ring == "extended" and pinky == "curled" and thumb == "curled":
        return "LOOK"

    if pinky == "extended" and index == "curled" and middle == "curled" and ring == "curled" and thumb == "bent":
        return "LISTEN"

    if thumb == "extended" and index == "curled" and middle == "curled" and ring == "extended" and pinky == "curled":
        return "FINE"

    if index == "extended" and middle == "extended" and ring == "curled" and pinky == "curled" and thumb == "extended":
        return "AGREE"

    if index == "curled" and middle == "curled" and ring == "extended" and pinky == "extended" and thumb == "curled":
        return "DISAGREE"

    if index == "bent" and middle == "bent" and ring == "bent" and pinky == "curled" and thumb == "curled":
        return "WRITE"

    if thumb == "curled" and index == "curled" and middle == "curled" and ring == "curled" and pinky == "extended":
        return "THIRSTY"

    if index == "extended" and middle == "curled" and ring == "curled" and pinky == "curled" and thumb == "bent":
        return "HERE"

    if index == "curled" and middle == "curled" and ring == "curled" and pinky == "extended" and thumb == "bent":
        return "THERE"

    if index == "extended" and middle == "extended" and ring == "curled" and pinky == "curled" and thumb == "bent":
        return "TODAY"

    if index == "extended" and middle == "curled" and ring == "extended" and pinky == "curled" and thumb == "bent":
        return "TOMORROW"

    if index == "bent" and middle == "curled" and ring == "curled" and pinky == "curled" and thumb == "curled":
        return "HURT"

    if index == "extended" and middle == "curled" and ring == "curled" and pinky == "curled" and thumb == "extended":
        return "HAPPY"

    if index == "curled" and middle == "extended" and ring == "curled" and pinky == "curled" and thumb == "curled":
        return "SAD"

    if index == "curled" and middle == "curled" and ring == "extended" and pinky == "curled" and thumb == "extended":
        return "ANGRY"

    if index == "bent" and middle == "bent" and ring == "curled" and pinky == "curled" and thumb == "curled":
        return "TIRED"

    if thumb == "extended" and index == "curled" and middle == "curled" and ring == "curled" and pinky == "curled":
        return "MUTE"

    if thumb == "extended" and index == "extended" and middle == "curled" and ring == "curled" and pinky == "curled":
        return "UNMUTE"

    if index == "extended" and middle == "extended" and ring == "curled" and pinky == "bent" and thumb == "curled":
        return "ANSWER"

    if index == "bent" and middle == "extended" and ring == "curled" and pinky == "curled" and thumb == "curled":
        return "MAYBE"

    if index == "extended" and middle == "extended" and ring == "extended" and pinky == "extended" and thumb == "extended":
        return "NICE"

    return None
