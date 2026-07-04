"""Shared constants for gesture recognition and UI rendering."""

# Gesture must be held this many consecutive frames before it is accepted.
HOLD_FRAMES = 10

# Shorter hold in conference mode for faster conversation flow.
CONFERENCE_HOLD_FRAMES = 8

# Ignore the same gesture if it was just added within this many frames.
COOLDOWN_FRAMES = 18

# Caption panel height in pixels.
CAPTION_BAR_HEIGHT = 120

# Large caption strip for Zoom/Meet overlay window.
CONFERENCE_CAPTION_HEIGHT = 160

# Colors (BGR)
COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (0, 0, 0)
COLOR_GREEN = (0, 200, 80)
COLOR_CYAN = (255, 220, 80)
COLOR_DARK_BG = (30, 30, 30)
COLOR_PREVIEW = (180, 180, 180)

# MediaPipe hand landmark indices
WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20
