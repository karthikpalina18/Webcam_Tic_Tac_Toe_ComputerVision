import cv2
import math
import time
import os

from mediapipe import Image, ImageFormat
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ============================================================
# CONFIGURATION
# ============================================================

CAMERA_ID = 0

MODEL_PATH = "hand_landmarker.task"

CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720

# Board position
BOARD_X = 300
BOARD_Y = 120

CELL_SIZE = 140
BOARD_SIZE = CELL_SIZE * 3

# Transparency
BOARD_ALPHA = 0.22
CELL_ALPHA = 0.18

# Finger smoothing
SMOOTHING = 0.28

# Pinch thresholds
PINCH_START = 38
PINCH_RELEASE = 55


# ============================================================
# COLORS - BGR
# ============================================================

WHITE = (255, 255, 255)
BLACK = (20, 20, 20)

RED = (40, 80, 255)
BLUE = (255, 100, 40)

GREEN = (80, 255, 120)
YELLOW = (0, 230, 255)

CYAN = (255, 220, 80)

DARK = (20, 20, 20)


# ============================================================
# CHECK MODEL
# ============================================================

if not os.path.isfile(MODEL_PATH):

    print("ERROR: hand_landmarker.task not found!")
    print()
    print("Expected:")
    print(os.path.abspath(MODEL_PATH))

    raise SystemExit


# ============================================================
# MEDIAPIPE HAND LANDMARKER
# ============================================================

base_options = python.BaseOptions(
    model_asset_path=MODEL_PATH
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,

    num_hands=1,

    min_hand_detection_confidence=0.55,
    min_hand_presence_confidence=0.55,
    min_tracking_confidence=0.55
)

detector = vision.HandLandmarker.create_from_options(
    options
)


# ============================================================
# GAME VARIABLES
# ============================================================

board = [
    ["", "", ""],
    ["", "", ""],
    ["", "", ""]
]

current_player = "X"

game_over = False

winner = ""

pinch_active = False


# ============================================================
# SMOOTHING VARIABLES
# ============================================================

smooth_x = None
smooth_y = None

smooth_thumb_x = None
smooth_thumb_y = None


# ============================================================
# RESET GAME
# ============================================================

def reset_game():

    global board
    global current_player
    global game_over
    global winner
    global pinch_active

    board = [
        ["", "", ""],
        ["", "", ""],
        ["", "", ""]
    ]

    current_player = "X"

    game_over = False

    winner = ""

    pinch_active = False


# ============================================================
# SMOOTH COORDINATE
# ============================================================

def smooth_coordinate(
    old_x,
    old_y,
    new_x,
    new_y
):

    if old_x is None:

        return new_x, new_y

    x = (
        old_x * (1 - SMOOTHING)
        +
        new_x * SMOOTHING
    )

    y = (
        old_y * (1 - SMOOTHING)
        +
        new_y * SMOOTHING
    )

    return int(x), int(y)


# ============================================================
# CHECK WINNER
# ============================================================

def check_winner():

    # Rows

    for row in board:

        if (
            row[0] != ""
            and row[0] == row[1]
            and row[1] == row[2]
        ):

            return row[0]


    # Columns

    for col in range(3):

        if (
            board[0][col] != ""
            and board[0][col] == board[1][col]
            and board[1][col] == board[2][col]
        ):

            return board[0][col]


    # Main diagonal

    if (
        board[0][0] != ""
        and board[0][0] == board[1][1]
        and board[1][1] == board[2][2]
    ):

        return board[0][0]


    # Other diagonal

    if (
        board[0][2] != ""
        and board[0][2] == board[1][1]
        and board[1][1] == board[2][0]
    ):

        return board[0][2]


    return ""


# ============================================================
# CHECK DRAW
# ============================================================

def check_draw():

    for row in board:

        for cell in row:

            if cell == "":

                return False

    return True


# ============================================================
# DRAW X
# ============================================================

def draw_x(
    frame,
    x,
    y
):

    padding = 28

    cv2.line(
        frame,

        (
            x + padding,
            y + padding
        ),

        (
            x + CELL_SIZE - padding,
            y + CELL_SIZE - padding
        ),

        BLUE,
        7,
        cv2.LINE_AA
    )

    cv2.line(
        frame,

        (
            x + CELL_SIZE - padding,
            y + padding
        ),

        (
            x + padding,
            y + CELL_SIZE - padding
        ),

        BLUE,
        7,
        cv2.LINE_AA
    )


# ============================================================
# DRAW O
# ============================================================

def draw_o(
    frame,
    x,
    y
):

    center = (
        x + CELL_SIZE // 2,
        y + CELL_SIZE // 2
    )

    cv2.circle(
        frame,
        center,
        47,
        RED,
        7,
        cv2.LINE_AA
    )


# ============================================================
# DRAW TRANSPARENT BOARD
# ============================================================

def draw_board(
    frame,
    highlighted_cell=None
):

    overlay = frame.copy()


    # --------------------------------------------------------
    # Transparent board background
    # --------------------------------------------------------

    cv2.rectangle(

        overlay,

        (
            BOARD_X,
            BOARD_Y
        ),

        (
            BOARD_X + BOARD_SIZE,
            BOARD_Y + BOARD_SIZE
        ),

        (35, 35, 35),

        -1
    )


    # --------------------------------------------------------
    # Highlight current cell
    # --------------------------------------------------------

    if highlighted_cell is not None:

        row, col = highlighted_cell

        x = BOARD_X + col * CELL_SIZE
        y = BOARD_Y + row * CELL_SIZE

        cv2.rectangle(

            overlay,

            (x, y),

            (
                x + CELL_SIZE,
                y + CELL_SIZE
            ),

            CYAN,

            -1
        )


    # --------------------------------------------------------
    # Blend transparency
    # --------------------------------------------------------

    cv2.addWeighted(
        overlay,
        BOARD_ALPHA,
        frame,
        1 - BOARD_ALPHA,
        0,
        frame
    )


    # --------------------------------------------------------
    # Grid
    # --------------------------------------------------------

    for i in range(1, 3):

        x = BOARD_X + i * CELL_SIZE

        cv2.line(

            frame,

            (
                x,
                BOARD_Y
            ),

            (
                x,
                BOARD_Y + BOARD_SIZE
            ),

            WHITE,

            3,

            cv2.LINE_AA
        )


        y = BOARD_Y + i * CELL_SIZE

        cv2.line(

            frame,

            (
                BOARD_X,
                y
            ),

            (
                BOARD_X + BOARD_SIZE,
                y
            ),

            WHITE,

            3,

            cv2.LINE_AA
        )


    # --------------------------------------------------------
    # Outer border
    # --------------------------------------------------------

    cv2.rectangle(

        frame,

        (
            BOARD_X,
            BOARD_Y
        ),

        (
            BOARD_X + BOARD_SIZE,
            BOARD_Y + BOARD_SIZE
        ),

        WHITE,

        3,

        cv2.LINE_AA
    )


    # --------------------------------------------------------
    # Draw symbols
    # --------------------------------------------------------

    for row in range(3):

        for col in range(3):

            x = BOARD_X + col * CELL_SIZE
            y = BOARD_Y + row * CELL_SIZE

            if board[row][col] == "X":

                draw_x(
                    frame,
                    x,
                    y
                )

            elif board[row][col] == "O":

                draw_o(
                    frame,
                    x,
                    y
                )


# ============================================================
# GET BOARD CELL
# ============================================================

def get_cell(
    x,
    y
):

    if not (
        BOARD_X <= x <= BOARD_X + BOARD_SIZE
        and
        BOARD_Y <= y <= BOARD_Y + BOARD_SIZE
    ):

        return None


    col = int(
        (x - BOARD_X) // CELL_SIZE
    )

    row = int(
        (y - BOARD_Y) // CELL_SIZE
    )


    if 0 <= row < 3 and 0 <= col < 3:

        return row, col


    return None


# ============================================================
# DRAW HAND
# ============================================================

def draw_hand(
    frame,
    landmarks
):

    height, width, _ = frame.shape

    points = []


    # --------------------------------------------------------
    # Landmarks
    # --------------------------------------------------------

    for landmark in landmarks:

        x = int(
            landmark.x * width
        )

        y = int(
            landmark.y * height
        )

        points.append(
            (x, y)
        )


    # --------------------------------------------------------
    # Hand connections
    # --------------------------------------------------------

    connections = [

        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),

        (0, 5),
        (5, 6),
        (6, 7),
        (7, 8),

        (0, 9),
        (9, 10),
        (10, 11),
        (11, 12),

        (0, 13),
        (13, 14),
        (14, 15),
        (15, 16),

        (0, 17),
        (17, 18),
        (18, 19),
        (19, 20)
    ]


    for start, end in connections:

        cv2.line(

            frame,

            points[start],

            points[end],

            GREEN,

            2,

            cv2.LINE_AA
        )


    # --------------------------------------------------------
    # Only important points
    # --------------------------------------------------------

    index_tip = points[8]
    thumb_tip = points[4]


    cv2.circle(
        frame,
        index_tip,
        9,
        CYAN,
        -1,
        cv2.LINE_AA
    )

    cv2.circle(
        frame,
        thumb_tip,
        7,
        YELLOW,
        -1,
        cv2.LINE_AA
    )


    return points


# ============================================================
# OPEN CAMERA
# ============================================================

cap = cv2.VideoCapture(
    CAMERA_ID
)


if not cap.isOpened():

    print("ERROR: Could not open webcam.")

    detector.close()

    raise SystemExit


# ------------------------------------------------------------
# Camera resolution
# ------------------------------------------------------------

cap.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    CAMERA_WIDTH
)

cap.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    CAMERA_HEIGHT
)


# ------------------------------------------------------------
# Reduce camera buffering
# ------------------------------------------------------------

cap.set(
    cv2.CAP_PROP_BUFFERSIZE,
    1
)


# ============================================================
# FPS
# ============================================================

previous_time = time.perf_counter()

timestamp_ms = 0


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    success, frame = cap.read()


    if not success:

        print(
            "ERROR: Could not read webcam."
        )

        break


    # --------------------------------------------------------
    # Mirror webcam
    # --------------------------------------------------------

    frame = cv2.flip(
        frame,
        1
    )


    # --------------------------------------------------------
    # RGB
    # --------------------------------------------------------

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    # --------------------------------------------------------
    # MediaPipe image
    # --------------------------------------------------------

    mp_image = Image(

        image_format=ImageFormat.SRGB,

        data=rgb_frame
    )


    # --------------------------------------------------------
    # Timestamp
    # --------------------------------------------------------

    timestamp_ms = int(
        time.monotonic() * 1000
    )


    # --------------------------------------------------------
    # Detect hand
    # --------------------------------------------------------

    result = detector.detect_for_video(

        mp_image,

        timestamp_ms
    )


    highlighted_cell = None

    distance = 0


    # ========================================================
    # HAND FOUND
    # ========================================================

    if result.hand_landmarks:

        landmarks = result.hand_landmarks[0]


        points = draw_hand(
            frame,
            landmarks
        )


        # ----------------------------------------------------
        # Raw index position
        # ----------------------------------------------------

        raw_index_x = points[8][0]
        raw_index_y = points[8][1]


        # ----------------------------------------------------
        # Smooth index position
        # ----------------------------------------------------

        smooth_x, smooth_y = smooth_coordinate(

            smooth_x,
            smooth_y,

            raw_index_x,
            raw_index_y
        )


        # ----------------------------------------------------
        # Raw thumb
        # ----------------------------------------------------

        raw_thumb_x = points[4][0]
        raw_thumb_y = points[4][1]


        smooth_thumb_x, smooth_thumb_y = smooth_coordinate(

            smooth_thumb_x,
            smooth_thumb_y,

            raw_thumb_x,
            raw_thumb_y
        )


        # ----------------------------------------------------
        # Smooth cursor
        # ----------------------------------------------------

        cv2.circle(

            frame,

            (
                smooth_x,
                smooth_y
            ),

            13,

            CYAN,

            2,

            cv2.LINE_AA
        )


        # ----------------------------------------------------
        # Pinch distance
        # ----------------------------------------------------

        distance = math.sqrt(

            (
                smooth_x
                -
                smooth_thumb_x
            ) ** 2

            +

            (
                smooth_y
                -
                smooth_thumb_y
            ) ** 2
        )


        # ====================================================
        # STABLE PINCH
        # ====================================================

        if pinch_active:

            # Release only after distance becomes larger

            if distance > PINCH_RELEASE:

                pinch_active = False


        else:

            # Start pinch only below smaller threshold

            if distance < PINCH_START:

                pinch_active = True


                # ------------------------------------------------
                # SELECT CELL
                # ------------------------------------------------

                if not game_over:

                    cell = get_cell(
                        smooth_x,
                        smooth_y
                    )


                    if cell is not None:

                        row, col = cell


                        if board[row][col] == "":

                            board[row][col] = current_player


                            # Check winner

                            result_winner = check_winner()


                            if result_winner != "":

                                winner = result_winner

                                game_over = True


                            elif check_draw():

                                winner = "DRAW"

                                game_over = True


                            else:

                                if current_player == "X":

                                    current_player = "O"

                                else:

                                    current_player = "X"


        # ----------------------------------------------------
        # Current cell
        # ----------------------------------------------------

        highlighted_cell = get_cell(
            smooth_x,
            smooth_y
        )


    # ========================================================
    # DRAW TRANSPARENT BOARD
    # ========================================================

    draw_board(

        frame,

        highlighted_cell
    )


    # ========================================================
    # TOP TITLE
    # ========================================================

    cv2.putText(

        frame,

        "WEBCAM TIC-TAC-TOE",

        (30, 45),

        cv2.FONT_HERSHEY_SIMPLEX,

        1.0,

        WHITE,

        2,

        cv2.LINE_AA
    )


    # ========================================================
    # PLAYER STATUS
    # ========================================================

    if game_over:

        if winner == "DRAW":

            status = "DRAW!"

        else:

            status = (
                f"PLAYER {winner} WINS!"
            )


        cv2.putText(

            frame,

            status,

            (780, 150),

            cv2.FONT_HERSHEY_SIMPLEX,

            1.0,

            GREEN,

            3,

            cv2.LINE_AA
        )


    else:

        status = (
            f"PLAYER {current_player}"
        )


        cv2.putText(

            frame,

            status,

            (850, 150),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.9,

            YELLOW,

            2,

            cv2.LINE_AA
        )


    # ========================================================
    # INSTRUCTIONS PANEL
    # ========================================================

    instruction_overlay = frame.copy()


    cv2.rectangle(

        instruction_overlay,

        (750, 350),

        (1240, 520),

        (20, 20, 20),

        -1
    )


    cv2.addWeighted(

        instruction_overlay,

        0.35,

        frame,

        0.65,

        0,

        frame
    )


    cv2.putText(

        frame,

        "INDEX FINGER",

        (780, 390),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.65,

        CYAN,

        2,

        cv2.LINE_AA
    )


    cv2.putText(

        frame,

        "Move over a cell",

        (780, 420),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.55,

        WHITE,

        1,

        cv2.LINE_AA
    )


    cv2.putText(

        frame,

        "PINCH",

        (780, 460),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.65,

        YELLOW,

        2,

        cv2.LINE_AA
    )


    cv2.putText(

        frame,

        "Index + Thumb = Select",

        (780, 490),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.55,

        WHITE,

        1,

        cv2.LINE_AA
    )


    # ========================================================
    # FPS
    # ========================================================

    current_time = time.perf_counter()

    elapsed = (
        current_time
        -
        previous_time
    )


    if elapsed > 0:

        fps = 1.0 / elapsed

    else:

        fps = 0


    previous_time = current_time


    cv2.putText(

        frame,

        f"FPS: {int(fps)}",

        (30, 690),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.6,

        GREEN,

        2,

        cv2.LINE_AA
    )


    # ========================================================
    # PINCH STATUS
    # ========================================================

    pinch_text = (
        "PINCHING"
        if pinch_active
        else "READY"
    )


    cv2.putText(

        frame,

        pinch_text,

        (30, 655),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.6,

        YELLOW if pinch_active else WHITE,

        2,

        cv2.LINE_AA
    )


    # ========================================================
    # KEYBOARD
    # ========================================================

    cv2.putText(

        frame,

        "R: Restart    Q: Quit",

        (30, 725),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.55,

        WHITE,

        1,

        cv2.LINE_AA
    )


    # ========================================================
    # SHOW
    # ========================================================

    cv2.imshow(

        "Smooth Transparent Webcam Tic-Tac-Toe",

        frame
    )


    # ========================================================
    # KEYBOARD INPUT
    # ========================================================

    key = cv2.waitKey(1) & 0xFF


    if key == ord("q"):

        break


    if key == ord("r"):

        reset_game()


# ============================================================
# CLEANUP
# ============================================================

cap.release()

cv2.destroyAllWindows()

detector.close()

print("Game closed.")