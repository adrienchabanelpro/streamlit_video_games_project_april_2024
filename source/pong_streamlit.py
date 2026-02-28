"""Pong -- Streamlit-native game (no Pygame, works on cloud deployment).

Uses PIL for rendering and st.session_state for game state persistence.
Player controls the bottom paddle with buttons; a simple AI controls the top paddle.
"""

from io import BytesIO

import streamlit as st
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# Game constants
# ---------------------------------------------------------------------------
CANVAS_W: int = 600
CANVAS_H: int = 400
PADDLE_W: int = 80
PADDLE_H: int = 12
BALL_SIZE: int = 10
BG_COLOR: str = "#0D0D0D"
CYAN: str = "#00FFCC"
PINK: str = "#FF6EC7"
YELLOW: str = "#FFFF00"
WHITE: str = "#E0E0E0"
BALL_SPEED: float = 6.0
PADDLE_SPEED: int = 30
AI_SPEED: int = 4
WIN_SCORE: int = 5


# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------


def _init_state() -> None:
    """Initialise all game state variables in st.session_state."""
    defaults: dict = {
        "pong_ball_x": CANVAS_W / 2,
        "pong_ball_y": CANVAS_H / 2,
        "pong_ball_dx": BALL_SPEED,
        "pong_ball_dy": BALL_SPEED,
        "pong_player_x": CANVAS_W / 2 - PADDLE_W / 2,
        "pong_ai_x": CANVAS_W / 2 - PADDLE_W / 2,
        "pong_player_score": 0,
        "pong_ai_score": 0,
        "pong_running": False,
        "pong_game_over": False,
        "pong_winner": "",
        "pong_last_tick": 0.0,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def _reset_game() -> None:
    """Reset all game state for a fresh round."""
    st.session_state.pong_ball_x = CANVAS_W / 2
    st.session_state.pong_ball_y = CANVAS_H / 2
    st.session_state.pong_ball_dx = BALL_SPEED
    st.session_state.pong_ball_dy = BALL_SPEED
    st.session_state.pong_player_x = CANVAS_W / 2 - PADDLE_W / 2
    st.session_state.pong_ai_x = CANVAS_W / 2 - PADDLE_W / 2
    st.session_state.pong_player_score = 0
    st.session_state.pong_ai_score = 0
    st.session_state.pong_running = True
    st.session_state.pong_game_over = False
    st.session_state.pong_winner = ""


def _reset_ball() -> None:
    """Reset ball to center after a point is scored."""
    st.session_state.pong_ball_x = CANVAS_W / 2
    st.session_state.pong_ball_y = CANVAS_H / 2
    # Reverse direction towards the scorer's opponent
    st.session_state.pong_ball_dy = -st.session_state.pong_ball_dy


# ---------------------------------------------------------------------------
# Game logic (one tick per interaction)
# ---------------------------------------------------------------------------


def _tick() -> None:
    """Advance the game by one frame."""
    if not st.session_state.pong_running or st.session_state.pong_game_over:
        return

    s = st.session_state  # shorthand

    # --- Move ball ---
    s.pong_ball_x += s.pong_ball_dx
    s.pong_ball_y += s.pong_ball_dy

    # --- Wall bounces (left/right) ---
    if s.pong_ball_x <= 0 or s.pong_ball_x >= CANVAS_W - BALL_SIZE:
        s.pong_ball_dx = -s.pong_ball_dx

    # --- Player paddle collision (bottom) ---
    if s.pong_ball_y >= CANVAS_H - PADDLE_H - BALL_SIZE - 10:
        if s.pong_player_x <= s.pong_ball_x <= s.pong_player_x + PADDLE_W:
            s.pong_ball_dy = -abs(s.pong_ball_dy)

    # --- AI paddle collision (top) ---
    if s.pong_ball_y <= PADDLE_H + 10:
        if s.pong_ai_x <= s.pong_ball_x <= s.pong_ai_x + PADDLE_W:
            s.pong_ball_dy = abs(s.pong_ball_dy)

    # --- Scoring ---
    if s.pong_ball_y >= CANVAS_H:
        s.pong_ai_score += 1
        _reset_ball()
    elif s.pong_ball_y <= 0:
        s.pong_player_score += 1
        _reset_ball()

    # --- Win condition ---
    if s.pong_player_score >= WIN_SCORE:
        s.pong_game_over = True
        s.pong_winner = "Joueur"
    elif s.pong_ai_score >= WIN_SCORE:
        s.pong_game_over = True
        s.pong_winner = "IA"

    # --- AI movement ---
    ai_center = s.pong_ai_x + PADDLE_W / 2
    if ai_center < s.pong_ball_x - 5:
        s.pong_ai_x = min(s.pong_ai_x + AI_SPEED, CANVAS_W - PADDLE_W)
    elif ai_center > s.pong_ball_x + 5:
        s.pong_ai_x = max(s.pong_ai_x - AI_SPEED, 0)


# ---------------------------------------------------------------------------
# Rendering with PIL
# ---------------------------------------------------------------------------


def _render_frame() -> Image.Image:
    """Draw the current game state onto a PIL Image."""
    img = Image.new("RGB", (CANVAS_W, CANVAS_H), BG_COLOR)
    draw = ImageDraw.Draw(img)

    s = st.session_state

    # Centre line (dashed effect)
    for x in range(0, CANVAS_W, 20):
        draw.rectangle([x, CANVAS_H // 2 - 1, x + 10, CANVAS_H // 2 + 1], fill="#333366")

    # AI paddle (top)
    draw.rectangle(
        [int(s.pong_ai_x), 10, int(s.pong_ai_x) + PADDLE_W, 10 + PADDLE_H],
        fill=PINK,
    )

    # Player paddle (bottom)
    draw.rectangle(
        [
            int(s.pong_player_x),
            CANVAS_H - PADDLE_H - 10,
            int(s.pong_player_x) + PADDLE_W,
            CANVAS_H - 10,
        ],
        fill=CYAN,
    )

    # Ball
    bx, by = int(s.pong_ball_x), int(s.pong_ball_y)
    draw.ellipse([bx, by, bx + BALL_SIZE, by + BALL_SIZE], fill=YELLOW)

    # Scores
    try:
        score_font = ImageFont.truetype(
            str(
                __import__("pathlib").Path(__file__).resolve().parent.parent
                / "fonts"
                / "PressStart2P-Regular.ttf"
            ),
            14,
        )
    except (OSError, ImportError):
        score_font = ImageFont.load_default()
    draw.text((20, CANVAS_H // 2 + 10), f"IA: {s.pong_ai_score}", fill=PINK, font=score_font)
    draw.text(
        (20, CANVAS_H // 2 - 25),
        f"Joueur: {s.pong_player_score}",
        fill=CYAN,
        font=score_font,
    )

    return img


def _image_to_bytes(img: Image.Image) -> bytes:
    """Convert PIL Image to PNG bytes for st.image()."""
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Player controls (callbacks)
# ---------------------------------------------------------------------------


def _move_left() -> None:
    """Move player paddle left and advance one tick."""
    st.session_state.pong_player_x = max(0, st.session_state.pong_player_x - PADDLE_SPEED)
    _tick()


def _move_right() -> None:
    """Move player paddle right and advance one tick."""
    st.session_state.pong_player_x = min(
        CANVAS_W - PADDLE_W, st.session_state.pong_player_x + PADDLE_SPEED
    )
    _tick()


def _stay() -> None:
    """Keep paddle in place and advance one tick."""
    _tick()


# ---------------------------------------------------------------------------
# Streamlit page
# ---------------------------------------------------------------------------


def pong_page() -> None:
    """Render the Pong game page in Streamlit."""
    _init_state()

    st.title("PONG")
    st.markdown(
        """
        <style>
        .pong-info {
            font-family: 'Press Start 2P', monospace;
            color: #E0E0E0;
            font-size: 13px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    s = st.session_state

    if not s.pong_running and not s.pong_game_over:
        st.markdown(
            '<p class="pong-info">'
            "Bienvenue dans Pong ! Vous controlez la raquette du bas (cyan).<br>"
            "L'IA controle la raquette du haut (rose).<br>"
            f"Premier a {WIN_SCORE} points gagne.<br>"
            "Utilisez les boutons Gauche / Droite pour bouger."
            "</p>",
            unsafe_allow_html=True,
        )
        if st.button("Jouer", key="pong_start"):
            _reset_game()
            st.rerun()
        return

    if s.pong_game_over:
        st.success(f"Partie terminee ! Gagnant : {s.pong_winner}")
        st.markdown(f"**Score final** -- Joueur: {s.pong_player_score} | IA: {s.pong_ai_score}")
        if st.button("Rejouer", key="pong_restart"):
            _reset_game()
            st.rerun()
        return

    # --- Game in progress ---
    # Render frame
    frame = _render_frame()
    st.image(_image_to_bytes(frame), use_container_width=False, width=CANVAS_W)

    # Controls
    col_left, col_stay, col_right = st.columns(3)
    with col_left:
        st.button("Gauche", key="pong_left", on_click=_move_left)
    with col_stay:
        st.button("Rester", key="pong_stay", on_click=_stay)
    with col_right:
        st.button("Droite", key="pong_right", on_click=_move_right)

    st.caption("Chaque clic fait avancer la balle d'un pas.")
