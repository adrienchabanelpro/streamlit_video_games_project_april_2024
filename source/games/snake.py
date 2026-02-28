"""Snake -- Pygame mini-game with difficulty levels and high-score persistence."""

import json
import random
import sys
from pathlib import Path

import pygame

# ---------------------------------------------------------------------------
# Paths (config-style resolution)
# ---------------------------------------------------------------------------
_ROOT: Path = Path(__file__).resolve().parent.parent
_FONTS_DIR: Path = _ROOT / "fonts"
_DATA_DIR: Path = _ROOT / "data"
_HIGHSCORES_FILE: Path = _DATA_DIR / "highscores.json"
_FONT_PATH: str = str(_FONTS_DIR / "PressStart2P-Regular.ttf")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SCREEN_WIDTH: int = 1000
SCREEN_HEIGHT: int = 600
CELL_SIZE: int = 20
WHITE: tuple[int, int, int] = (255, 255, 255)
RED: tuple[int, int, int] = (255, 0, 0)
GREEN: tuple[int, int, int] = (0, 255, 0)
BLACK: tuple[int, int, int] = (0, 0, 0)
CYAN: tuple[int, int, int] = (0, 255, 204)
PINK: tuple[int, int, int] = (255, 110, 199)
YELLOW: tuple[int, int, int] = (255, 255, 0)
BACKGROUND_COLOR: tuple[int, int, int] = (30, 30, 30)

# Difficulty presets: base tick speed
DIFFICULTY_SETTINGS: dict[str, int] = {
    "Facile": 3,
    "Moyen": 5,
    "Difficile": 8,
}

# ---------------------------------------------------------------------------
# High-score helpers
# ---------------------------------------------------------------------------


def load_highscores() -> dict[str, int]:
    """Load high scores from JSON file, creating defaults if missing."""
    if _HIGHSCORES_FILE.exists():
        try:
            with open(_HIGHSCORES_FILE) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    return {"snake": 0, "casse_brique": 0}


def save_highscores(scores: dict[str, int]) -> None:
    """Persist high scores to JSON file."""
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(_HIGHSCORES_FILE, "w") as f:
        json.dump(scores, f, indent=2)


# ---------------------------------------------------------------------------
# Initialise Pygame
# ---------------------------------------------------------------------------
pygame.init()

screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Snake Game")

try:
    font = pygame.font.Font(_FONT_PATH, 25)
except FileNotFoundError:
    print(f"Police {_FONT_PATH} introuvable.")
    sys.exit()


# ---------------------------------------------------------------------------
# Game classes
# ---------------------------------------------------------------------------


class Apple:
    """Randomly-placed food item for the snake."""

    def __init__(self) -> None:
        self.position: tuple[int, int] = (0, 0)
        self.randomize_position()

    def randomize_position(self) -> None:
        """Move apple to a new random grid cell."""
        self.position = (
            random.randint(0, (SCREEN_WIDTH // CELL_SIZE) - 1) * CELL_SIZE,
            random.randint(0, (SCREEN_HEIGHT // CELL_SIZE) - 1) * CELL_SIZE,
        )

    def draw(self, surface: pygame.Surface) -> None:
        """Render the apple as a red square."""
        pygame.draw.rect(
            surface,
            RED,
            pygame.Rect(self.position[0], self.position[1], CELL_SIZE, CELL_SIZE),
        )


class Snake:
    """The player-controlled snake."""

    def __init__(self) -> None:
        self.positions: list[tuple[int, int]] = [(100, 100)]
        self.direction: tuple[int, int] = (0, 0)
        self.grow: bool = False

    def set_direction(self, direction: tuple[int, int]) -> None:
        """Set movement direction (prevents 180-degree reversal)."""
        if (direction[0] * -1, direction[1] * -1) != self.direction:
            self.direction = direction

    def move(self) -> None:
        """Advance the snake by one cell in the current direction."""
        head_x, head_y = self.positions[0]
        new_head_x = (head_x + self.direction[0] * CELL_SIZE) % SCREEN_WIDTH
        new_head_y = (head_y + self.direction[1] * CELL_SIZE) % SCREEN_HEIGHT
        new_head = (new_head_x, new_head_y)

        if self.grow:
            self.positions.insert(0, new_head)
            self.grow = False
        else:
            self.positions.insert(0, new_head)
            self.positions.pop()

    def grow_snake(self) -> None:
        """Mark the snake to grow by one segment on next move."""
        self.grow = True

    def draw(self, surface: pygame.Surface) -> None:
        """Render each segment of the snake."""
        for position in self.positions:
            pygame.draw.rect(
                surface,
                GREEN,
                pygame.Rect(position[0], position[1], CELL_SIZE, CELL_SIZE),
            )

    def collides_with_self(self) -> bool:
        """Check if the head overlaps any body segment."""
        return self.positions[0] in self.positions[1:]


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------


def draw_text(
    surface: pygame.Surface,
    text: str,
    used_font: pygame.font.Font,
    color: tuple[int, int, int],
    position: tuple[int, int],
) -> None:
    """Render text at the given pixel position."""
    font_surface = used_font.render(text, True, color)
    surface.blit(font_surface, position)


def draw_hud(current_score: int, high_score_val: int) -> None:
    """Draw score and high score on screen."""
    try:
        hud_font = pygame.font.Font(_FONT_PATH, 12)
    except FileNotFoundError:
        hud_font = font
    score_text = hud_font.render(f"Score: {current_score}", True, CYAN)
    hs_text = hud_font.render(f"Record: {high_score_val}", True, YELLOW)
    screen.blit(score_text, (10, 10))
    screen.blit(hs_text, (SCREEN_WIDTH - 250, 10))


# ---------------------------------------------------------------------------
# Screen functions
# ---------------------------------------------------------------------------


def show_game_over(current_score: int, high_score_val: int) -> None:
    """Display the Game Over screen with score information."""
    screen.fill(BACKGROUND_COLOR)
    go_font = pygame.font.Font(_FONT_PATH, 25)
    draw_text(screen, "Game Over", go_font, RED, (SCREEN_WIDTH // 2 - 80, SCREEN_HEIGHT // 2 - 80))

    try:
        info_font = pygame.font.Font(_FONT_PATH, 15)
    except FileNotFoundError:
        info_font = font
    info_surf = info_font.render(f"Score: {current_score}  |  Record: {high_score_val}", True, CYAN)
    info_rect = info_surf.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 20))
    screen.blit(info_surf, info_rect)

    pygame.draw.rect(screen, WHITE, quit_button)
    quit_text = font.render("Quitter", True, BLACK)
    quit_text_x = quit_button.x + (quit_button.width - quit_text.get_width()) // 2
    draw_text(screen, "Quitter", font, BLACK, (quit_text_x, quit_button.y + 10))

    pygame.draw.rect(screen, WHITE, restart_button)
    restart_text = font.render("Recommencer", True, BLACK)
    restart_text_x = restart_button.x + (restart_button.width - restart_text.get_width()) // 2
    draw_text(screen, "Recommencer", font, BLACK, (restart_text_x, restart_button.y + 10))

    pygame.display.flip()
    wait_for_input()


def show_difficulty_selection() -> str:
    """Display a difficulty selection screen. Returns the chosen difficulty key."""
    buttons: dict[str, pygame.Rect] = {}
    y_start = SCREEN_HEIGHT // 2 - 60
    for i, label in enumerate(DIFFICULTY_SETTINGS):
        buttons[label] = pygame.Rect(SCREEN_WIDTH // 2 - 120, y_start + i * 70, 240, 50)

    while True:
        screen.fill(BACKGROUND_COLOR)
        try:
            title_font = pygame.font.Font(_FONT_PATH, 20)
        except FileNotFoundError:
            title_font = pygame.font.SysFont(None, 30)

        title_surf = title_font.render("Choisir la difficulte", True, CYAN)
        title_rect = title_surf.get_rect(center=(SCREEN_WIDTH // 2, y_start - 60))
        screen.blit(title_surf, title_rect)

        for label, rect in buttons.items():
            pygame.draw.rect(screen, WHITE, rect)
            draw_text(
                screen,
                label,
                font,
                BACKGROUND_COLOR,
                (
                    rect.x + (rect.width - font.size(label)[0]) // 2,
                    rect.y + (rect.height - font.size(label)[1]) // 2,
                ),
            )

        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                for label, rect in buttons.items():
                    if rect.collidepoint(event.pos):
                        return label


def show_instructions() -> None:
    """Display game instructions with a Start button."""
    screen.fill(BACKGROUND_COLOR)
    instructions = [
        "Instructions du jeu Snake:",
        "Utilisez les fleches pour deplacer le serpent.",
        "Mangez les pommes rouges pour grandir.",
        "Le jeu se termine si le serpent se mord.",
        "Appuyez sur Commencer pour debuter le jeu.",
    ]
    try:
        instr_font = pygame.font.Font(_FONT_PATH, 15)
    except FileNotFoundError:
        instr_font = pygame.font.SysFont(None, 20)

    for i, line in enumerate(instructions):
        draw_text(screen, line, instr_font, WHITE, (50, 20 + i * 40))

    pygame.draw.rect(screen, WHITE, start_button)
    start_text = font.render("Commencer", True, BLACK)
    start_text_x = start_button.x + (start_button.width - start_text.get_width()) // 2
    draw_text(screen, "Commencer", font, BLACK, (start_text_x, start_button.y + 10))

    pygame.display.flip()


def wait_for_input() -> None:
    """Wait for the user to click Quit, Restart, or Start."""
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                if quit_button.collidepoint(event.pos):
                    pygame.quit()
                    sys.exit()
                if restart_button.collidepoint(event.pos):
                    return
                if start_button.collidepoint(event.pos):
                    return


# ---------------------------------------------------------------------------
# Main game loop
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point: difficulty selection, instructions, then game loop."""
    global quit_button, restart_button, start_button

    # Load high score
    hs = load_highscores()
    high_score = hs.get("snake", 0)

    # Buttons
    quit_button = pygame.Rect(SCREEN_WIDTH // 2 - 120, SCREEN_HEIGHT // 2 + 50, 240, 50)
    restart_button = pygame.Rect(SCREEN_WIDTH // 2 - 150, SCREEN_HEIGHT // 2 + 150, 300, 50)
    start_button = pygame.Rect(SCREEN_WIDTH // 2 - 120, SCREEN_HEIGHT // 2 + 100, 240, 50)

    # Difficulty selection
    difficulty = show_difficulty_selection()
    base_speed = DIFFICULTY_SETTINGS[difficulty]

    snake = Snake()
    apple = Apple()
    clock = pygame.time.Clock()
    score = 0

    show_instructions()
    wait_for_input()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    snake.set_direction((0, -1))
                elif event.key == pygame.K_DOWN:
                    snake.set_direction((0, 1))
                elif event.key == pygame.K_LEFT:
                    snake.set_direction((-1, 0))
                elif event.key == pygame.K_RIGHT:
                    snake.set_direction((1, 0))

        snake.move()

        if snake.positions[0] == apple.position:
            snake.grow_snake()
            apple.randomize_position()
            score += 1

        if snake.collides_with_self():
            # Update high score if new record
            if score > high_score:
                high_score = score
                scores = load_highscores()
                scores["snake"] = high_score
                save_highscores(scores)
            show_game_over(score, high_score)
            break

        screen.fill(BACKGROUND_COLOR)
        snake.draw(screen)
        apple.draw(screen)
        draw_hud(score, high_score)

        pygame.display.flip()

        speed = min(30, base_speed + len(snake.positions) // 2)
        clock.tick(speed)


if __name__ == "__main__":
    main()
