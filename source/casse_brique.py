"""Casse-Briques (Breakout) -- Pygame mini-game with difficulty levels and high scores."""

import json
import sys
from pathlib import Path

import pygame

# ---------------------------------------------------------------------------
# Paths (use config-style path resolution)
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
BRICK_WIDTH: int = 91
BRICK_HEIGHT: int = 30
PADDLE_WIDTH: int = 100
PADDLE_HEIGHT: int = 20
BALL_RADIUS: int = 10
WHITE: tuple[int, int, int] = (255, 255, 255)
RED: tuple[int, int, int] = (255, 0, 0)
BLUE: tuple[int, int, int] = (0, 0, 255)
GREEN: tuple[int, int, int] = (0, 255, 0)
YELLOW: tuple[int, int, int] = (255, 255, 0)
CYAN: tuple[int, int, int] = (0, 255, 204)
PINK: tuple[int, int, int] = (255, 110, 199)
BACKGROUND_COLOR: tuple[int, int, int] = (30, 30, 30)

# Difficulty presets: (ball_speed, lives)
DIFFICULTY_SETTINGS: dict[str, dict[str, int]] = {
    "Facile": {"ball_speed": 2, "lives": 3},
    "Moyen": {"ball_speed": 3, "lives": 2},
    "Difficile": {"ball_speed": 5, "lives": 1},
}

# ---------------------------------------------------------------------------
# High-score helpers
# ---------------------------------------------------------------------------


def load_highscores() -> dict[str, int]:
    """Load high scores from JSON file, creating it if missing."""
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
try:
    pygame.init()
except pygame.error as e:
    print(f"Erreur lors de l'initialisation de Pygame : {e}")
    sys.exit()

screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Casse-Briques")

# Load font
try:
    font = pygame.font.Font(_FONT_PATH, 15)
except FileNotFoundError:
    print(f"Police {_FONT_PATH} introuvable, chargement de la police par defaut.")
    font = pygame.font.SysFont(None, 20)


# ---------------------------------------------------------------------------
# Game classes
# ---------------------------------------------------------------------------
class Brick(pygame.sprite.Sprite):
    """A single brick in the breakout grid."""

    def __init__(self, x: int, y: int, color: tuple[int, int, int]) -> None:
        super().__init__()
        self.image = pygame.Surface((BRICK_WIDTH, BRICK_HEIGHT))
        self.image.fill(color)
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y

    def draw(self, surface: pygame.Surface) -> None:
        """Draw brick with white border."""
        surface.blit(self.image, self.rect.topleft)
        pygame.draw.rect(surface, WHITE, self.rect, 2)


class Paddle(pygame.sprite.Sprite):
    """Player-controlled paddle at the bottom of the screen."""

    def __init__(self) -> None:
        super().__init__()
        self.image = pygame.Surface((PADDLE_WIDTH, PADDLE_HEIGHT))
        self.image.fill(BLUE)
        self.rect = self.image.get_rect()
        self.rect.x = (SCREEN_WIDTH - PADDLE_WIDTH) // 2
        self.rect.y = SCREEN_HEIGHT - PADDLE_HEIGHT - 10

    def update(self) -> None:
        """Move paddle left/right with arrow keys."""
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT] and self.rect.x > 0:
            self.rect.x -= 5
        if keys[pygame.K_RIGHT] and self.rect.x < SCREEN_WIDTH - PADDLE_WIDTH:
            self.rect.x += 5


class Ball(pygame.sprite.Sprite):
    """Bouncing ball that destroys bricks on contact."""

    def __init__(self, speed: int = 3) -> None:
        super().__init__()
        self.image = pygame.Surface((BALL_RADIUS * 2, BALL_RADIUS * 2), pygame.SRCALPHA)
        pygame.draw.circle(self.image, YELLOW, (BALL_RADIUS, BALL_RADIUS), BALL_RADIUS)
        self.rect = self.image.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2))
        self.speed_x: int = speed
        self.speed_y: int = -speed

    def update(self) -> None:
        """Move ball, handle wall bounces, paddle/brick collisions, lives, and scoring."""
        global lives, score, high_score

        self.rect.x += self.speed_x
        self.rect.y += self.speed_y

        # Wall bounces
        if self.rect.left <= 0 or self.rect.right >= SCREEN_WIDTH:
            self.speed_x = -self.speed_x
        if self.rect.top <= 0:
            self.speed_y = -self.speed_y

        # Ball fell below screen
        if self.rect.bottom >= SCREEN_HEIGHT:
            lives -= 1
            self.rect.center = (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)
            if lives < 0:
                # Update high score before game over
                if score > high_score:
                    high_score = score
                    scores = load_highscores()
                    scores["casse_brique"] = high_score
                    save_highscores(scores)
                show_game_over()
            else:
                pygame.time.wait(1000)

        # Paddle bounce
        if pygame.sprite.collide_rect(self, paddle):
            self.speed_y = -self.speed_y

        # Brick collisions
        hit_list = pygame.sprite.spritecollide(self, bricks, True)
        if hit_list:
            self.speed_y = -self.speed_y
            score += len(hit_list)
            if len(bricks) == 0:
                # Update high score before win screen
                if score > high_score:
                    high_score = score
                    scores = load_highscores()
                    scores["casse_brique"] = high_score
                    save_highscores(scores)
                show_you_win()


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------


def draw_text(
    surface: pygame.Surface,
    text: str,
    used_font: pygame.font.Font,
    color: tuple[int, int, int],
    rect: pygame.Rect,
) -> None:
    """Render text centred within the given rect."""
    font_surface = used_font.render(text, True, color)
    font_rect = font_surface.get_rect(center=rect.center)
    surface.blit(font_surface, font_rect.topleft)


def draw_hud() -> None:
    """Draw score, lives, and high score on the game screen."""
    hud_font = pygame.font.Font(_FONT_PATH, 12) if Path(_FONT_PATH).exists() else font
    score_text = hud_font.render(f"Score: {score}", True, CYAN)
    lives_text = hud_font.render(f"Vies: {lives + 1}", True, PINK)
    hs_text = hud_font.render(f"Record: {high_score}", True, YELLOW)
    screen.blit(score_text, (10, SCREEN_HEIGHT - 25))
    screen.blit(lives_text, (SCREEN_WIDTH // 2 - 50, SCREEN_HEIGHT - 25))
    screen.blit(hs_text, (SCREEN_WIDTH - 250, SCREEN_HEIGHT - 25))


# ---------------------------------------------------------------------------
# Screen functions
# ---------------------------------------------------------------------------


def show_game_over() -> None:
    """Display the Game Over screen with score and high score."""
    screen.fill(BACKGROUND_COLOR)
    try:
        font_large = pygame.font.Font(_FONT_PATH, 25)
    except FileNotFoundError:
        font_large = pygame.font.SysFont(None, 40)

    draw_text(screen, "Game Over", font_large, RED, screen.get_rect().move(0, -80))

    score_font = pygame.font.Font(_FONT_PATH, 15) if Path(_FONT_PATH).exists() else font
    score_surf = score_font.render(f"Score: {score}  |  Record: {high_score}", True, CYAN)
    score_rect = score_surf.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 20))
    screen.blit(score_surf, score_rect)

    pygame.draw.rect(screen, WHITE, quit_button)
    draw_text(screen, "Quitter", font, BACKGROUND_COLOR, quit_button)

    restart_button.width = 200
    pygame.draw.rect(screen, WHITE, restart_button)
    draw_text(screen, "Recommencer", font, BACKGROUND_COLOR, restart_button)

    pygame.display.flip()
    wait_for_input()


def show_you_win() -> None:
    """Display the victory screen with score and high score."""
    screen.fill(BACKGROUND_COLOR)
    try:
        font_large = pygame.font.Font(_FONT_PATH, 25)
    except FileNotFoundError:
        font_large = pygame.font.SysFont(None, 40)

    draw_text(screen, "Victoire !", font_large, GREEN, screen.get_rect().move(0, -80))

    score_font = pygame.font.Font(_FONT_PATH, 15) if Path(_FONT_PATH).exists() else font
    score_surf = score_font.render(f"Score: {score}  |  Record: {high_score}", True, CYAN)
    score_rect = score_surf.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 20))
    screen.blit(score_surf, score_rect)

    pygame.draw.rect(screen, WHITE, quit_button)
    draw_text(screen, "Quitter", font, BACKGROUND_COLOR, quit_button)

    restart_button.width = 200
    pygame.draw.rect(screen, WHITE, restart_button)
    draw_text(screen, "Recommencer", font, BACKGROUND_COLOR, restart_button)

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
            draw_text(screen, label, font, BACKGROUND_COLOR, rect)

        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                for label, rect in buttons.items():
                    if rect.collidepoint(event.pos):
                        return label


def wait_for_input() -> None:
    """Wait for the user to click Quit or Restart."""
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
                    main()


# ---------------------------------------------------------------------------
# Brick factory
# ---------------------------------------------------------------------------


def create_bricks() -> pygame.sprite.Group:
    """Create a 9x5 grid of coloured bricks."""
    brick_group: pygame.sprite.Group = pygame.sprite.Group()
    colors = [RED, GREEN, BLUE]
    for i in range(9):
        for j in range(5):
            brick = Brick(
                i * (BRICK_WIDTH + 13) + 35,
                j * (BRICK_HEIGHT + 10) + 35,
                colors[j % len(colors)],
            )
            brick_group.add(brick)
    return brick_group


# ---------------------------------------------------------------------------
# Main game loop
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point: difficulty selection then game loop."""
    global bricks, paddle, ball, all_sprites, lives, score, high_score
    global quit_button, restart_button

    # Load high score
    hs = load_highscores()
    high_score = hs.get("casse_brique", 0)

    # Difficulty selection
    difficulty = show_difficulty_selection()
    settings = DIFFICULTY_SETTINGS[difficulty]
    ball_speed = settings["ball_speed"]
    lives = settings["lives"]
    score = 0

    bricks = create_bricks()
    paddle = Paddle()
    ball = Ball(speed=ball_speed)

    all_sprites = pygame.sprite.Group(bricks, paddle, ball)

    quit_button = pygame.Rect(SCREEN_WIDTH // 2 - 100, SCREEN_HEIGHT // 2 + 50, 200, 50)
    restart_button = pygame.Rect(SCREEN_WIDTH // 2 - 100, SCREEN_HEIGHT // 2 + 150, 200, 50)

    clock = pygame.time.Clock()
    game_started = False

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN and not game_started:
                game_started = True

        if game_started:
            all_sprites.update()
            screen.fill(BACKGROUND_COLOR)
            all_sprites.draw(screen)
            draw_hud()
            pygame.display.flip()
            clock.tick(60)


if __name__ == "__main__":
    main()
