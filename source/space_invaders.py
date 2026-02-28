"""Space Invaders -- Pygame mini-game with retro neon theme, difficulty levels, and high scores."""

import json
import random
import sys
from pathlib import Path

import pygame

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_ROOT: Path = Path(__file__).resolve().parent.parent
_FONTS_DIR: Path = _ROOT / "fonts"
_DATA_DIR: Path = _ROOT / "data"
_HIGHSCORES_FILE: Path = _DATA_DIR / "highscores.json"
_FONT_PATH: str = str(_FONTS_DIR / "PressStart2P-Regular.ttf")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SCREEN_WIDTH: int = 800
SCREEN_HEIGHT: int = 600
FPS: int = 60

# Neon colours matching project theme
BG_COLOR: tuple[int, int, int] = (13, 13, 13)
CYAN: tuple[int, int, int] = (0, 255, 204)
PINK: tuple[int, int, int] = (255, 110, 199)
YELLOW: tuple[int, int, int] = (255, 255, 0)
WHITE: tuple[int, int, int] = (255, 255, 255)
RED: tuple[int, int, int] = (255, 0, 0)
GREEN: tuple[int, int, int] = (0, 255, 0)
PURPLE: tuple[int, int, int] = (123, 104, 238)

# Player
PLAYER_WIDTH: int = 50
PLAYER_HEIGHT: int = 30
PLAYER_SPEED: int = 5
PLAYER_COLOR: tuple[int, int, int] = CYAN

# Bullet
BULLET_WIDTH: int = 4
BULLET_HEIGHT: int = 12
BULLET_SPEED: int = 7
BULLET_COLOR: tuple[int, int, int] = YELLOW

# Alien
ALIEN_WIDTH: int = 36
ALIEN_HEIGHT: int = 28
ALIEN_ROWS: int = 4
ALIEN_COLS: int = 8
ALIEN_PADDING: int = 12
ALIEN_COLORS: list[tuple[int, int, int]] = [PINK, PURPLE, CYAN, GREEN]

# Alien bullet
ALIEN_BULLET_WIDTH: int = 4
ALIEN_BULLET_HEIGHT: int = 10
ALIEN_BULLET_SPEED: int = 4
ALIEN_BULLET_COLOR: tuple[int, int, int] = PINK

# Difficulty presets: (alien_speed, alien_shoot_chance_per_frame)
DIFFICULTY_SETTINGS: dict[str, dict[str, float]] = {
    "Facile": {"alien_speed": 1.0, "shoot_chance": 0.002},
    "Moyen": {"alien_speed": 2.0, "shoot_chance": 0.005},
    "Difficile": {"alien_speed": 3.5, "shoot_chance": 0.010},
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
    return {"snake": 0, "casse_brique": 0, "space_invaders": 0}


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
pygame.display.set_caption("Space Invaders")

try:
    font_small = pygame.font.Font(_FONT_PATH, 12)
    font_medium = pygame.font.Font(_FONT_PATH, 18)
    font_large = pygame.font.Font(_FONT_PATH, 28)
except FileNotFoundError:
    font_small = pygame.font.SysFont(None, 18)
    font_medium = pygame.font.SysFont(None, 26)
    font_large = pygame.font.SysFont(None, 40)


# ---------------------------------------------------------------------------
# Game objects
# ---------------------------------------------------------------------------


class Player:
    """The player ship at the bottom of the screen."""

    def __init__(self) -> None:
        self.rect = pygame.Rect(
            SCREEN_WIDTH // 2 - PLAYER_WIDTH // 2,
            SCREEN_HEIGHT - PLAYER_HEIGHT - 20,
            PLAYER_WIDTH,
            PLAYER_HEIGHT,
        )

    def update(self) -> None:
        """Move left/right with arrow keys."""
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT] and self.rect.left > 0:
            self.rect.x -= PLAYER_SPEED
        if keys[pygame.K_RIGHT] and self.rect.right < SCREEN_WIDTH:
            self.rect.x += PLAYER_SPEED

    def draw(self, surface: pygame.Surface) -> None:
        """Draw a simple triangular ship shape."""
        # Main body
        pygame.draw.rect(surface, PLAYER_COLOR, self.rect)
        # Nose (triangle on top)
        nose_points = [
            (self.rect.centerx, self.rect.top - 10),
            (self.rect.left + 5, self.rect.top),
            (self.rect.right - 5, self.rect.top),
        ]
        pygame.draw.polygon(surface, PLAYER_COLOR, nose_points)


class Bullet:
    """A player bullet moving upward."""

    def __init__(self, x: int, y: int) -> None:
        self.rect = pygame.Rect(x - BULLET_WIDTH // 2, y, BULLET_WIDTH, BULLET_HEIGHT)
        self.alive: bool = True

    def update(self) -> None:
        """Move bullet upward; mark dead if off-screen."""
        self.rect.y -= BULLET_SPEED
        if self.rect.bottom < 0:
            self.alive = False

    def draw(self, surface: pygame.Surface) -> None:
        """Render the bullet."""
        pygame.draw.rect(surface, BULLET_COLOR, self.rect)


class AlienBullet:
    """An alien bullet moving downward."""

    def __init__(self, x: int, y: int) -> None:
        self.rect = pygame.Rect(x, y, ALIEN_BULLET_WIDTH, ALIEN_BULLET_HEIGHT)
        self.alive: bool = True

    def update(self) -> None:
        """Move bullet downward; mark dead if off-screen."""
        self.rect.y += ALIEN_BULLET_SPEED
        if self.rect.top > SCREEN_HEIGHT:
            self.alive = False

    def draw(self, surface: pygame.Surface) -> None:
        """Render the alien bullet."""
        pygame.draw.rect(surface, ALIEN_BULLET_COLOR, self.rect)


class Alien:
    """A single alien in the formation."""

    def __init__(self, x: int, y: int, color: tuple[int, int, int], row: int) -> None:
        self.rect = pygame.Rect(x, y, ALIEN_WIDTH, ALIEN_HEIGHT)
        self.color = color
        self.row = row
        self.alive: bool = True

    def draw(self, surface: pygame.Surface) -> None:
        """Draw alien as a coloured rectangle with 'eyes'."""
        if not self.alive:
            return
        pygame.draw.rect(surface, self.color, self.rect)
        # Eyes
        eye_y = self.rect.y + 8
        pygame.draw.circle(surface, BG_COLOR, (self.rect.x + 10, eye_y), 4)
        pygame.draw.circle(surface, BG_COLOR, (self.rect.x + ALIEN_WIDTH - 10, eye_y), 4)


class AlienFormation:
    """Manages the grid of aliens and their lateral + downward movement."""

    def __init__(self, speed: float) -> None:
        self.aliens: list[Alien] = []
        self.direction: float = speed
        self.base_speed: float = speed
        self._create_formation()

    def _create_formation(self) -> None:
        """Build the initial grid of aliens."""
        start_x = (SCREEN_WIDTH - (ALIEN_COLS * (ALIEN_WIDTH + ALIEN_PADDING))) // 2
        start_y = 50
        for row in range(ALIEN_ROWS):
            color = ALIEN_COLORS[row % len(ALIEN_COLORS)]
            for col in range(ALIEN_COLS):
                x = start_x + col * (ALIEN_WIDTH + ALIEN_PADDING)
                y = start_y + row * (ALIEN_HEIGHT + ALIEN_PADDING)
                self.aliens.append(Alien(x, y, color, row))

    def alive_aliens(self) -> list[Alien]:
        """Return list of living aliens."""
        return [a for a in self.aliens if a.alive]

    def update(self) -> None:
        """Move aliens sideways and drop down at screen edges."""
        alive = self.alive_aliens()
        if not alive:
            return

        # Check edges
        move_down = False
        for alien in alive:
            alien.rect.x += int(self.direction)
            if alien.rect.right >= SCREEN_WIDTH or alien.rect.left <= 0:
                move_down = True

        if move_down:
            self.direction = -self.direction
            for alien in alive:
                alien.rect.y += ALIEN_HEIGHT // 2

    def draw(self, surface: pygame.Surface) -> None:
        """Draw all living aliens."""
        for alien in self.aliens:
            alien.draw(surface)

    def random_shooter(self) -> Alien | None:
        """Pick a random living alien to fire."""
        alive = self.alive_aliens()
        if alive:
            return random.choice(alive)
        return None


# ---------------------------------------------------------------------------
# Drawing / UI helpers
# ---------------------------------------------------------------------------


def draw_centred_text(
    surface: pygame.Surface,
    text: str,
    used_font: pygame.font.Font,
    color: tuple[int, int, int],
    y: int,
) -> None:
    """Render text horizontally centred at the given y position."""
    rendered = used_font.render(text, True, color)
    rect = rendered.get_rect(center=(SCREEN_WIDTH // 2, y))
    surface.blit(rendered, rect)


def draw_hud(score: int, high_score: int, lives: int) -> None:
    """Draw score, high score, and lives on screen."""
    score_surf = font_small.render(f"Score: {score}", True, CYAN)
    hs_surf = font_small.render(f"Record: {high_score}", True, YELLOW)
    lives_surf = font_small.render(f"Vies: {lives}", True, PINK)
    screen.blit(score_surf, (10, 10))
    screen.blit(hs_surf, (SCREEN_WIDTH // 2 - 80, 10))
    screen.blit(lives_surf, (SCREEN_WIDTH - 160, 10))


# ---------------------------------------------------------------------------
# Screen functions
# ---------------------------------------------------------------------------


def show_difficulty_selection() -> str:
    """Display a difficulty selection screen. Returns the chosen key."""
    buttons: dict[str, pygame.Rect] = {}
    y_start = SCREEN_HEIGHT // 2 - 60
    for i, label in enumerate(DIFFICULTY_SETTINGS):
        buttons[label] = pygame.Rect(SCREEN_WIDTH // 2 - 120, y_start + i * 70, 240, 50)

    while True:
        screen.fill(BG_COLOR)
        draw_centred_text(screen, "SPACE INVADERS", font_large, CYAN, 80)
        draw_centred_text(screen, "Choisir la difficulte", font_medium, WHITE, y_start - 50)

        for label, rect in buttons.items():
            pygame.draw.rect(screen, PINK, rect, 2)
            txt = font_small.render(label, True, WHITE)
            txt_rect = txt.get_rect(center=rect.center)
            screen.blit(txt, txt_rect)

        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                for label, rect in buttons.items():
                    if rect.collidepoint(event.pos):
                        return label


def show_game_over(score: int, high_score: int) -> None:
    """Display the Game Over screen."""
    quit_btn = pygame.Rect(SCREEN_WIDTH // 2 - 120, SCREEN_HEIGHT // 2 + 60, 240, 50)
    restart_btn = pygame.Rect(SCREEN_WIDTH // 2 - 120, SCREEN_HEIGHT // 2 + 130, 240, 50)

    while True:
        screen.fill(BG_COLOR)
        draw_centred_text(screen, "GAME OVER", font_large, RED, SCREEN_HEIGHT // 2 - 80)
        draw_centred_text(
            screen,
            f"Score: {score}  |  Record: {high_score}",
            font_small,
            CYAN,
            SCREEN_HEIGHT // 2 - 20,
        )

        pygame.draw.rect(screen, WHITE, quit_btn, 2)
        txt_q = font_small.render("Quitter", True, WHITE)
        screen.blit(txt_q, txt_q.get_rect(center=quit_btn.center))

        pygame.draw.rect(screen, WHITE, restart_btn, 2)
        txt_r = font_small.render("Recommencer", True, WHITE)
        screen.blit(txt_r, txt_r.get_rect(center=restart_btn.center))

        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                if quit_btn.collidepoint(event.pos):
                    pygame.quit()
                    sys.exit()
                if restart_btn.collidepoint(event.pos):
                    main()
                    return


def show_you_win(score: int, high_score: int) -> None:
    """Display the victory screen."""
    quit_btn = pygame.Rect(SCREEN_WIDTH // 2 - 120, SCREEN_HEIGHT // 2 + 60, 240, 50)
    restart_btn = pygame.Rect(SCREEN_WIDTH // 2 - 120, SCREEN_HEIGHT // 2 + 130, 240, 50)

    while True:
        screen.fill(BG_COLOR)
        draw_centred_text(screen, "VICTOIRE !", font_large, GREEN, SCREEN_HEIGHT // 2 - 80)
        draw_centred_text(
            screen,
            f"Score: {score}  |  Record: {high_score}",
            font_small,
            CYAN,
            SCREEN_HEIGHT // 2 - 20,
        )

        pygame.draw.rect(screen, WHITE, quit_btn, 2)
        txt_q = font_small.render("Quitter", True, WHITE)
        screen.blit(txt_q, txt_q.get_rect(center=quit_btn.center))

        pygame.draw.rect(screen, WHITE, restart_btn, 2)
        txt_r = font_small.render("Recommencer", True, WHITE)
        screen.blit(txt_r, txt_r.get_rect(center=restart_btn.center))

        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                if quit_btn.collidepoint(event.pos):
                    pygame.quit()
                    sys.exit()
                if restart_btn.collidepoint(event.pos):
                    main()
                    return


# ---------------------------------------------------------------------------
# Main game loop
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point: difficulty selection then game loop."""
    # Load high score
    hs_data = load_highscores()
    high_score: int = hs_data.get("space_invaders", 0)

    # Difficulty
    difficulty = show_difficulty_selection()
    settings = DIFFICULTY_SETTINGS[difficulty]
    alien_speed = settings["alien_speed"]
    shoot_chance = settings["shoot_chance"]

    # Game objects
    player = Player()
    formation = AlienFormation(speed=alien_speed)
    bullets: list[Bullet] = []
    alien_bullets: list[AlienBullet] = []
    score: int = 0
    lives: int = 3
    clock = pygame.time.Clock()
    shoot_cooldown: int = 0

    running = True
    while running:
        clock.tick(FPS)

        # --- Events ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                if shoot_cooldown <= 0:
                    bullets.append(Bullet(player.rect.centerx, player.rect.top))
                    shoot_cooldown = 15  # frames between shots

        # --- Update ---
        if shoot_cooldown > 0:
            shoot_cooldown -= 1

        player.update()
        formation.update()

        # Player bullets
        for bullet in bullets:
            bullet.update()
        bullets = [b for b in bullets if b.alive]

        # Alien bullets
        for ab in alien_bullets:
            ab.update()
        alien_bullets = [ab for ab in alien_bullets if ab.alive]

        # Aliens shoot randomly
        if random.random() < shoot_chance:
            shooter = formation.random_shooter()
            if shooter:
                alien_bullets.append(AlienBullet(shooter.rect.centerx, shooter.rect.bottom))

        # --- Collision: player bullets vs aliens ---
        for bullet in bullets:
            for alien in formation.alive_aliens():
                if bullet.rect.colliderect(alien.rect):
                    bullet.alive = False
                    alien.alive = False
                    # Score based on row (top rows = more points)
                    score += (ALIEN_ROWS - alien.row) * 10
                    break

        # --- Collision: alien bullets vs player ---
        for ab in alien_bullets:
            if ab.rect.colliderect(player.rect):
                ab.alive = False
                lives -= 1
                if lives <= 0:
                    if score > high_score:
                        high_score = score
                        scores = load_highscores()
                        scores["space_invaders"] = high_score
                        save_highscores(scores)
                    show_game_over(score, high_score)
                    return

        # --- Check if aliens reached the bottom ---
        for alien in formation.alive_aliens():
            if alien.rect.bottom >= player.rect.top:
                if score > high_score:
                    high_score = score
                    scores = load_highscores()
                    scores["space_invaders"] = high_score
                    save_highscores(scores)
                show_game_over(score, high_score)
                return

        # --- All aliens destroyed = win ---
        if not formation.alive_aliens():
            if score > high_score:
                high_score = score
                scores = load_highscores()
                scores["space_invaders"] = high_score
                save_highscores(scores)
            show_you_win(score, high_score)
            return

        # --- Draw ---
        screen.fill(BG_COLOR)
        draw_hud(score, high_score, lives)
        player.draw(screen)
        formation.draw(screen)
        for bullet in bullets:
            bullet.draw(screen)
        for ab in alien_bullets:
            ab.draw(screen)

        # Decorative bottom line
        pygame.draw.line(
            screen, PURPLE, (0, SCREEN_HEIGHT - 5), (SCREEN_WIDTH, SCREEN_HEIGHT - 5), 2
        )

        pygame.display.flip()


if __name__ == "__main__":
    main()
