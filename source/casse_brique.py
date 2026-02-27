import sys

import pygame

# Initialisation de Pygame avec gestion d'erreur
try:
    pygame.init()
except pygame.error as e:
    print(f"Erreur lors de l'initialisation de Pygame : {e}")
    sys.exit()

# Définition des constantes
SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 600
BRICK_WIDTH = 91
BRICK_HEIGHT = 30
PADDLE_WIDTH = 100
PADDLE_HEIGHT = 20
BALL_RADIUS = 10
WHITE = (255, 255, 255)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
YELLOW = (255, 255, 0)
BACKGROUND_COLOR = (30, 30, 30)
LIVES = 1

# Initialisation de l'écran
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Casse-Briques")

# Charger la police
font_path = "PressStart2P-Regular.ttf"
try:
    font = pygame.font.Font(font_path, 15)
except FileNotFoundError:
    print(f"Police {font_path} introuvable, chargement de la police par défaut.")
    font = pygame.font.SysFont(None, 20)  # Police de secours


# Classe pour la brique
class Brick(pygame.sprite.Sprite):
    def __init__(self, x, y, color):
        super().__init__()
        self.image = pygame.Surface((BRICK_WIDTH, BRICK_HEIGHT))
        self.image.fill(color)
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y

    def draw(self, surface):
        surface.blit(self.image, self.rect.topleft)
        pygame.draw.rect(surface, WHITE, self.rect, 2)


# Classe pour la raquette
class Paddle(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        self.image = pygame.Surface((PADDLE_WIDTH, PADDLE_HEIGHT))
        self.image.fill(BLUE)
        self.rect = self.image.get_rect()
        self.rect.x = (SCREEN_WIDTH - PADDLE_WIDTH) // 2
        self.rect.y = SCREEN_HEIGHT - PADDLE_HEIGHT - 10

    def update(self):
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT] and self.rect.x > 0:
            self.rect.x -= 5
        if keys[pygame.K_RIGHT] and self.rect.x < SCREEN_WIDTH - PADDLE_WIDTH:
            self.rect.x += 5


# Classe pour la balle
class Ball(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        self.image = pygame.Surface((BALL_RADIUS * 2, BALL_RADIUS * 2), pygame.SRCALPHA)
        pygame.draw.circle(self.image, YELLOW, (BALL_RADIUS, BALL_RADIUS), BALL_RADIUS)
        self.rect = self.image.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2))
        self.speed_x = 3
        self.speed_y = -3

    def update(self):
        self.rect.x += self.speed_x
        self.rect.y += self.speed_y

        if self.rect.left <= 0 or self.rect.right >= SCREEN_WIDTH:
            self.speed_x = -self.speed_x
        if self.rect.top <= 0:
            self.speed_y = -self.speed_y
        if self.rect.bottom >= SCREEN_HEIGHT:
            global lives
            lives -= 1
            self.rect.center = (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)
            if lives < 0:
                show_game_over()
            else:
                pygame.time.wait(1000)

        if pygame.sprite.collide_rect(self, paddle):
            self.speed_y = -self.speed_y

        hit_list = pygame.sprite.spritecollide(self, bricks, True)
        if hit_list:
            self.speed_y = -self.speed_y
            if len(bricks) == 0:
                show_you_win()


# Fonction pour afficher du texte
def draw_text(surface, text, font, color, rect):
    font_surface = font.render(text, True, color)
    font_rect = font_surface.get_rect(center=rect.center)
    surface.blit(font_surface, font_rect.topleft)


# Fonction pour afficher Game Over
def show_game_over():
    screen.fill(BACKGROUND_COLOR)
    font_large = pygame.font.Font(font_path, 25)
    draw_text(screen, "Game Over", font_large, RED, screen.get_rect().move(0, -50))

    pygame.draw.rect(screen, WHITE, quit_button)
    draw_text(screen, "Quitter", font, BACKGROUND_COLOR, quit_button)

    restart_button.width = 200
    pygame.draw.rect(screen, WHITE, restart_button)
    draw_text(screen, "Recommencer", font, BACKGROUND_COLOR, restart_button)

    pygame.display.flip()
    wait_for_input()


# Fonction pour attendre une action de l'utilisateur
def wait_for_input():
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


# Création des briques
def create_bricks():
    bricks = pygame.sprite.Group()
    colors = [RED, GREEN, BLUE]
    for i in range(9):
        for j in range(5):
            brick = Brick(
                i * (BRICK_WIDTH + 13) + 35, j * (BRICK_HEIGHT + 10) + 35, colors[j % len(colors)]
            )
            bricks.add(brick)
    return bricks


# Fonction principale du jeu
def main():
    global bricks, paddle, ball, all_sprites, lives
    lives = LIVES

    bricks = create_bricks()
    paddle = Paddle()
    ball = Ball()

    all_sprites = pygame.sprite.Group(bricks, paddle, ball)

    global quit_button, restart_button
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
            pygame.display.flip()
            clock.tick(60)


if __name__ == "__main__":
    main()
