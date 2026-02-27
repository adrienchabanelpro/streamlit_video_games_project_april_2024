import os
import random
import sys

import pygame

# Initialisation de Pygame
pygame.init()

# Définition des constantes
SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 600
CELL_SIZE = 20
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLACK = (0, 0, 0)
BACKGROUND_COLOR = (30, 30, 30)

# Initialisation de l'écran
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Snake Game")

# Charger la police avec un chemin correct
font_path = os.path.join(os.path.dirname(__file__), "..", "fonts", "PressStart2P-Regular.ttf")

try:
    font = pygame.font.Font(font_path, 25)
except FileNotFoundError:
    print(f"Police {font_path} introuvable.")
    sys.exit()


# Classe pour la pomme
class Apple:
    def __init__(self):
        self.position = (0, 0)
        self.randomize_position()

    def randomize_position(self):
        self.position = (
            random.randint(0, (SCREEN_WIDTH // CELL_SIZE) - 1) * CELL_SIZE,
            random.randint(0, (SCREEN_HEIGHT // CELL_SIZE) - 1) * CELL_SIZE,
        )

    def draw(self, surface):
        pygame.draw.rect(
            surface, RED, pygame.Rect(self.position[0], self.position[1], CELL_SIZE, CELL_SIZE)
        )


# Classe pour le serpent
class Snake:
    def __init__(self):
        self.positions = [(100, 100)]
        self.direction = (0, 0)
        self.grow = False

    def set_direction(self, direction):
        if (direction[0] * -1, direction[1] * -1) != self.direction:
            self.direction = direction

    def move(self):
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

    def grow_snake(self):
        self.grow = True

    def draw(self, surface):
        for position in self.positions:
            pygame.draw.rect(
                surface, GREEN, pygame.Rect(position[0], position[1], CELL_SIZE, CELL_SIZE)
            )

    def collides_with_self(self):
        return self.positions[0] in self.positions[1:]


# Fonction pour afficher du texte
def draw_text(surface, text, font, color, position):
    font_surface = font.render(text, True, color)
    surface.blit(font_surface, position)


# Fonction pour afficher la page de Game Over
def show_game_over():
    screen.fill(BACKGROUND_COLOR)
    font = pygame.font.Font(font_path, 25)
    draw_text(screen, "Game Over", font, RED, (SCREEN_WIDTH // 2 - 80, SCREEN_HEIGHT // 2 - 50))

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


# Fonction pour afficher les instructions
def show_instructions():
    screen.fill(BACKGROUND_COLOR)
    instructions = [
        "Instructions du jeu Snake:",
        "Utilisez les flèches directionnelles pour déplacer le serpent.",
        "Le but est de manger les pommes rouges pour grandir.",
        "Le jeu se termine si le serpent se mord lui-même.",
        "Appuyez sur le bouton Commencer pour débuter le jeu.",
    ]
    for i, line in enumerate(instructions):
        draw_text(screen, line, pygame.font.Font(font_path, 15), WHITE, (50, 20 + i * 40))

    pygame.draw.rect(screen, WHITE, start_button)
    start_text = font.render("Commencer", True, BLACK)
    start_text_x = start_button.x + (start_button.width - start_text.get_width()) // 2
    draw_text(screen, "Commencer", font, BLACK, (start_text_x, start_button.y + 10))

    pygame.display.flip()


# Fonction pour attendre l'entrée de l'utilisateur
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
                    return  # Quitte proprement pour relancer le jeu
                if start_button.collidepoint(event.pos):
                    return  # Démarrer le jeu


# Fonction principale
def main():
    snake = Snake()
    apple = Apple()
    clock = pygame.time.Clock()
    score = 0

    # Définition des boutons
    global quit_button, restart_button, start_button
    quit_button = pygame.Rect(
        SCREEN_WIDTH // 2 - 120, SCREEN_HEIGHT // 2 + 50, 240, 50
    )  # Bouton plus large
    restart_button = pygame.Rect(
        SCREEN_WIDTH // 2 - 150, SCREEN_HEIGHT // 2 + 150, 300, 50
    )  # Plus large pour "Recommencer"
    start_button = pygame.Rect(SCREEN_WIDTH // 2 - 120, SCREEN_HEIGHT // 2 + 100, 240, 50)

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
            show_game_over()
            break

        screen.fill(BACKGROUND_COLOR)
        snake.draw(screen)
        apple.draw(screen)
        draw_text(screen, f"Score: {score}", font, WHITE, (100, 30))

        pygame.display.flip()

        speed = min(30, 5 + len(snake.positions) // 2)
        clock.tick(speed)


if __name__ == "__main__":
    main()
