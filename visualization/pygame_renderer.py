class PygameRenderer:
    COLOR_MAP = {
        0: (40, 40, 40),    # wall — dark grey
        1: (255, 255, 255), # passage — white
        2: (0, 120, 255),   # agent — bright blue
        3: (220, 50, 50),   # goal — red
    }

    @staticmethod
    def draw(surface, obs, size: int, cell_px: int = 30) -> None:
        import pygame
        grid = obs.reshape((2 * size + 1, 2 * size + 1))
        surface.fill((0, 0, 0))
        for row in range(2 * size + 1):
            for col in range(2 * size + 1):
                color = PygameRenderer.COLOR_MAP.get(int(grid[row, col]), (0, 0, 0))
                rect = pygame.Rect(col * cell_px, row * cell_px, cell_px, cell_px)
                pygame.draw.rect(surface, color, rect)

    @staticmethod
    def get_window_size(size: int, cell_px: int = 30) -> tuple:
        dim = (2 * size + 1) * cell_px
        return (dim, dim)
