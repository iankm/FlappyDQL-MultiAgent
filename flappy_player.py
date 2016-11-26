from pygame.constants import K_UP
from pygame_player import PyGamePlayer


class FlappyPlayer(PyGamePlayer):
    def __init__(self, force_game_fps=10, run_real_time=False):
        """
        Example class for playing Flappy Bird
        """
        super(FlappyPlayer, self).__init__(force_game_fps=force_game_fps, run_real_time=run_real_time)
        self.last_score = 0.0

    def get_keys_pressed(self, screen_array, feedback, terminal):
        # TODO: put an actual learning agent here
        return [K_UP]

    def get_feedback(self):
        # import must be done here because otherwise importing would cause the game to start playing
        from games.flappy import score

        # get the difference in score between this and the last run
        score_change = (score - self.last_score)
        self.last_score = score

        return float(score_change), score_change != 0

    def start(self):
        super(FlappyPlayer, self).start()
        import games.flappy


if __name__ == '__main__':
    player = FlappyPlayer()
    player.start()
