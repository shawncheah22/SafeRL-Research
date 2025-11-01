import safety_gymnasium

class Environment:
    def __init__(self, env_id='SafetyPointGoal1-v0', render_mode='human'):
        self.env_id = env_id
        self.render_mode = render_mode
        self.env = safety_gymnasium.make(self.env_id, render_mode=self.render_mode)
        self.collision_count = 0
        self.in_collision_state = False # Tracks if we are currently in a collision

    def step(self, action):
        assert self.env.action_space.contains(action)
        obs, reward, cost, terminated, truncated, info = self.env.step(action)

        is_colliding = cost > 0
        if is_colliding and not self.in_collision_state:
            # A new collision event has started
            self.collision_count += 1
        
        self.in_collision_state = is_colliding

        return obs, reward, cost, terminated, truncated, info
    
    def action_space(self):
        return self.env.action_space

    def render(self):
        return self.env.render()

    def reset(self):
        obs, info = self.env.reset()
        self.collision_count = 0
        self.in_collision_state = False
        return obs, info
