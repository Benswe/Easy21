import random


class easy21Env:
    def __init__(self):
        self.actions = ["hit", "stick"]
        self.dealer_card = 0
        self.player_sum = 0
        self.done = False
        self.terminal_state = False

    def reset(self): # resets env for new round
        self.dealer_card = self.draw_black_card()
        self.player_sum = self.draw_black_card()
        self.done = False
        return (self.dealer_card, self.player_sum) # return the initial state
    
    def draw_black_card(self):
        return random.randint(1, 10)
    
    def draw_card(self):
        num = random.random()
        if num > 1/3: # 2/3 chance (black)
            card = self.draw_black_card()
        else:
            card = self.draw_black_card() * -1 # will subtract from the total if red
        return card




    def step(self, action): # should return next_state, reward, done
        if action == "hit":
            self.player_sum += self.draw_card()
            if self.player_sum > 21 or self.player_sum < 1:
                reward = -1
                return (self.dealer_card, self.player_sum), reward, True
            else:
                reward = 0
                return (self.dealer_card, self.player_sum), reward, False

                
        if action == "stick":
            while 1 <= self.dealer_card < 17:
                self.dealer_card += self.draw_card()
            if self.dealer_card > 21 or self.dealer_card < 1: # if dealer busts
                reward = 1
                self.done = True
                return (self.dealer_card, self.player_sum), reward, self.done
            elif self.dealer_card > self.player_sum: #  dealer wins 
                reward = -1
                self.done = True
                return (self.dealer_card, self.player_sum), reward, self.done
            elif self.dealer_card == self.player_sum: # tie case
                reward = 0
                self.done = True
                return (self.dealer_card, self.player_sum), reward, self.done
            else: 
                reward = 1
                self.done = True
                return (self.dealer_card, self.player_sum), reward, self.done
        else:
            raise ValueError(f"Invalid action: {action}")

if __name__ == "__main__":
    env = easy21Env()

    state = env.reset() # (dealer_sum ,player_sum)

    while True:
        print(state)
        user_action = input("Take an action: ")
        next_state, reward, done = env.step(user_action)
        print(next_state, reward)
        if done:
            break
        state = next_state


                
        
                

        
        

