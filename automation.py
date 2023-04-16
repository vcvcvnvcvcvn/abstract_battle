import random
import time
import numpy as np


class automation():
    def __init__(self, max_hp = 5.,print_info = True):
        self.max_hp = max_hp
        self.hp = self.max_hp
        self.att = 1.
        self.shield_rate = 1.
        self.print_info = print_info
    def attack(self):
        damage = self.att
        self.att = 1.
        self.shield_rate = 1.
        return ['attack',damage]
    def charge(self):
        self.att +=1.
        self.shield_rate = 1.
        if self.print_info:
            print('charge:',self.att)
        return ['charge']
    def shield(self):
        dice = random.uniform(0,1)
        success_rate = self.shield_rate
        self.shield_rate = self.shield_rate*0.5
        if dice < success_rate:
            return ['shield']
        else:
            if self.print_info:
                print('SHIELD FAIL')
            self.shield_rate = 1.
            return ['Fail']
    def random_action(self):
        action = random.choice([0,1,2])
        if action == 0:
            return self.attack()
        elif action == 1:
            return self.shield()
        elif action == 2:
            return self.charge()
    def cmd(self,action):
        if action == 0:
            return self.attack()
        elif action == 1:
            return self.shield()
        elif action == 2:
            return self.charge()
    

def battle(A:automation,B:automation,A_action,B_action,print_info = True):
    if B_action[0] == 'attack':
        if A_action[0]=='shield':
            A.hp = min(A.hp+B_action[1],A.max_hp)
        else:
            A.hp = max(A.hp-B_action[1],0)
    if A_action[0] == 'attack':
        if B_action[0]=='shield':
            B.hp = min(B.hp+A_action[1],B.max_hp)
        else:
            B.hp = max(B.hp-A_action[1],0)
    if print_info:
        print('HP of A:{}, HP of B:{}'.format(A.hp,B.hp))

def random_battle(n_round = 100):
    A = automation()
    B = automation()
    for i in range(n_round):
        if A.hp==0 and B.hp==0:
            print('TIE')
            break
        elif A.hp ==0:
            print('Win: B')
            break
        elif B.hp ==0:
            print('Win: A')
            break
        time.sleep(0.3)
        A_action = A.random_action()
        B_action = B.random_action()
        print('A:{}, B:{}'.format(A_action,B_action))
        battle(A,B,A_action,B_action)

def pve_battle(n_round = 100):
    A = automation()
    B = automation()
    for i in range(n_round):
        if A.hp==0 and B.hp==0:
            print('TIE')
            break
        elif A.hp ==0:
            print('Win: B')
            break
        elif B.hp ==0:
            print('Win: A')
            break
        time.sleep(0.3)
        action = input()
        A_action = A.cmd(int(action))
        B_action = B.random_action()
        print('A:{}, B:{}'.format(A_action,B_action))
        battle(A,B,A_action,B_action)

class random_battle_env():
    def __init__(self) -> None:
        self.enemy = automation(max_hp=5,print_info = False)
        self.player = automation(print_info = False)
        self.end = False
        self.reward = 0
        self.result = -1
    def reset(self):
        self.enemy = automation(max_hp=5,print_info = False)
        self.player = automation(print_info = False)
        self.end = False
        self.reward = 0
        self.result = -1
    
    def get_status(self):
        return np.array([self.enemy.hp,self.enemy.att,self.enemy.shield_rate,self.player.hp,self.player.att,self.player.shield_rate]).astype('float32')
    def step(self,action):
        ###one step
        prev_player_hp = self.player.hp
        prev_enemy_hp = self.enemy.hp
        # if action == 0:
        #     player_action = self.player.attack()
        # elif action == 1:
        #     player_action = self.player.shield()
        # elif action == 2:
        #     player_action = self.player.charge()
        enemy_action = self.enemy.random_action()
        battle(self.player,self.enemy,self.player.cmd(action),enemy_action,print_info = False)
        self.reward = 0#self.player.hp+prev_enemy_hp-prev_player_hp-self.enemy.hp#prev_enemy_hp-self.enemy.hp#
        if self.player.hp==0 and self.enemy.hp==0:
            self.reward+=0
            self.end = True
            self.result = 0.5
        elif self.enemy.hp ==0:
            self.end = True
            self.reward+=10
            self.result = 1
        elif self.player.hp ==0:
            self.end = True
            self.reward-=10
            self.result = 0



class general_battle_env():
    def __init__(self,max_hp = 5,print_info = False) -> None:
        self.enemy = automation(max_hp=max_hp,print_info = print_info)
        self.player = automation(max_hp=max_hp,print_info = print_info)
        self.end = False
        self.reward = [0.,0.]
        self.result = -1
        self.history = []
        self.print_info = print_info
    def reset(self):
        self.enemy = automation(max_hp=5,print_info = False)
        self.player = automation(print_info = False)
        self.end = False
        self.reward = [0.,0.]
        self.result = -1
        self.history = []

    def get_status(self):
        return np.array([self.enemy.hp,self.enemy.att,self.enemy.shield_rate,self.player.hp,self.player.att,self.player.shield_rate]),np.array([self.player.hp,self.player.att,self.player.shield_rate,self.enemy.hp,self.enemy.att,self.enemy.shield_rate])
    def step(self,action_a,action_b):
        self.history.append([self.get_status()[0],action_a,action_b])
        ###one step
        prev_player_hp = self.player.hp
        prev_enemy_hp = self.enemy.hp
        # if action == 0:
        #     player_action = self.player.attack()
        # elif action == 1:
        #     player_action = self.player.shield()
        # elif action == 2:
        #     player_action = self.player.charge()
        battle(self.player,self.enemy,self.player.cmd(action_a),self.enemy.cmd(action_b),print_info=self.print_info)
        #self.reward = [0.,0.]#self.player.hp+prev_enemy_hp-prev_player_hp-self.enemy.hp#prev_enemy_hp-self.enemy.hp#
        if self.player.hp==0 and self.enemy.hp==0:
            self.reward = [0.,0.]
            self.end = True
            self.result = 0.5
        elif self.enemy.hp ==0:
            self.end = True
            self.reward = [1.,-1.]
            self.result = 1
        elif self.player.hp ==0:
            self.end = True
            self.reward = [-1.,1.]
            self.result = 0
