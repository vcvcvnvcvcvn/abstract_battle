import random
from flask import Flask, render_template, request, redirect,url_for
import time

app = Flask(__name__)

class Player:
    def __init__(self):
        self.hp = 5
        self.attack = 1
        self.defense_success = 1

class Computer:
    def __init__(self):
        self.hp = 5
        self.attack = 1
        self.defense_success = 1

player = Player()
computer = Computer()
message = ''
n = 0

def message_update(A_action,B_action):
    message_dict = {
        'attack':'进行了攻击',
        'shield':'进行了防守',
        'charge':'进行了充能',
        'FAIL':'进行了防守，但是失败了'
    }
    player_message = '你刚才'+message_dict[A_action[0]]
    computer_message = '电脑'+message_dict[B_action[0]]
    #tail_message = '双方剩余hp为{}, {}\n'.format(player.hp,computer.hp)
    return player_message+'; '+computer_message#+'; '+tail_message



def battle(A_action,B_action,print_info = False):

    if B_action[0] == 'attack':
        if A_action[0]=='shield':
            player.hp = min(player.hp+B_action[1],5)
        else:
            player.hp = max(player.hp-B_action[1],0)
    if A_action[0] == 'attack':
        if B_action[0]=='shield':
            computer.hp = min(computer.hp+A_action[1],5)
        else:
            computer.hp = max(computer.hp-A_action[1],0)
    if print_info:
        print('HP of A:{}, HP of B:{}'.format(player.hp,computer.hp))

def computer_beh(policy = None):
    if policy is None:
        #time.sleep(2)
        action = random.choice([['attack'],['shield'],['charge']])
    else:
        pass
    if action[0] == 'shield':
        dice = random.uniform(0,1)
        if dice > computer.defense_success:
            action = ['FAIL']
            computer.defense_success = 1
        else:
            computer.defense_success*=0.5
    if action[0] == 'charge':
        computer.defense_success = 1
        computer.attack+=1
    if action[0] == 'attack':
        computer.defense_success = 1
        action = ['attack',computer.attack]
        computer.attack = 1
    return action

@app.route('/', methods=['GET', 'POST'])
def index():
    global message, n
    if request.method == 'POST':
        return render_template('index.html', player=player, computer=computer)
    else:
        player.hp = 5
        player.attack = 1
        player.defense_success = 1
        computer.hp = 5
        computer.attack = 1
        computer.defense_success = 1
        message = ''
        n = 0
    return render_template('index.html', player=player, computer=computer, message=message)



@app.route('/attack', methods=['POST'])
def attack():
    global message
    # player = Player()
    # computer = Computer()
    player.defense_success = 1
    player_action = ['attack',player.attack]
    player.attack = 1
    computer_action = computer_beh()
    
    battle(player_action,computer_action)
    message = message_update(player_action,computer_action)
    if player.hp==0 and computer.hp==0:
        return render_template('tie.html', player=player, computer=computer, message=message)        
    if player.hp <= 0:
        return render_template('player_lose.html', player=player, computer=computer, message=message)
    elif computer.hp == 0:
        return render_template('player_win.html', player=player, computer=computer, message=message)
    return render_template('index.html', player=player, computer=computer, message=message)

@app.route('/defend', methods=['POST'])
def defend():
    global message
    # player = Player()
    # computer = Computer()
    dice = random.uniform(0,1)
    if dice > player.defense_success:
        player_action = ['FAIL']
        player.defense_success = 1
    else:
        player_action = ['shield']
        player.defense_success*=0.5
    computer_action = computer_beh()
    
    battle(player_action,computer_action)
    message = message_update(player_action,computer_action)
    if player.hp==0 and computer.hp==0:
        return render_template('tie.html', player=player, computer=computer, message=message)        
    if player.hp <= 0:
        return render_template('player_lose.html', player=player, computer=computer, message=message)
    elif computer.hp == 0:
        return render_template('player_win.html', player=player, computer=computer, message=message)
    return render_template('index.html', player=player, computer=computer, message=message)

@app.route('/charge', methods=['POST'])
def charge():
    global message
    player.defense_success = 1
    player_action = ['charge']
    player.attack+=1
    computer_action = computer_beh()
    battle(player_action,computer_action)
    message = message_update(player_action,computer_action)
    if player.hp==0 and computer.hp==0:
        return render_template('tie.html', player=player, computer=computer, message=message)        
    if player.hp <= 0:
        return render_template('player_lose.html', player=player, computer=computer, message=message)
    elif computer.hp == 0:
        return render_template('player_win.html', player=player, computer=computer, message=message)
    return render_template('index.html', player=player, computer=computer, message = message)



if __name__ == '__main__':
    app.run(port=3877,debug=True)
